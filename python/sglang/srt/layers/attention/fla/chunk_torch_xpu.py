"""Pure-PyTorch fallback for chunk_gated_delta_rule on Intel XPU.

Triton-XPU 3.7.0 cannot compile the block-pointer-heavy GDN kernels in
chunk_delta_h.py / chunk_fwd.py / chunk_o.py (TritonIntelStrideVersioning and
downstream TTGIR passes fail). This module provides an eager reference
implementation that is semantically equivalent to chunk_gated_delta_rule, at the
cost of performance — one loop iteration per token, no fusion. The goal is
correctness so Qwen3.5 (and other GDN-based hybrid models) can at least run
end-to-end on XPU while the Triton backend catches up.
"""

from typing import Optional, Tuple

import torch


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Match the behavior of fla/l2norm.py (normalize last dim, float32 reduction).
    orig_dtype = x.dtype
    x32 = x.to(torch.float32)
    denom = x32.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps).sqrt()
    return (x32 / denom).to(orig_dtype)


def chunk_gated_delta_rule_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, None, Optional[torch.Tensor]]:
    """Drop-in for sglang.srt.layers.attention.fla.chunk.chunk_gated_delta_rule.

    Shapes (head_first=False, the only mode sglang uses today):
      q, k : [1, T_total, H_k, K]   (T_total = sum of sequence lengths when cu_seqlens given)
      v    : [1, T_total, H,   V]
      g    : [1, T_total, H]        — log-space decay
      beta : [1, T_total, H]
      initial_state         : [N_slots, H, V, K]  — full state pool
      initial_state_indices : [N_seqs]            — per-sequence slot index
      cu_seqlens            : [N_seqs + 1]
    Returns:
      (o, None, last_recurrent_state)  where
        o                    : [1, T_total, H, V]
        last_recurrent_state : [N_seqs, H, V, K]   — callers are responsible for
                               scattering this back into initial_state at
                               initial_state_indices.
    """
    assert not head_first, "head_first=True is not supported in the XPU fallback."
    assert q.shape[0] == 1, "XPU fallback expects batch=1 with cu_seqlens."
    assert (
        cu_seqlens is not None
    ), "XPU fallback currently requires cu_seqlens (sglang always provides it)."

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)

    T_total, H_k, K = q.shape[1], q.shape[2], q.shape[3]
    _, _, H, V = v.shape
    assert H % H_k == 0, f"H ({H}) must be a multiple of H_k ({H_k}) for GQA."
    repeat = H // H_k

    if scale is None:
        scale = K ** -0.5

    # Work in float32 on the accumulator, cast output back to input dtype.
    out_dtype = q.dtype

    # Drop the leading batch dim for convenience.
    q = q[0]      # [T_total, H_k, K]
    k = k[0]      # [T_total, H_k, K]
    v = v[0]      # [T_total, H,   V]
    g = g[0]      # [T_total, H]
    beta = beta[0]  # [T_total, H]

    o = torch.empty((T_total, H, V), dtype=out_dtype, device=q.device)

    n_seqs = cu_seqlens.numel() - 1
    last_state = torch.empty(
        (n_seqs, H, V, K), dtype=torch.float32, device=q.device
    )

    cu = cu_seqlens.tolist()
    idx_list = (
        initial_state_indices.tolist() if initial_state_indices is not None else None
    )

    for s in range(n_seqs):
        t0, t1 = cu[s], cu[s + 1]

        if initial_state is not None:
            # Two shapes are possible depending on the caller:
            #   idx_list given   -> initial_state is the full state pool
            #                       [N_slots, H, V, K], pick slot idx_list[s].
            #   idx_list is None -> caller already sliced with cache_indices
            #                       so initial_state is [N_seqs, H, V, K].
            slot = idx_list[s] if idx_list is not None else s
            state = initial_state[slot].to(torch.float32).clone()  # [H, V, K]
        else:
            state = torch.zeros((H, V, K), dtype=torch.float32, device=q.device)

        for t in range(t0, t1):
            q_t = q[t].to(torch.float32)                     # [H_k, K]
            k_t = k[t].to(torch.float32)                     # [H_k, K]
            v_t = v[t].to(torch.float32)                     # [H, V]
            g_t = g[t].to(torch.float32)                     # [H]
            b_t = beta[t].to(torch.float32)                  # [H]

            # Expand GQA: replicate q/k heads to full V head count.
            q_full = q_t.repeat_interleave(repeat, dim=0)    # [H, K]
            k_full = k_t.repeat_interleave(repeat, dim=0)    # [H, K]

            # state *= exp(g_t)  (per-head decay in log space).
            decay = torch.exp(g_t).view(H, 1, 1)
            state = state * decay

            # Delta-rule update:
            #   v_pred_h = state_h @ k_h         -> [H, V]
            #   v_err_h  = v_h - v_pred_h        -> [H, V]
            #   state_h += beta_h * outer(v_err_h, k_h)
            v_pred = torch.einsum("hvk,hk->hv", state, k_full)
            v_err = v_t - v_pred
            update = torch.einsum("hv,hk->hvk", v_err, k_full)
            state = state + b_t.view(H, 1, 1) * update

            # Output: o_h = scale * state_h @ q_h   -> [H, V]
            o_t = scale * torch.einsum("hvk,hk->hv", state, q_full)
            o[t] = o_t.to(out_dtype)

        last_state[s] = state

    o = o.unsqueeze(0)  # [1, T_total, H, V]
    # Return contract: (o, last_recurrent_state, h_aux). The caller in
    # gdn_backend.forward_extend scatters last_recurrent_state back into
    # ssm_states[cache_indices] on non-CUDA backends; returning it as the
    # middle element (not as h_aux) is required for that scatter to run.
    return o, last_state, None
