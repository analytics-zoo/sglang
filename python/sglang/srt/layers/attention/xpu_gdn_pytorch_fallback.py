"""Pure-PyTorch reference implementations of the GDN decode-path kernels so
we can bisect whether the Triton kernels are corrupting memory on XPU.

Enable individual fallbacks via env vars:
  SGLANG_XPU_GDN_PY_CONV1D=1   -> replace causal_conv1d_update
  SGLANG_XPU_GDN_PY_PACKED=1   -> replace fused_recurrent_gated_delta_rule_packed_decode
  SGLANG_XPU_GDN_PY_UPDATE=1   -> replace fused_sigmoid_gating_delta_rule_update (non-packed decode)

All three take effect in gdn_backend.py forward_decode. They are correctness
references only, not performance-tuned — this is for debugging.
"""
from __future__ import annotations

from typing import Optional, Union

import torch


def causal_conv1d_update_pytorch(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    **_unused,
) -> torch.Tensor:
    """Replacement for causal_conv1d_update (single-token decode version).

    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (num_cache_lines, dim, state_len)
    weight: (dim, width)
    bias: (dim,) optional
    conv_state_indices: (batch,) into conv_state[0]

    For single-token decode (seqlen==1 or x is 2D), performs:
      - shift left conv_state by 1 along state_len dim
      - write x into the last slot
      - compute y = sum over width of conv_state[...] * weight[d, width-1-i]
      - optionally apply activation
    """
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(-1)  # (B, dim, 1)
        squeeze = True
    assert x.dim() == 3
    B, dim, seqlen = x.shape
    assert seqlen == 1, "pytorch fallback only supports single-token decode"
    _, width = weight.shape
    assert conv_state.dim() == 3
    num_cache_lines, _, state_len = conv_state.shape
    assert state_len >= width - 1

    if conv_state_indices is None:
        idx = torch.arange(B, device=x.device, dtype=torch.int64)
    else:
        idx = conv_state_indices.to(torch.int64)

    # Gather per-batch conv_state slabs: (B, dim, state_len)
    state = conv_state.index_select(0, idx).clone()

    # Build full window [state[0], state[1], ..., state[state_len-1], x[0]] and take
    # last `width` elements. This matches the Triton decode kernel, which computes
    # y = state[0]*w[0] + state[1]*w[1] + ... + state[state_len-1]*w[state_len-1]
    #     + x*w[width-1] when state_len == width-1.
    full_window = torch.cat([state, x[:, :, :1]], dim=-1)  # (B, dim, state_len+1)
    conv_window = full_window[:, :, -width:]  # (B, dim, width)
    y = (conv_window * weight.unsqueeze(0)).sum(dim=-1)  # (B, dim)

    # Shift left and append x for new state: new_state = [state[1], ..., state[-1], x[0]]
    new_state = torch.empty_like(state)
    new_state[:, :, :-1] = state[:, :, 1:]
    new_state[:, :, -1] = x[:, :, 0]
    if bias is not None:
        y = y + bias

    if activation is True or activation in ("silu", "swish"):
        y = y * torch.sigmoid(y)
    elif activation is None or activation is False:
        pass
    else:
        raise ValueError(f"Unsupported activation {activation}")

    # Scatter new_state back into conv_state via index_copy_.
    conv_state.index_copy_(0, idx, new_state.to(conv_state.dtype))

    if squeeze:
        return y  # (B, dim)
    return y.unsqueeze(-1)  # (B, dim, 1)


def fused_recurrent_gated_delta_rule_packed_decode_pytorch(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    **_unused,
) -> None:
    """PyTorch reference for fused_recurrent_gated_delta_rule_packed_decode.

    Shapes:
      mixed_qkv: (B, qkv_dim) where qkv_dim = 2*H*K + HV*V
      a, b:      (B, HV)
      A_log:     (HV,)
      dt_bias:   (HV,)
      initial_state: (num_slots, HV, V, K)  -- updated in place
      out:       (B, 1, HV, V) -- written
      ssm_state_indices: (B,) int, -1 means skip/zero
    """
    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]
    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    q_dim = qk_dim // 2
    H = q_dim // K

    softplus_threshold = 20.0

    # Slice mixed_qkv into q (B, H*K), k (B, H*K), v (B, HV*V).
    q_all = mixed_qkv[:, : H * K].view(B, H, K).to(torch.float32)
    k_all = mixed_qkv[:, H * K : 2 * H * K].view(B, H, K).to(torch.float32)
    v_all = mixed_qkv[:, 2 * H * K :].view(B, HV, V).to(torch.float32)

    if use_qk_l2norm_in_kernel:
        q_all = q_all / torch.sqrt((q_all * q_all).sum(dim=-1, keepdim=True) + 1e-6)
        k_all = k_all / torch.sqrt((k_all * k_all).sum(dim=-1, keepdim=True) + 1e-6)
    q_all = q_all * scale

    # Gating.
    A_log_f = A_log.to(torch.float32)
    dt_bias_f = dt_bias.to(torch.float32)
    x = a.to(torch.float32) + dt_bias_f.unsqueeze(0)  # (B, HV)
    softplus_x = torch.where(
        x <= softplus_threshold, torch.log1p(torch.exp(x)), x
    )
    g = (-torch.exp(A_log_f).unsqueeze(0)) * softplus_x  # (B, HV)
    beta = torch.sigmoid(b.to(torch.float32))  # (B, HV)

    # Fully vectorized: no Python-level per-batch loop, no device syncs.
    head_group = HV // H
    idx_long = ssm_state_indices.to(torch.int64)  # (B,)
    # gather states: (B, HV, V, K)
    h_all = initial_state.index_select(0, idx_long.clamp(min=0)).to(torch.float32)
    # Expand q/k along head groups: (B, HV, K)
    q_exp = q_all.repeat_interleave(head_group, dim=1)
    k_exp = k_all.repeat_interleave(head_group, dim=1)
    # h = h * exp(g)
    h_all = h_all * torch.exp(g).unsqueeze(-1).unsqueeze(-1)
    # hk = h * k[:, :, None, :]  → (B, HV, V, K)
    hk = h_all * k_exp.unsqueeze(2)
    v_new = v_all - hk.sum(dim=-1)  # (B, HV, V)
    v_new = v_new * beta.unsqueeze(-1)
    # h = h + v_new[:, :, :, None] * k_exp[:, :, None, :]
    h_all = h_all + v_new.unsqueeze(-1) * k_exp.unsqueeze(2)
    # o = sum(h * q[:, :, None, :], dim=-1) → (B, HV, V)
    o_all = (h_all * q_exp.unsqueeze(2)).sum(dim=-1)

    out.copy_(o_all.unsqueeze(1).to(out.dtype))  # (B, 1, HV, V)

    # Scatter updated states back. For decode all indices are always >= 0.
    initial_state.index_copy_(0, idx_long, h_all.to(initial_state.dtype))


def fused_sigmoid_gating_delta_rule_update_pytorch(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    **_unused,
) -> torch.Tensor:
    """PyTorch reference for the non-packed decode path.

    Shapes:
      q: (1, B, H, K)
      k: (1, B, H, K)
      v: (1, B, HV, V) -- note H=1 or HV per caller
      a, b: (B, HV)
      A_log, dt_bias: (HV,)
      initial_state_source: (num_slots, HV, V, K)
      initial_state_indices: (B,)
      cu_seqlens: (B+1,)
    Returns tensor with shape matching existing decode kernel output: (1, B, HV, V).
    """
    assert q.dim() == 4 and q.shape[0] == 1
    B = q.shape[1]
    H = q.shape[2]
    K = q.shape[3]
    HV = v.shape[2]
    V = v.shape[3]
    softplus_threshold = 20.0
    head_group = HV // H

    # Remove leading seq dim.
    q_bhk = q[0].to(torch.float32)  # (B, H, K)
    k_bhk = k[0].to(torch.float32)  # (B, H, K)
    v_bhk = v[0].to(torch.float32)  # (B, HV, V)

    # L2 norm q, k.
    q_bhk = q_bhk / torch.sqrt((q_bhk * q_bhk).sum(dim=-1, keepdim=True) + 1e-6)
    k_bhk = k_bhk / torch.sqrt((k_bhk * k_bhk).sum(dim=-1, keepdim=True) + 1e-6)
    scale = K ** -0.5
    q_bhk = q_bhk * scale

    A_log_f = A_log.to(torch.float32)
    dt_bias_f = dt_bias.to(torch.float32)
    x = a.to(torch.float32) + dt_bias_f.unsqueeze(0)
    softplus_x = torch.where(
        x <= softplus_threshold, torch.log1p(torch.exp(x)), x
    )
    g = (-torch.exp(A_log_f).unsqueeze(0)) * softplus_x
    beta = torch.sigmoid(b.to(torch.float32))

    out = torch.empty((1, B, HV, V), device=q.device, dtype=q.dtype)

    for n in range(B):
        state_idx = int(initial_state_indices[n].item())
        if state_idx < 0:
            out[0, n].zero_()
            continue
        h = initial_state_source[state_idx].to(torch.float32)  # (HV, V, K)
        q_exp = q_bhk[n].repeat_interleave(head_group, dim=0)  # (HV, K)
        k_exp = k_bhk[n].repeat_interleave(head_group, dim=0)  # (HV, K)
        v_n = v_bhk[n]  # (HV, V)
        g_n = g[n]
        beta_n = beta[n]
        h = h * torch.exp(g_n).view(HV, 1, 1)
        hk = h * k_exp.unsqueeze(1)
        v_new = v_n - hk.sum(dim=-1)
        v_new = v_new * beta_n.unsqueeze(-1)
        h = h + v_new.unsqueeze(-1) * k_exp.unsqueeze(1)
        o = (h * q_exp.unsqueeze(1)).sum(dim=-1)
        out[0, n] = o.to(out.dtype)
        initial_state_source[state_idx] = h.to(initial_state_source.dtype)

    return out
