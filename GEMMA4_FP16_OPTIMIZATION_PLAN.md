# Gemma4-31B FP16 Decode Optimization Plan (BMG TP=2)

## Baseline (2026-06-30)

| Config | TPOT | gsm8k | Notes |
|--------|------|-------|-------|
| bf16 (overlap ON) | 65.5ms | 0.975 | Previous best |
| fp16 (kernel fix only) | 56.15ms | 0.975 | sgl-kernel-xpu fp16 dispatch fix |
| fp16 + Step 1 (ESIMD QKV) | 53.68ms | — | + esimd_qkv_split_norm_rope |
| **fp16 + Step 1+2 (QKV + fused norm)** | **52.27ms** | **0.980** | + esimd_fused_add_rms_norm |

Decode profile (pure-decode, 12 steps, fp16 native):
- ESIMD FP8 GEMV: **50.4%** (2880 calls, M=1)
- oneCCL allreduce: **28.8%** (1452 calls, TP comm)
- oneDNN bf16 GEMM (lm_head): **9.3%** (12 calls = 1/step, 4.7ms each)
- SYCL elem/copy: 4.5%
- RMSNorm (SYCL sgl_kernel): **2.9%**
- FMHA attention: 1.7%
- Triton gemma norm (qkv_rmsnorm + residual_scalar): **1.1%**
- Activations (gelu etc): 1.0%
- RoPE: 0.2%

## Available ESIMD Ops (container `custom_esimd_kernels_sglang`)

```
esimd_fused_add_rms_norm          # residual_add + rmsnorm in-place
esimd_fused_add_rms_norm_batched  # batched variant
esimd_qkv_split_norm_rope         # QKV split + Q/K RMSNorm + RoPE
esimd_norm_gemv_fp8_pert          # rmsnorm + FP8 GEMV fused
esimd_resadd_norm_gemv_fp8_pert   # residual_add + rmsnorm + FP8 GEMV fused
esimd_resadd_norm_gemv2_fp8_pert  # residual_add + rmsnorm + 2x FP8 GEMV fused
esimd_gemv_fp8_pert_fused2        # 2x FP8 GEMV fused
esimd_gemv_fp8_pert_fused3        # 3x FP8 GEMV fused
esimd_rms_norm_gated              # gated rmsnorm (for GDN, not gemma4)
```

## Optimization Steps

### Step 1: `esimd_qkv_split_norm_rope`

**What:** Replace Triton `gemma_qkv_rmsnorm` + separate RoPE with a single ESIMD kernel that does QKV split + per-head Q/K RMSNorm + RoPE in one launch.

**Where:** `gemma4_causal.py` → `Gemma4Attention.forward()`, lines 415-498.

**Constraints from vLLM:**
- Gated `head_dim == 256` only (kernel hardcoded); global attention layers (head_dim=512) must fallback
- Requires fp16 input (we now run fp16 ✓)
- Norm weight convention: kernel expects `w - 1.0` (Qwen convention); gemma4 stores raw weight → must precompute `q_norm.weight - 1.0` and `k_norm.weight - 1.0`
- V norm has `with_scale=False` (weight=1) → pass zeros to kernel (kernel adds 1.0 internally)
- `partial_rotary_factor=0.25` for gemma4 sliding layers → the rope_dim arg to kernel = `head_dim * 0.25 = 64`

**Covers:** 50/60 layers (sliding attention, head_dim=256). 10 global layers (head_dim=512) stay on the current path.

**Expected gain:** ~1.5-2% decode time (eliminates Triton qkv_rmsnorm 1.1% + RoPE 0.2% for 83% of layers + reduces launch count by ~100/step)

---

### Step 2: `esimd_fused_add_rms_norm`

**What:** At decode M=1, fuse `hidden_states + residual → residual; rmsnorm(residual) → hidden_states` into one ESIMD kernel call, replacing the current `sgl_kernel.fused_add_rmsnorm` (SYCL) + the Triton `gemma_rmsnorm_residual_scalar`.

**Where:** `gemma4_causal.py` → `Gemma4DecoderLayer.forward()`:
- Line 678/723: `self.pre_feedforward_layernorm(hidden_states, residual)` — this is `RMSNorm.forward_xpu` which calls `sgl_kernel.fused_add_rmsnorm`
- Lines 728-742: `gemma_rmsnorm_residual_scalar` (Triton, fused post_ff_norm + residual + scalar)

**API (from container):**
```python
from custom_esimd_kernels_sglang import esimd_fused_add_rms_norm
# Signature: esimd_fused_add_rms_norm(x, residual, weight, eps)
# In-place: x = rmsnorm(x + residual), residual = x + residual
```

**Constraints:**
- Decode M=1 only, fp16, contiguous
- The `layer_scalar` multiplication (gemma4-specific) needs to be handled separately or folded in

**Expected gain:** ~1-2% (reduces norm kernel launches from ~4/layer to ~2/layer for 60 layers)

---

### Step 3: `esimd_resadd_norm_gemv_fp8_pert` — DEFERRED

**Status:** Investigated but deferred. The available kernels have constraints:
- `esimd_norm_gemv_fp8_pert` is GDN-specific (gated norm, needs x + z inputs)
- `esimd_resadd_norm_gemv_fp8_pert` does residual_add + norm + GEMV but requires the FP8 weight in `[N, K]` format (transposed from sglang's `[K, N]` storage) — the `_esimd_t` cache in fp8_utils.py provides this but is only created lazily on first forward. Also requires restructuring MLP call path to extract gate_up_proj from MLP.forward().
- `esimd_gemv_fp8_pert_fused2/3` can batch 2-3 GEMVs sharing one input read, but gemma4 doesn't have consecutive GEMVs without intermediate ops (norms/activations between them).

**Conclusion:** The remaining ~4% norm overhead is now split across many small kernels that already individually run fast. Further fusion needs either XPU graph capture (bundles all launches) or custom kernels not in this container.

---

### Future Opportunities (not achievable with current container ops)

| # | Op | What | Est. gain | Blocker |
|---|---|---|---|---|
| 4 | `esimd_gemv_fp16` for lm_head | Replace oneDNN bf16 GEMM [1,5376]×[5376,131072] | ~9% (4.7ms/step) | Op not in `custom_esimd_kernels_sglang` |
| 5 | XPU Graph decode capture | Bundle all decode-step kernels into one launch | ~5-10% (kills 38% inter-op gap) | Needs SGLANG_XPU_ENABLE_GRAPH validation for gemma4 |
| 6 | `esimd_norm_add_norm` | post_attn + pre_ff double-norm in one kernel | ~1% | Op not in container (vLLM has it) |
| 7 | `esimd_fused_scaled_add_rms_norm` | Cross-layer: defer scalar×add to next layer's input_norm | ~0.5% | Op not in container |
| 8 | Allreduce overlap with compute | Overlap TP comm with non-dependent kernels | ~5-10% | Algorithmic; ⚠️ **修正 2026-07-06**：`overlap_schedule` 实测在 bsz=1 **decode 上有收益（TPOT −1.4ms/~3.7%）**，对 prefill(TTFT) 反而略有负担（见 STATUS 文档 "Overlap Schedule A/B"）。此处旧论断"helps prefill not decode"已被推翻。保持 overlap ON（默认）。 |
