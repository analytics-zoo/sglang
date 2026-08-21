# hd512 page_attn_decode & XPU Graph TODO

> ⚠️ **状态横幅（2026-07-06）**：本文是 2026-06-30 的 **kernel-bug + piecewise-graph 调查记录**，两点已过时：
> (1) 文末 "Current State" 的 **TPOT=49ms 是融合优化前的旧值**，当前 shippable eager 已到 ~37.7ms。
> (2) 本文的 hd512-ESIMD / piecewise-graph 是为**启用 XPU graph** 服务的，但 XPU graph 现被一个**更底层的
> blocker** 拦住——captured **oneCCL allreduce 在本栈 replay stale**（见 `GEMMA4_XPU_GRAPH_STATUS.md`
> "captured-collective staleness"）。即便 hd512 attention 修好、piecewise 打通，graph decode 仍会
> deadlock。**故本文 TODO 对当前交付（eager）无影响**，仅作 kernel bug 存档；除非先解决 collective 捕获问题，
> 否则不必推进 hd512/piecewise。

## Goal
Enable XPU Graph capture for gemma4-31B decode by replacing `sgl_kernel.fwd` (flash attention using SLM/work_group_scratch) with ESIMD `page_attn_decode` (no SLM, graph-capturable).

- hd256 (sliding layers, 50/60): **DONE** — GQA=2 kernel working correctly (cosine=1.0)
- hd512 (global layers, 10/60): **BLOCKED** — kernel produces wrong results

## What Works
- `page_attn_decode` with headDim=256 (original kernel) — correct for GQA=2,4,8
- All changes to eagle.sycl dispatch (GQA=2 path, hd512 path, TORCH_CHECK relaxed)
- Phase2 channelOffset fix for hd512 (`& 0x1f`, `>> 5`)

## The Bug
When `constexpr uint32_t headDim = 512` (only change from working hd256 copy), Phase1 cannot read K values for tokens with index > 0.

### Symptoms
- `seq_len=1`: correct (only token 0)
- `seq_len=2, equal K[dim0] for both tokens`: output = V[token0] only (0.3 instead of 0.5)
- `seq_len=64, all equal K`: output = V[token0] only (not mean of all)
- `V[0]=V[1]` test passes (0.42) → Phase2 V-loading is correct for all tokens
- K at any dim (0, 32, 128, 256, 384, 511) for token 0: works ✓
- K at any dim for token 1+: invisible (score = 0)

### Root Cause Analysis
The K gather `lsc_gather<uint32_t, 4, ...>((uint32_t*)kState, simdOffsetsK, mask)` for lane 1+ returns 0.

`simdOffsetsK[i] = i * headKv * headDim * sizeof(T) + offsetBaseK + ...`

For headDim=512: `simdOffsetsK[1] = 1 * headKv * 1024 + ...` (token 1 byte offset).

- With headDim=256 (constexpr), token stride = `headKv * 512`: **works**
- With headDim=512 (constexpr), token stride = `headKv * 1024`: **broken**
- With headDim=256 + num_kv_heads=2 (runtime token stride = 1024): **works**

This points to the **constexpr headDim=512 causing a compile-time miscomputation** — possibly in how the ESIMD compiler generates the SIMD multiply `baseOffsetInc16AsSimd * (headKv * headDim * sizeof(T))` when `headDim*sizeof(T)=1024`. The same runtime value (1024) works when achieved via `headKv=2, headDim=256`.

### What Was Tried
1. volatile tokenStrideBytes → no effect
2. Per-kk Q load (reduce GRF pressure) → no effect
3. Full rewrite from GQA2 base (single-thread full dot) → same bug
4. Mechanical scaling of all simd sizes (128×4 qqFp32, 16×128 kkCache) → same bug

### Hypothesis (Unverified)
The ESIMD compiler generates incorrect code for the gather offset when `headDim` as a constexpr > 256 participates in a SIMD vector multiplication. This may be:
- A compiler constant-folding bug specific to large compile-time constants in SIMD multiply
- An issue with how `sizeof(T)` interacts with the constexpr in the multiplication chain
- A register allocation issue masked as data corruption

### What's Needed to Fix
1. **IGC (Intel Graphics Compiler) ISA dump** of the hd512 kernel to verify the actual instructions for `simdOffsetsK` computation
2. Or: **kernel printf** to dump `simdOffsetsK[0]` and `simdOffsetsK[1]` at runtime
3. Or: **pass token stride as a runtime kernel parameter** (not computed from constexpr headDim inside kernel) — this would bypass the compiler issue

### Key Finding: vLLM Also Doesn't Support hd512
vLLM's `flash_attn.py:1238` gates page_attn_decode with `self.head_size == 256`. vLLM also uses flash attention (SLM-based) for gemma4 global layers (head_dim=512). **The page_attn_decode kernel was never designed for head_dim=512.**

### Alternative Approaches
- Cannot split head_dim=512 into 2×head_dim=256 (softmax is per-full-score)
- **Piecewise/breakable graph** (capture SWA layers in graph, global layers run eager between graph segments) — most viable path, needs sglang `SGLANG_USE_BREAKABLE_CUDA_GRAPH` enabled for XPU
- Write a new hd512 decode attention kernel from scratch (not just patching the hd256 one — the interleaved 4-thread tiling pattern doesn't trivially scale)
- Use `torch.nn.functional.scaled_dot_product_attention` as fallback — also not graph-capturable on XPU (uses events)

## Files
- `/workspace/custom-esimd-kernels-sglang/csrc/eagle/page.attn.hd512.h` — current broken hd512 kernel
- `/workspace/custom-esimd-kernels-sglang/csrc/eagle/page.attn.h` — working hd256 reference
- `/workspace/custom-esimd-kernels-sglang/csrc/eagle/eagle.sycl` — dispatch (hd512 block at line ~275)
- `/llm/workspace/sgl_gemma/sglang/python/sglang/srt/layers/attention/xpu_backend.py` — Python integration (line ~1012)

## Current State
- SWA layers (50/60, head_dim=256): using ESIMD page_attn_decode ✓ (graph-capturable)
- Global layers (10/60, head_dim=512): using flash_attn_with_kvcache (has SLM, blocks graph)
- Server TPOT = 49ms, gsm8k = 0.975 (correct without hd512 ESIMD)
- XPU Graph capture fails with "sycl_ext_oneapi_work_group_scratch_memory not available" due to the 10 global layers

## TODO: Piecewise Graph (In Progress)

### Done
- `eager_on_graph(True)` wrapper on `_flash_attn_decode_eager` in xpu_backend.py
- XPU shim: added `torch.cuda.stream`, `torch.cuda.current_stream`, `torch.cuda.synchronize` aliases
- Fixed `BreakableCUDAGraphCapture._begin_new_segment`: catch TypeError for XPU's `capture_begin()` (no `capture_error_mode` kwarg)
- Launch with `--enable-breakable-cuda-graph` + `SGLANG_XPU_ENABLE_GRAPH=1` + `SGLANG_USE_BREAKABLE_CUDA_GRAPH=1`

### Current Blocker
Graph capture fails at the FIRST Triton kernel (e.g. `gemma_qkv_rmsnorm` for global hd512 layers) which executes BEFORE the `_flash_attn_decode_eager` break point. Error: "wait method cannot be used for an event associated with a command graph" = Triton kernel uses SLM.

### Remaining Work
The breakable graph only helps if ALL kernels before the first break point are SLM-free. In gemma4 decode, the forward order is:
```
Layer 0 (SWA, hd256):
  input_layernorm → sgl_kernel rmsnorm (no SLM) ✓
  qkv_proj → ESIMD FP8 GEMV (no SLM) ✓
  esimd_qkv_split_norm_rope (no SLM) ✓
  page_attn_decode ESIMD (no SLM) ✓
  o_proj → ESIMD FP8 GEMV ✓
  post_attn_layernorm → sgl_kernel rmsnorm ✓
  esimd_fused_add_rms_norm ✓
  gate_up_proj → ESIMD FP8 GEMV ✓
  gelu → elementwise ✓
  down_proj → ESIMD FP8 GEMV ✓
  esimd_rmsnorm_residual_scalar ✓
...
Layer 5 (Global, hd512):
  input_layernorm → sgl_kernel rmsnorm ✓
  qkv_proj → ESIMD FP8 GEMV ✓
  *** gemma_qkv_rmsnorm (Triton, uses SLM!) ← BREAKS HERE ***
  flash_attn (would have broken via eager_on_graph, but never reached)
```

Fix options:
1. Replace `gemma_qkv_rmsnorm` for global layers with sgl_kernel rmsnorm (3 separate calls) — eliminates last Triton kernel in decode
2. Wrap the ENTIRE global-layer self_attn call with `eager_on_graph` so it breaks before qkv_rmsnorm
3. Disable fused qkv_rmsnorm for global layers (use per-norm fallback which calls sgl_kernel rmsnorm, no SLM)

Option 3 is simplest: set `SGLANG_GEMMA4_DISABLE_FUSED_QKV_NORM=1` or gate the fused path on `head_dim == 256` (same as ESIMD qkv eligibility).
