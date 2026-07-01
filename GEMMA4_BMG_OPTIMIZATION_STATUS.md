# Gemma4-31B FP16 FP8 Decode Optimization Status (BMG TP=2)

## ⚠️ STATUS (2026-07-01): XPU GRAPH IS BROKEN — SHIP EAGER FOR NOW

The "XPU Graph" column below was measured on a config whose decode output is
**garbled** (see "Known Issue #2: XPU Graph decode garble" below). Those graph
TPOT/accuracy numbers are NOT valid — they were taken before correctness was
checked. Do NOT enable XPU graph until the allreduce fix lands.

**Currently shippable = fp16 + ESIMD + intel_xpu + layered_fp8, GRAPH OFF (eager)**,
verified 2026-07-01: gsm8k chat 0.950 (38/40, 0 invalid), coherent output, and
BETTER TPOT-vs-ctx than the (broken) graph at every length:

| Metric | bf16 baseline | EAGER fp16+ESIMD (SHIPPABLE) | fp16+ESIMD+XPU Graph (BROKEN — garbles) |
|--------|--------------|------------------------------|------------------------------------------|
| TPOT (1K ctx) | 65.5ms | **40.5ms** | 45.8ms |
| TPOT (4K ctx) | — | **46.2ms** | 72.0ms |
| TPOT (8K ctx) | — | **54.8ms** | 106.9ms |
| gsm8k accuracy | 0.990 | **0.950** (chat n=40) | ~0 (garbled decode) |
| Decode tok/s (1K) | 15.3 | 24.7 | 21.8 |

A *working* graph should beat eager (removes ~420 kernel host-launches/step);
"graph slower" here is purely a symptom of the allreduce bug. Fix in progress
(Option A: piecewise capture with collectives kept eager).

## Completed Optimizations

1. **sgl-kernel-xpu fp16 FMHA dispatch** — 3 `.cpp.in` files add fp16 template instantiation (root cause fix for fp16 broken attention)
2. **ESIMD `qkv_split_norm_rope`** — fused QKV split + Q/K RMSNorm + RoPE for sliding layers (50/60)
3. **ESIMD `fused_add_rms_norm`** — fused residual_add + pre_ff_norm
4. **ESIMD `rmsnorm_residual_scalar`** — new kernel replacing Triton `_gemma_rmsnorm_residual_kernel`
5. **ESIMD `page_attn_decode` GQA=2** — SLM-free decode attention for SWA layers (graph-capturable)
6. **Split-K decode attention (hd512)** — SLM-free decode attention for global layers (graph-capturable), from `/home/intel/xiangyu/hd512_decode/splitk_kernel.h`
7. **XPU Graph capture** — full decode forward captured into XPU graph (allreduce included). ⚠️ BROKEN at TP>1: see Known Issue #2. Correct only at TP=1 (no collectives) which we don't run.
8. **Vision tower CPU offload** — `SGLANG_SKIP_VISION_GPU=1` keeps vision tower on CPU for text-only
9. **SWA ratio tuning** — `--swa-full-tokens-ratio 0.05` maximizes full-pool for long context

## Known Issue: TPOT Linear Growth with Context Length

### Problem
TPOT grows linearly with context length due to 10 global attention layers (head_dim=512) doing full O(n) attention over entire KV history:

```
1K ctx:  47.7ms
4K ctx:  73.9ms  (+26ms for +3K tokens)
8K ctx: 108.8ms  (+35ms for +4K tokens)
```

~8.7ms per 1K additional context tokens. This is **abnormally high** — 10 layers × split-K with G=4 splits over the full KV is highly inefficient for the XPU graph captured configuration.

### Root Cause
The split-K kernel is launched with `max_seq = page_table.shape[1] * page_size = 70016` at graph capture time. This means:
- Grid = `B * HQ * G = 1 * 16 * 4 = 64` workitems (fixed, OK)
- But `chunk = (70016 + 4 - 1) / 4 = 17504` tokens per split
- **CORRECTION (2026-07-01):** the earlier claim "each WI loops over 17504 tokens (early-exits per token)" is WRONG. The kernel sets `end = min(start+chunk, kvSeqLen)` BEFORE the loop, so split g=0 loops exactly `kvSeqLen` (real ctx) times, splits g=1..3 hit `start>=end` and return immediately. Cost is O(ctx) on a SINGLE work-item, not 17504.
- The large chunk means only split 0 ever has work: for any real ctx < 17504, split 0 does ALL tokens serially, splits 1-3 idle → split-K degenerates to G=1 → poor occupancy (16 of 64 WIs active).
- **All work lands on a single WI per q-head** until context exceeds chunk size (17.5K).
- VERIFIED by kernel microbench (`/home/intel/xiangyu/hd512_decode/bug_repro.cpp`, gemma4 global shape, all cos=1.0000): per-global-layer time PROD(max=70016,G=4) grows linearly 451us@1K → 7234us@8K; stopgap G=64 plateaus ~1050us; in-kernel-chunk fix + best-G ~30x faster. E2E A/B (same server, only SGLANG_SPLITK_G flipped): G=4 = 45.8/72.0/106.9ms @ 1K/4K/8K, G=64 = flat ~49ms. NOTE this A/B was on the GARBLED graph path — valid for TPOT (ignore_eos runs identical kernels) but the config is not correctness-valid; see Known Issue #2.

### Potential Fixes (TODO) — only relevant once graph is fixed (Known Issue #2); moot for eager (eager uses all 4 splits, chunk≈ctx/4)
1. **In-kernel chunk (BEST, graph-safe)** — compute `chunk=(seqLens[b]+G-1)/G` INSIDE `phaseSplit` (kernel already reads `seqLens[b]`; grid `B*HQ*G` is independent of chunk). All G splits become active regardless of the graph-baked `max_seq`. Needs esimd pkg rebuild. Raise G to ~32.
2. **Increase G (num_splits), stopgap** — env `SGLANG_SPLITK_G=64` (added, `xpu_backend.py`). chunk=70016/64=1094; flattens the curve, no rebuild. Reduce overhead at short ctx (+2.8ms @1K).
3. **Dynamic G based on actual seq_len** — incompatible with XPU graph (fixed at capture time).

## Known Issue #2: XPU GRAPH DECODE GARBLE (TP>1) — ROOT CAUSE = captured oneCCL allreduce replays STALE

**Symptom:** with `SGLANG_XPU_ENABLE_GRAPH=1` at TP=2, decode output is garbled: token 0
is correct (logprob −0.00) then token 1+ degenerate ("Paris ( ( (", "46 a a a"). gsm8k ≈ 0.
Eager (graph off) with the SAME kernels is correct (gsm8k 0.950). Localized 2026-07-01.

**Root cause (isolated, proven — not attention):** `dist.all_reduce` (oneCCL/xccl) captured
inside `torch.xpu.XPUGraph` does NOT replay correctly — on the 2nd+ replay it returns
stale/lagged results (mixes current-rank + stale-other-rank data). TP=2 gemma4 decode
captures ~120 allreduces/step (after every o_proj + down_proj across 60 layers); token 0 is
the first replay (OK), token 1+ reuse stale reductions → every layer's TP output corrupts.

Ruled OUT by isolation tests (`/home/intel/xiangyu/cc_workspace/gemma_splitk/test_*.py`):
- torch compute op captured + replayed 3× w/ mutated input → fresh every time (correct).
- split-K ESIMD kernel captured + replayed w/ in-place-mutated seqlen/q → cos 1.0000 (correct).
- `dist.all_reduce` captured + replayed w/ fresh inputs → STALE (out = 3,3,5,10,13 vs expect
  3,3,6,9,12). `dist.barrier()`+`synchronize()` around replay does NOT fix it.
- FIX DIRECTION VALIDATED: capture compute in segments, run `dist.all_reduce` EAGER between
  `g1.replay()` / `g2.replay()` → correct every step (piecewise graph, collectives eager).

Also refuted (each tested, only perturbed the garble): fresh `torch.empty` attention out/scratch
buffers (made persistent via `_decode_buf`); `swa_page_table` / `swa_out_cache_loc` freshly
allocated (made persistent+in-place — kept, they ARE latent graph bugs); split-K two-phase
in-order race (added explicit `h.depends_on(evA)` in `splitk_decode.h` + rebuilt eagle_ops —
no fix, because split-K captures fine in isolation).

**Fix (Option A, in progress):** piecewise capture keeping collectives eager. sglang has
`breakable_cuda_graph` (`break_graph`) but it is CUDA-only (`cuda.bindings.runtime`); needs an
XPU port using `torch.xpu.XPUGraph.capture_begin()/capture_end()` (both exposed), breaking the
captured decode forward at each `tensor_model_parallel_all_reduce` and running the collective
eager between segments. Alternative (Option B): build+wire the sgl-kernel-xpu custom SYCL
allreduce (`python/sgl_kernel/allreduce.py`, currently NOT built on this XPU) and verify it
captures like split-K did.

**vLLM-xpu note:** `vllm/platforms/xpu.py:255` claims captured allreduce works under
FULL_DECODE_ONLY, but that's vLLM's torch.compile + CUDAGraphWrapper + `splitting_ops` path
(FX-ordered), "validated on Qwen TP=4" not gemma4 — does NOT transfer to sglang's raw
`torch.xpu.graph` monolithic capture.

## Launch Configuration

```bash
export SGLANG_XPU_ENABLE_GRAPH=1
export SGLANG_SKIP_VISION_GPU=1
export ZE_AFFINITY_MASK=0,1
export SGLANG_USE_SGL_XPU=1
export SGLANG_FP8_IGNORED_LAYERS=vision_tower,embed_vision
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLTOALL_TMP_BUF=1

python3 -m sglang.launch_server \
  --model-path /llm/models/gemma-4-31B-it \
  --device xpu --tp 2 --quantization fp8 --dtype float16 \
  --load-format layered_fp8 --attention-backend intel_xpu \
  --page-size 64 --mem-fraction-static 0.85 \
  --swa-full-tokens-ratio 0.05 \
  --chunked-prefill-size 1024 \
  --disable-radix-cache \
  --max-running-requests 1 \
  --context-length 70000 \
  --cuda-graph-bs 1 \
  --skip-server-warmup \
  --trust-remote-code --model-impl sglang \
  --host 0.0.0.0 --port 30000
```

## Files Modified

### sgl-kernel-xpu (kernel fixes)
- `src/sycl/xe_fmha_fwd_prefill_kernel.cpp.in` — fp16 dispatch
- `src/sycl/xe_fmha_fwd_decode_kernel.cpp.in` — fp16 dispatch
- `src/sycl/xe_fmha_fwd_split_decode_kernel.cpp.in` — fp16 dispatch

### custom-esimd-kernels-sglang (new kernels)
- `csrc/xpu/esimd_kernels/rmsnorm_residual_scalar.h` — new ESIMD kernel
- `csrc/eagle/page.attn.gqa2.h` — GQA=2 page_attn_decode
- `csrc/eagle/splitk_decode.h` — split-K decode attention (hd256+hd512)
- `csrc/eagle/eagle.sycl` — dispatch + torch op registration
- Various headers ported from vLLM: `fp16_GEMV.h`, `norm_add_norm.h`, etc.

### sglang (model + backend)
- `python/sglang/srt/layers/attention/xpu_backend.py` — ESIMD page_attn + split-K integration, XPU graph metadata, SWA max_seq fix
- `python/sglang/srt/models/gemma4_causal.py` — ESIMD QKV/norm/residual-scalar fusions, Triton fused qkv_norm gate (head_dim==256 only)
- `python/sglang/srt/models/gemma4_mm.py` — decode PLE boolean-indexing skip for graph
- `python/sglang/srt/model_executor/cuda_graph_runner.py` — XPU stream/sync/graph patches, try/except in capture
- `python/sglang/srt/model_executor/breakable_cuda_graph/breakable_cuda_graph.py` — XPU capture_begin fix
- `python/sglang/srt/model_loader/loader.py` — SGLANG_SKIP_VISION_GPU
- `python/sglang/srt/layers/linear.py` — SGLANG_SKIP_ALLREDUCE (debug only)
- `python/sglang/srt/layers/logits_processor.py` — esimd_gemv_fp16 (disabled, oneDNN faster for lm_head)
