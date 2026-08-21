# Gemma4-31B XPU Graph — Current Status (archive, 2026-07-01)

Scope: the XPU-graph decode path for **gemma-4-31B-it, TP=2, `--attention-backend
intel_xpu`, fp16 + ESIMD + layered_fp8** on Intel BMG (2 cards, GPU 0,1), container
`txytest_sgl_bmg`.

## TL;DR

- **Shippable today = EAGER** (`SGLANG_XPU_ENABLE_GRAPH` off): gsm8k 0.950, coherent,
  TPOT 40.5 / 46.2 / 54.8 ms @ 1K / 4K / 8K ctx. Recommended config.
- **XPU graph = NOT usable yet.** After fixing a real garble bug it produces
  *correct* output on the graph, but decode **deadlocks after 1–2 requests**
  (GPU pinned ~95%, hard busy-spin in the oneCCL collective). Cannot run gsm8k /
  perf on the graph path in this state.
- This whole intel_xpu graph-decode path is **new local work on `dev-bmg-gemma`**
  (commits `2e811f49d3`, `6eaf848187`). It is **not** on `origin/dev-bmg` or
  `origin/dev-bmg-gemma`. Nothing pushed. There is **no validated upstream
  reference** for "gemma4 + intel_xpu + XPU-graph decode".

## What was FIXED (correct, kept, default-on when graph is enabled)

**SWA page-table build-order bug in `init_forward_metadata_out_graph`
(`xpu_backend.py`).**
`token_to_kv_pool.translate_loc_from_full_to_swa(idx)` maps full-pool KV **slot**
(token-granularity) indices → SWA-pool slots. Order matters:

- **Eager** (`init_forward_metadata`, correct): build token-granularity page_table →
  translate to SWA → THEN stride + `//page_size`.
- **Graph** (was buggy): built page_table **already** `//page_size`, then fed those
  page-divided indices into the slot-index map → wrong SWA page table → the **50
  sliding layers** (of 60) read the wrong KV → decode garbled from **token 1**
  (token 0 correct because it comes from prefill).

Fix: in the graph builder, translate the **raw** `req_to_token` slots first, then
stride + `//page_size` (mirrors eager). Result: first-ever coherent graph output —
`capital of France → "Paris"`, `17×24 → "408"`, `count 1..15 → "1,2,…"`; log
confirms `cuda graph: True` (genuinely the captured path, not a fallback).

## What is STILL BROKEN: decode deadlock under load

Symptom: with `SGLANG_XPU_ENABLE_GRAPH=1`, the first 1–2 requests return correct
answers, then a later request hangs — **GPU 0,1 pinned ~95%**, schedulers alive,
`0`/few "Decode batch" logged before the hang, **no watchdog / no exception** =
a hard **busy-spin inside the collective**, not a watchdog kill.

Key scoping facts (all measured this session):
- Happens with **breakable graph ON** (allreduce run eager between segments) **and**
  with plain **monolithic capture** (`BREAKABLE=0`, allreduce captured). So it is
  **not** caused by the breakable machinery — it is inherent to
  capturing/driving the oneCCL collective on sglang's raw `torch.xpu.graph` path.
- Toggling the `CCL_SYCL_*` overrides (ALLREDUCE/REDUCE_SCATTER/ALLGATHERV
  `_SIMPLE_THRESHOLD`, `ALLTOALL_TMP_BUF`) **on or off does not fix it** (verified
  container base env does not set them, so "off" is real). Earlier "CCL_SIMPLE=0
  survived 2 requests" was timing variance, not a real effect.

### Root-cause evidence (captured collective replays stale)

Isolated primitive tests on GPU 0,1 (`/home/intel/xiangyu/cc_workspace/gemma_splitk/test_*.py`):

| Test | Result |
|------|--------|
| torch compute op captured + replayed 3× w/ mutated input | ✅ fresh every replay |
| split-K ESIMD attn captured + replayed w/ mutated seqlen/q | ✅ cos 1.0000 |
| indexed KV scatter+gather, embedding gather, mutated index | ✅ fresh |
| **`dist.all_reduce` captured in `torch.xpu.graph`** (default stream) | ❌ **STALE**: replay 2+ returns lagged/mixed data (e.g. 3,3,6,9 vs expect 3,6,9,12) |
| same, side stream / out-of-place clone form | ❌ still stale |
| segments + `dist.all_reduce` run **eager** between replays | ✅ correct (basis of the breakable experiment) |

So: torch ops, ESIMD attn, KV/embedding all capture+replay correctly; **only the
oneCCL collective does not** on raw `torch.xpu.graph`. In TP=2 decode there are
~120 all-reduces/step (after every o_proj + down_proj × 60 layers).

## Why vLLM-xpu / dev-bmg / the Qwen server don't hit this

- The **Qwen server** currently running on GPU 2,3 uses **`--disable-cuda-graph`** —
  it does not capture at all (my earlier assumption that it "uses graph and is fine"
  was wrong).
- **`origin/dev-bmg` has no `intel_xpu` graph-decode path** (`init_forward_metadata_out_graph`
  absent). It never runs this path.
- **vLLM-xpu** runs gemma4 on XPU graph, but via a **different stack**: torch.compile
  + inductor + `CUDAGraphWrapper` + **PIECEWISE**, with the collective registered as
  a **functionalized custom op** (`torch.ops.vllm.all_reduce`, out-of-place, with a
  `fake_impl`) and `splitting_ops`. Its `xpu.py:255` comment ("captured allreduce
  works under FULL_DECODE_ONLY") is "validated on Qwen3-Coder-Next TP=4", not gemma4.
  This is NOT the same as sglang's raw monolithic `torch.xpu.graph` capture of a raw
  `dist.all_reduce`.

## Refuted hypotheses (each tested; do not re-try without new evidence)

- Attention kernels (split-K / page_attn) broken under graph — **NO**, both capture
  fine in isolation (cos 1.0).
- Fresh `torch.empty` attention out/scratch buffers — made persistent (`_decode_buf`),
  **no fix**.
- `swa_page_table` / `swa_out_cache_loc` unstable addresses — made persistent + in-place,
  **no fix** (but they were latent bugs, kept).
- Split-K two-phase in-order race — added explicit `h.depends_on()` in the kernel +
  rebuilt eagle_ops, **no fix** (split-K captures fine in isolation anyway).
- Breakable-graph replay stream mismatch — tried capture-stream + `wait_stream`
  (deadlocked) and current-stream (1st decode ok then hang).
- CCL busy-spin thresholds — **not** the cause (on/off unchanged).

The one confirmed cause is the **captured-collective staleness** + the resulting
need to either (a) run the collective eager (breakable → but multi-collective
interleave deadlocks) or (b) make the collective graph-capturable.

## Fix options (ranked)

1. **Adopt vLLM's mechanism (recommended, the validated route):** sglang's
   `PiecewiseCudaGraphRunner` (torch.compile + inductor + `splitting_ops`). Register
   the TP collective as a functionalized custom op and let piecewise/inductor manage
   capture + stream ordering, matching how vLLM-xpu actually makes gemma4 work.
   Larger effort; depends on torch.compile running on this XPU stack.
2. **Graph-capturable custom SYCL allreduce:** `sgl-kernel-xpu/python/sgl_kernel/allreduce.py`
   defines `all_reduce` / `mscclpp_allreduce` / `init_custom_ar` ops, but they are
   **not built** in this container (`torch.ops.sgl_kernel.all_reduce` = False) and the
   XPU path hard-routes to `dist.all_reduce`. If built + wired + verified capturable
   (like split-K), the collective becomes a normal in-graph kernel → no breaks, no
   deadlock. Depends on the XPU custom-allreduce kernel's maturity.
3. **Breakable graph (this session's experiment, `SGLANG_XPU_BREAKABLE_GRAPH`, default off):**
   correct output but deadlocks under load. Would need the collective-interleave
   deadlock solved (explicit cross-rank barrier cadence, fewer break points via
   fused allreduce+rmsnorm, etc.) — uncertain.

## Config / how to reproduce

- Launch/probe scripts: `/home/intel/xiangyu/cc_workspace/gemma_splitk/`
  (`probe.sh`, `ask3.sh`, `bench_bsz1.py`, `test_*.py` isolation tests).
- Env flags added this session (all default-off, graph-only):
  `SGLANG_XPU_BREAKABLE_GRAPH`, `SGLANG_SPLITK_G` (default 4),
  `SGLANG_DISABLE_ESIMD_DECODE`, `SGLANG_DISABLE_PAGE_ATTN` (debug).
- Eager (shippable) = the same launch **without** `SGLANG_XPU_ENABLE_GRAPH` /
  `--cuda-graph-bs`; drop to `--disable-cuda-graph`.

## Related

- Split-K TPOT growth with context is graph-only and does not affect eager
  mode, which uses all four splits with chunk size approximately `ctx/4`.
- Commits (local only, unpushed): `2e811f49d3` (garble root-cause + eager verified),
  `6eaf848187` (SWA order fix + breakable-graph WIP).
