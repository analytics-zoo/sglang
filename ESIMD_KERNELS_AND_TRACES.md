# Intel XPU ESIMD Kernel and Trace Guide

Last audited: 2026-07-21

This document describes the ESIMD/SYCL fast paths used by the current
`dev-bmg-onyx` SGLang branch for Onyx and Gemma4 on Intel Battlemage (BMG).
It also indexes the profiling artifacts that are still present in this
workspace.

The scope is deliberately narrower than every ESIMD export in the package:

- included: kernels reached by the current Onyx and Gemma4 TP=2 FP8 paths;
- included: adjacent XPU kernels needed to interpret a trace, even when they
  are oneDNN or ordinary SYCL rather than ESIMD;
- excluded: inherited GGUF, Qwen3.5/3.6 GDN, and MoE-only kernels unless they
  are shared by Onyx or Gemma4.

## 1. Source of truth and runtime layout

There are two similarly named kernel trees in this workspace. They are not
equivalent.

The current source and compiled extension set is:

```text
/llm/workspace/sgl_gemma/llm-scaler/sglang/custom-esimd-kernels
```

The older top-level tree below is a Gemma-era build. It does not export the
packaged `onednn_fp8_gemm_w8a16` op and should not be placed first on
`PYTHONPATH` for the current Onyx path:

```text
/llm/workspace/sgl_gemma/custom-esimd-kernels-sglang
```

The current development sources can be selected without installing them:

```bash
export PYTHONPATH=/llm/workspace/sgl_gemma/sglang/python:\
/llm/workspace/sgl_gemma/llm-scaler/sglang/custom-esimd-kernels/python\
${PYTHONPATH:+:$PYTHONPATH}
```

This matters because the system-wide editable `sglang` currently resolves to
`/llm/sglang/python`, not this checkout.

The runtime binding chain is:

```text
SGLang call site
  -> custom_esimd_kernels_sglang/__init__.py (best-effort .so loading)
  -> custom_esimd_kernels_sglang/ops.py (Python ABI wrapper)
  -> torch op or pybind registration
  -> C++/SYCL launcher
  -> ESIMD kernel header
```

Important compiled modules are:

| Extension | Main contents |
|---|---|
| `custom_esimd_kernels*.so` | FP8 GEMV and decode fusions, norm fusions, KV scatter, FP16 GEMV |
| `custom_esimd_kernels_gemm*.so` | general-M FP8/INT4 GEMM |
| `eagle_ops*.so` | paged decode attention and split-K decode attention |
| `custom_esimd_kernels_attn*.so` | flat-NHD HD256 decode attention |
| `custom_esimd_kernels_prefill_dpas*.so` | explicit HD256 FP16 DPAS prefill attention |
| `onednn_w8a16*.so` | oneDNN FP16-activation/FP8-weight GEMM; not an ESIMD kernel |

See the package loader in
[`__init__.py`](../llm-scaler/sglang/custom-esimd-kernels/python/custom_esimd_kernels_sglang/__init__.py)
and the build definitions in
[`setup.py`](../llm-scaler/sglang/custom-esimd-kernels/setup.py).

## 2. Shared FP8, cache, and output kernels

### 2.1 `esimd_gemm_fp8_pert`

Purpose: FP16 activation x E4M3 FP8 weight with one per-tensor weight scale.
This is the common online-FP8 linear fast path for Onyx and Gemma4.

Runtime selection in
[`apply_fp8_linear`](python/sglang/srt/layers/quantization/fp8_utils.py):

- XPU only;
- `M <= 64`;
- one weight scale, no bias, and not compressed-tensor quantization;
- the weight is cached as a transposed contiguous `[N, K]` tensor;
- Onyx `down_proj` `(K,N)=(9984,6656)` uses ESIMD only at `M=1`;
- Onyx `gate_up_proj` `(6656,19968)` uses ESIMD only for `M<=32`;
- an unavailable qualified kernel falls back normally, or raises when
  `SGLANG_XPU_FP8_STRICT_DISPATCH=1`.

Implementation:

- Python wrapper:
  [`ops.py`](../llm-scaler/sglang/custom-esimd-kernels/python/custom_esimd_kernels_sglang/ops.py)
- registration:
  [`torch_extension_gemm.cc`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/torch_extension_gemm.cc)
- launcher:
  [`esimd_kernel_gemm.sycl`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/esimd_kernel_gemm.sycl)
- kernels:
  [`fp8_GEMM_pert.h`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/esimd_kernels/fp8_GEMM_pert.h)
  and
  [`fp8_GEMV_bmg.h`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/esimd_kernels/fp8_GEMV_bmg.h)

Typical unitrace symbol at `M=1`:

```text
GEMV_fp8_pert_batched_kernel<256, 1>
```

### 2.2 `onednn_fp8_gemm_w8a16` (companion, not ESIMD)

Purpose: keep the prefill activation in FP16 and multiply it by the FP8
weight with oneDNN. It supersedes activation-quantized FP8 prefill when the
shape is qualified.

Selection:

- XPU, one per-tensor weight scale, `M > 64`;
- enabled by `SGLANG_XPU_FP8_W8A16_PREFILL`, default `true`;
- Onyx `output_gate_proj` `(6656,2048)` uses it only at `M>=256`;
- otherwise the standard FP8 path is used;
- strict mode raises if a qualified W8A16 dispatch is missing.

Implementation:

- [`bindings.cpp`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/onednn_w8a16/bindings.cpp)
- [`onednn_runtime.cpp`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/onednn_w8a16/onednn_runtime.cpp)

The retained Gemma trace used an older `mini_fp8_C` copy of this op. The
current package exports it from `custom_esimd_kernels_sglang`.

### 2.3 `esimd_kv_scatter`

Purpose: replace separate indexed K-cache and V-cache writes with one launch.
The current kernel accepts independent source row strides, which is required
for Onyx K/V views split from packed QKV.

Selection in
[`memory_pool.py`](python/sglang/srt/mem_cache/memory_pool.py):

- XPU, same K/V row dimension, two-byte cache dtype;
- `row_dim % 32 == 0` and at least one token;
- disabled by `SGLANG_XPU_DISABLE_ESIMD_KV_SCATTER=1`;
- fallback is the native indexed assignment path.

Implementation:

- [`kv_scatter.h`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/esimd_kernels/kv_scatter.h)
- registration in
  [`torch_extension.cc`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/torch_extension.cc)

Trace symbols:

```text
KVScatter_kernel<128>       # Onyx HD128
KVScatter_kernel<256/512>   # model-dependent Gemma layouts
```

### 2.4 `esimd_gemv_fp16`

Purpose: FP16 LM-head GEMV for batch-one decode. It avoids a slow oneDNN
variant seen on large vocabularies.

Selection in
[`logits_processor.py`](python/sglang/srt/layers/logits_processor.py):

- op is available;
- hidden state is on XPU;
- exactly one token row;
- fallback is the regular matrix multiplication.

The FP16 LM-head weight is cached in a contiguous tensor. A typical Gemma
trace symbol is `GEMV_fp16_kernel<256, 1>`.

## 3. Onyx-specific decode fusions

The relevant model code is
[`onyx.py`](python/sglang/srt/models/onyx.py). Onyx has 52 decoder layers,
HD128 attention, hidden size 6656, and uses FP16 activation with online-FP8
decoder weights in the qualified path.

### 3.1 `esimd_gemv_fp8_pert_fused2`

Purpose: compute `qkv_proj` and `output_gate_proj` from the same input in one
submission while preserving independent weight scales and outputs.

Selection:

- XPU, FP16 contiguous input, `M=1`;
- both weights are E4M3 FP8 with one finite per-tensor scale;
- neither linear has a persisted input scale;
- disabled by `SGLANG_ONYX_DISABLE_FUSED_QKV_GATE_GEMV=1`.

Fallback: run the two linears independently through the regular FP8 dispatch.

Implementation:

- [`fp8_GEMV_bmg.h`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/esimd_kernels/fp8_GEMV_bmg.h)
- launcher and registration in
  [`esimd_kernel.sycl`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/esimd_kernel.sycl)
  and
  [`torch_extension.cc`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/torch_extension.cc)

The compiler-level trace name is still a `GEMV_fp8_pert_batched_kernel`; it
does not contain the Python wrapper name `fused2`.

### 3.2 `sgl_kernel.fused_qk_norm_rope` (SYCL companion)

This is not part of `custom_esimd_kernels_sglang`, but it is central to the
current Onyx fusion chain. It fuses Q/K RMSNorm, query scaling, and interleaved
RoPE; NoPE layers pass `rotary_dim=0` through the same kernel.

Selection:

- Q/K normalization enabled;
- XPU FP16 contiguous packed QKV;
- disabled by `SGLANG_ONYX_DISABLE_FUSED_QK_NORM_ROPE=1`.

Implementation:

- Python API:
  [`elementwise.py`](../sgl-kernel-xpu/python/sgl_kernel/elementwise.py)
- SYCL kernel:
  [`FusedQKNormRope.cpp`](../sgl-kernel-xpu/src/sycl/FusedQKNormRope.cpp)

Onyx trace symbol:

```text
at::native::xpu::FusedQKNormRopeKernel<128, true, half>
```

### 3.3 `esimd_norm_add_norm`

Purpose: fuse the post-attention norm, residual add, and pre-FFN norm into one
launch.

Selection: XPU FP16 contiguous tensors with `M=1`; the Onyx-wide opt-out is
`SGLANG_ONYX_DISABLE_RESIDUAL_NORM_FUSION=1`.

Implementation:
[`norm_add_norm.h`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/esimd_kernels/norm_add_norm.h).

Trace symbol: `NormAddNorm_kernel<512>` for hidden size 6656.

### 3.4 `esimd_rmsnorm_residual_scalar`

Purpose: fuse post-FFN RMSNorm, residual addition, and a scalar multiplier.
Onyx passes scalar `1.0`; Gemma4 passes its layer scalar.

Selection: the same batch-one, XPU, FP16 contiguous gate as the preceding
residual/norm fusion.

Implementation:
[`rmsnorm_residual_scalar.h`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/esimd_kernels/rmsnorm_residual_scalar.h).

Trace symbol: `RmsNormResidualScalar_kernel`.

### 3.5 Onyx attention is not a custom ESIMD kernel

Onyx HD128 attention currently uses the generic `sgl_kernel` XeFMHA path. The
final unitrace attributes about 3.7% of GPU self-time to these attention
kernels, so a new HD128 ESIMD attention kernel was not justified.

Typical trace symbol:

```text
(anonymous namespace)::KernelCur<...XeFMHAFwdKernel...>
```

## 4. Gemma4-specific ESIMD path

The model call sites are in
[`gemma4_causal.py`](python/sglang/srt/models/gemma4_causal.py), while attention
dispatch is in
[`xpu_backend.py`](python/sglang/srt/layers/attention/xpu_backend.py).

### 4.1 `esimd_qkv_split_norm_rope`

Purpose: split packed QKV and fuse Q/K RMSNorm, V normalization, and RoPE for
Gemma HD256 sliding-window layers.

Actual code gates:

- op available, `head_dim==256`, FP16 contiguous QKV;
- not a cross-layer KV-shared layer;
- disabled by `SGLANG_DISABLE_ESIMD_QKV=1`.

The source comment calls this a decode fast path, but the code has no explicit
`M==1` or forward-mode check. Treat that discrepancy as part of the contract
when changing the kernel.

Implementation:
[`qkv_split_norm_rope.h`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/esimd_kernels/qkv_split_norm_rope.h).

Trace name begins with `qkv_split_norm_rope_host(...)`.

### 4.2 Gemma residual/norm fusions

Dense Gemma layers reuse `esimd_norm_add_norm` and
`esimd_rmsnorm_residual_scalar` under the same XPU FP16 contiguous `M=1`
conditions as Onyx. `esimd_fused_add_rms_norm` remains as the fallback fusion
when the two-norm fusion is unavailable.

### 4.3 `eagle_page_attn_decode`

Purpose: ESIMD paged decode attention for HD256 sliding-window layers. It has
no SLM and was introduced partly for graph capture.

Current eager gate:

- ESIMD decode and page attention not disabled;
- HD256, FP16, one query token, zero logit cap, no cascade attention;
- SWA page tables are host-windowed to resident pages before the call;
- disable with `SGLANG_DISABLE_ESIMD_DECODE=1` or
  `SGLANG_DISABLE_PAGE_ATTN=1`.

The paged kernel is known to be wrong for GQA ratio 8 with one KV head on a
separate Qwen graph path; that path uses `sglang_decode_attn` instead.

Implementation:

- [`eagle.sycl`](../llm-scaler/sglang/custom-esimd-kernels/csrc/eagle/eagle.sycl)
- [`page.attn.gqa2.h`](../llm-scaler/sglang/custom-esimd-kernels/csrc/eagle/page.attn.gqa2.h)

### 4.4 `splitk_decode_attention`

Purpose: ESIMD split-K decode attention. In the Gemma4 delivery path it is
primarily the HD512 global-attention kernel after the HD256 page-attention
gate is skipped.

Selection:

- FP16, one query token, HD256 or HD512;
- zero logit cap and no cascade attention;
- enabled unless `SGLANG_DISABLE_ESIMD_DECODE=1`;
- number of splits is `SGLANG_SPLITK_G`, default 64.

Implementation:
[`splitk_decode.h`](../llm-scaler/sglang/custom-esimd-kernels/csrc/eagle/splitk_decode.h).

Trace symbols:

```text
skattn::splitKern<half, 512u>
skattn::reduceKern<half, 512u>
```

### 4.5 `sglang_decode_attn` and graph-only routing

This flat-NHD HD256 ESIMD attention implementation consumes token-granular
`kv_indptr/kv_indices`. The `intel_xpu` backend uses it in the graph-state
branch when `SGL_XPU_DECODE_SGLANG_ATTN` is not `0`.

Implementation:

- [`sglang_attn.sycl`](../llm-scaler/sglang/custom-esimd-kernels/csrc/eagle/sglang_attn.sycl)
- [`sglang_decode_attn.h`](../llm-scaler/sglang/custom-esimd-kernels/csrc/eagle/sglang_decode_attn.h)

The current Gemma4 shippable configuration disables XPU graph because captured
oneCCL collectives replay stale and can deadlock. This op is therefore not part
of the recommended eager delivery path.

### 4.6 `esimd_sdpa_prefill_dpas`

Purpose: explicit FP16 HD256 prefill attention using DPAS/XMX when the generic
XeFMHA FP16 path is incorrect on the target stack.

Selection is opt-in rather than default:

- `SGL_XPU_PREFILL_DPAS=1`;
- FP16, HD256, not cross attention, not cascade attention.

Implementation:

- [`esimd_kernel_prefill_dpas.sycl`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/esimd_kernel_prefill_dpas.sycl)
- [`torch_extension_prefill_dpas.cc`](../llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/torch_extension_prefill_dpas.cc)

HD512 prefill uses the `sgl-kernel-xpu` XeFMHA implementation. The adopted
HD512 tile change is in
[`FMHAPrefillXe20.cmake`](../sgl-kernel-xpu/src/FMHAPrefillXe20.cmake); it is
not an ESIMD kernel.

## 5. Fast-path sequence by model

### Onyx online-FP8 decode, batch one

```text
input RMSNorm
  -> fused QKV + output-gate FP8 GEMV
  -> fused Q/K norm + query scale + RoPE/NoPE
  -> strided ESIMD KV scatter
  -> generic HD128 XeFMHA attention
  -> output projection through shared FP8 GEMV
  -> NormAddNorm
  -> gate/up projection through shared FP8 GEMV
  -> SiLU/mul
  -> down projection through shared FP8 GEMV
  -> RmsNormResidualScalar
  -> FP16 LM-head GEMV
```

### Gemma4 online-FP8 decode, batch one

```text
shared FP8 QKV projection
  -> HD256: QKV split/norm/RoPE ESIMD
  -> KV scatter
  -> HD256 SWA: eagle page attention
     HD512 global: split-K decode attention
  -> output projection through shared FP8 GEMV
  -> NormAddNorm / fused-add-RMSNorm fallback
  -> MLP projections through shared FP8 GEMV
  -> RmsNormResidualScalar
  -> FP16 LM-head GEMV
```

## 6. Environment switches

| Variable | Default/meaning |
|---|---|
| `SGLANG_XPU_FP8_W8A16_PREFILL` | `true`; use qualified oneDNN W8A16 prefill |
| `SGLANG_XPU_FP8_PERTENSOR_PREFILL` | `true`; activation-quantized fallback prefill optimization |
| `SGLANG_XPU_FP8_STRICT_DISPATCH` | `false`; raise instead of silently missing a qualified FP8 fast path |
| `SGLANG_XPU_DISABLE_ESIMD_KV_SCATTER` | `0`; set `1` for native indexed KV writes |
| `SGLANG_ONYX_DISABLE_FUSED_QKV_GATE_GEMV` | `0`; Onyx fused-two GEMV opt-out |
| `SGLANG_ONYX_DISABLE_FUSED_QK_NORM_ROPE` | `0`; Onyx fused Q/K/rope opt-out |
| `SGLANG_ONYX_DISABLE_RESIDUAL_NORM_FUSION` | `0`; Onyx two residual/norm fusions opt-out |
| `SGLANG_DISABLE_ESIMD_QKV` | `0`; Gemma HD256 QKV fusion opt-out |
| `SGLANG_DISABLE_ESIMD_DECODE` | `0`; disable custom page/split-K decode attention |
| `SGLANG_DISABLE_PAGE_ATTN` | `0`; disable only the page-attention branch |
| `SGLANG_SPLITK_G` | `64`; split count for split-K decode attention |
| `SGL_XPU_PREFILL_DPAS` | unset/off; opt in to HD256 FP16 DPAS prefill |
| `SGL_XPU_DECODE_SGLANG_ATTN` | `1`; select flat-NHD attention in the eligible graph branch |

There are two environment-variable prefixes in the current branch:
`SGLANG_*` and inherited `SGL_XPU_*`. Do not silently rename one to the other;
the code reads the exact names shown above.

## 7. Retained trace inventory

### 7.1 Onyx final optimized unitrace — preferred current evidence

Directory:

```text
/llm/workspace/copilot_workspace/onyx_unitrace_final2/
```

Files:

| File | Size | Meaning |
|---|---:|---|
| `python3.32015.json` | 261.6 MB | TP rank/device trace |
| `python3.32016.json` | 260.7 MB | TP rank/device trace |
| `serve.log` | 30.7 KB | exact launch arguments, loader and request log |
| `runtime.json` | 774 B | 1024-input/64-output capture metadata |

The run is TP=2, FP16 activation, online FP8, `intel_xpu`, eager, radix off,
one warmup plus one measured request. `serve.log` confirms the packaged W8A16
op and all 260 decoder FP8 linears.

Useful trace evidence on each rank:

- 6,656 `FusedQKNormRopeKernel<128,...>` launches;
- 6,552 `NormAddNorm_kernel<512>` launches;
- 6,552 `RmsNormResidualScalar_kernel` launches;
- 6,656 `KVScatter_kernel<128>` launches, including 104 prefill launches;
- FP8 GEMV kernels and 6,760 generic XeFMHA launches are present.

The 6,552 decode count equals `52 layers * 63 decode steps * 2 requests`.
The trace adds significant overhead: `runtime.json` reports 64.4 ms TPOT,
whereas the controlled non-unitrace optimized run reports about 34 ms at 1K.
Do not use unitrace wall time as serving performance.

The earlier `onyx_unitrace_final/` directory is an incomplete first attempt;
prefer `final2`.

### 7.2 Gemma4 W8A16 unitrace — closest retained delivery trace

Directory:

```text
/llm/workspace/cc_workspace/gemma_splitk/utrace_w8a16/
```

Files:

- `python3.1110.json`, 544.8 MB;
- `python3.1111.json`, 550.6 MB;
- `serve.log`, with FP16/FP8 TP=2 eager launch details.

This 2026-07-06 trace covers a 4K-input eager run after the major Gemma decode
fusions and W8A16 work. It contains the expected QKV split/norm/rope,
`NormAddNorm`, `RmsNormResidualScalar`, HD512 split-K, FP8 GEMV, FP16 LM-head,
XeFMHA, and oneCCL symbols.

It is close to the shippable Gemma configuration but not an exact trace of the
current packaged binary: its log says `mini_fp8_C loaded`, while current code
loads W8A16 from `custom_esimd_kernels_sglang`.

Other retained Gemma unitrace generations are:

| Directory | Date | Use |
|---|---|---|
| `/llm/workspace/cc_workspace/gemma_splitk/utrace_4k/` | 2026-07-01 | 4K baseline and early split-K investigation |
| `/llm/workspace/cc_workspace/gemma_splitk/utrace_cur/` | 2026-07-01 | intermediate current-state capture |
| `/llm/workspace/cc_workspace/gemma_splitk/utrace_on/` | 2026-07-02 | later fast-path-on capture |
| `/llm/workspace/sgl_gemma/_claude_tmp/utrace/` | 2026-06-29 | early BF16 baseline; historical only |

Always read the adjacent `serve.log` before comparing two directories.

### 7.3 Gemma stage-separated PyTorch profiler traces

Directory:

```text
/llm/workspace/sgl_gemma/_claude_tmp/prof_stage/
```

It contains TP0/TP1 `EXTEND` and `DECODE` traces as compressed Chrome JSON.
These are early 2026-06-29 captures, useful for stage separation and host
stacks but older than the final fusion work.

The existing analyzer is
[`analyze_decode_trace.py`](../_claude_tmp/scripts/analyze_decode_trace.py).
For example:

```bash
python3 /llm/workspace/sgl_gemma/_claude_tmp/scripts/analyze_decode_trace.py \
  /llm/workspace/sgl_gemma/_claude_tmp/prof_stage/\
1782703516.8744757-TP-0-DECODE.trace.json.gz
```

Its aggregate for the old TP0 decode capture is approximately:

| Bucket | GPU kernel time share |
|---|---:|
| ESIMD FP8 GEMM/GEMV | 50.4% |
| oneCCL TP communication | 28.8% |
| oneDNN BF16 GEMM | 9.3% |
| elementwise/copy/index | 4.5% |
| RMSNorm | 2.9% |
| XeFMHA attention | 1.7% |

TP1 attributes 79.1% of summed kernel duration to oneCCL in the same capture,
which is one of the pieces of evidence for the communication bottleneck. These
percentages are sums of kernel durations, not end-to-end wall-time shares.

### 7.4 Onyx numerical precision traces

[`benchmark/onyx/trace_precision.py`](benchmark/onyx/trace_precision.py)
implements a separate tensor-dump workflow. It captures ordered module inputs
and outputs as `Pass00000.pt` on both TP ranks and compares BF16/FP16 traces.
This is a numerical-debug trace, not a GPU timeline.

No `Pass00000.pt` files are currently present under `/llm/workspace`. The
status document records their former results, but only the capture/compare tool
remains in this checkout.

The benchmark and accuracy JSON artifacts are still available under:

```text
/llm/workspace/copilot_workspace/onyx_runtime_*.json
/llm/workspace/copilot_workspace/onyx_arc_*.json
```

## 8. Inspecting and recapturing traces

The unitrace files are Chrome trace JSON. Load one rank at a time in Perfetto
or `chrome://tracing`; the 250-700 MB files can exhaust browser memory if both
ranks are opened together.

Quick symbol checks can be done without parsing the whole JSON object:

```bash
rg -i 'GEMV_fp8|FusedQKNormRope|NormAddNorm|RmsNormResidualScalar|\
KVScatter|splitKern|XeFMHAFwdKernel|oneccl' \
  /llm/workspace/copilot_workspace/onyx_unitrace_final2/python3.32015.json
```

The old capture helper
[`launch_unitrace_gemma4.sh`](../_claude_tmp/scripts/launch_unitrace_gemma4.sh)
starts unitrace with both host and device Chrome logging. It begins with
`rm -rf "$OUT"`, so do not rerun it against a directory containing evidence
that must be preserved. Change `OUT` to a new timestamped directory first.

A new trace should record at minimum:

- SGLang, `llm-scaler`, and `sgl-kernel-xpu` commit IDs;
- exact extension directory or wheel hash;
- model, dtype, quantization, TP, eager/graph, radix, input/output length;
- environment switches from section 6;
- warmup/trial count and whether unitrace overhead is included;
- `serve.log`, request metadata, and one trace per TP rank.

## 9. Known interpretation traps

- A Python op name and the emitted kernel symbol often differ. For example,
  the Onyx `fused2` wrapper still appears as `GEMV_fp8_pert_batched_kernel`.
- oneDNN W8A16, XeFMHA, `sgl_kernel.fused_qk_norm_rope`, and oneCCL appear in
  the same timeline but are not custom ESIMD kernels.
- Import success does not prove that every optional `.so` loaded. Check
  `custom_esimd_kernels_sglang._MISSING_EXTS` and the server log.
- `unitrace` adds large overhead; use it for kernel attribution and ordering,
  not service latency.
- XPU graph traces are not delivery evidence for Gemma4. Eager remains the
  qualified path.
- The Gemma comments describe `esimd_qkv_split_norm_rope` as decode-only, but
  the current code does not enforce `M==1`.
