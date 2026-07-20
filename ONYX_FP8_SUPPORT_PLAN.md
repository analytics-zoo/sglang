# Onyx Complete Online FP8 Support Plan

Last updated: 2026-07-17

Status: Level A text FP8 and Phases 0-4 are complete. Phase 5 is unnecessary
for the functional target, and Phase 6 did not meet its trace-driven entry
condition. Phase 7 image bring-up is complete with a BF16 vision tower and the
online-FP8 text decoder; video integration and vision-FP8 qualification remain
open. Only results recorded in `ONYX_SGLANG_SUPPORT_STATUS.md` are validated.

## 1. Goal and definition of "complete"

This plan covers online FP8 weight quantization for Onyx on Intel XPU/BMG. The
source checkpoint remains BF16 and is quantized while loading. It deliberately
separates two delivery levels:

### Level A: complete text FP8

- TP=2, eager execution, `intel_xpu` attention.
- All 260 decoder linear modules use per-tensor E4M3 FP8 weights:
  - 52 `qkv_proj`
  - 52 `output_gate_proj`
  - 52 `o_proj`
  - 52 `gate_up_proj`
  - 52 `down_proj`
- Prefill uses W8A16: FP16 activations and FP8 weights, without activation
  quantization when the oneDNN extension is available.
- Decode and small-M work use the ESIMD FP8 GEMV/GEMM path only after every
  Onyx TP=2 shape passes a kernel microbenchmark.
- Input embedding, independent LM head, norms, RoPE, attention activations, and
  KV cache remain unquantized FP16, matching the Gemma4-31B FP8 boundary.
- "Complete text FP8" therefore means all decoder `LinearBase` weights are FP8;
  it does not mean every model tensor is stored as FP8.

### Level B: complete multimodal FP8

Level A plus, after the Onyx vision/video model itself is implemented in
SGLang:

- Eligible vision encoder, adapter, and projection linears use online FP8.
- Sensitive or unsupported vision modules remain in BF16 through a measured
  ignore list.
- Image and video task accuracy is qualified independently; text-only accuracy
  cannot approve multimodal FP8.

### Explicit non-goals

- XPU graph. TP>1 graph is not a prerequisite for FP8 delivery.
- TP=4. The target is TP=2.
- FP8 KV cache. Initial and shippable configurations keep KV in BF16/FP16.
- W8A8 activation quantization as the default path.
- An offline serialized-FP8 checkpoint. It can be considered only after online
  FP8 is correct and stable.
- Quantizing RMSNorm, RoPE, softmax, residuals, input embeddings, or `lm_head`.
- Claiming Gemma4 performance numbers for Onyx without Onyx measurements.

## 2. Established baseline

The starting point is the validated BF16 text path documented in
`ONYX_SGLANG_SUPPORT_STATUS.md`:

- 52 decoder layers, hidden size 6656, FFN size 19968.
- Q32/KV2/HD128, TP=2, 39 SWA layers and 13 global/NoPE layers.
- Hybrid SWA, interleaved RoPE, Q/K norm, query scale, output gate, norm
  conventions, and logits transform are implemented.
- Native and converted-HF CPU FP32 prefill/decode logits are bit-exact.
- TP=2 XPU BF16 raw parity returns reference token 328 for the fixed six-token
  prompt.
- English and Chinese multi-token decode is semantically correct.
- The packed-QKV strided KV-cache corruption has a regression test and is
  fixed in the common cache writer.
- `layered_fp8` already provides stable CPU-staged BF16 loading for Onyx, but
  Onyx online FP8 loading has not yet been qualified.

This BF16 path remains the architecture reference. The direct FP8 control will
be an unquantized FP16 run, because the target configuration is FP8 weights plus
FP16 activation.

## 3. Current source readiness and real gaps

### Already wired

`python/sglang/srt/models/onyx.py` already propagates `quant_config` into:

- `QKVParallelLinear`
- `ColumnParallelLinear` for `output_gate_proj`
- `RowParallelLinear` for `o_proj`
- `MergedColumnParallelLinear` for `gate_up_proj`
- `RowParallelLinear` for `down_proj`

`Fp8Config.get_quant_method()` assigns `Fp8LinearMethod` to these
`LinearBase` modules. Therefore the first functional decoder-FP8 bring-up
should not require a new Onyx linear implementation.

The input embedding intentionally does not receive `quant_config`.

### Missing or unqualified

1. **Onyx FP8 Phase 1 is complete.**
   Two clean `--quantization fp8 --load-format layered_fp8` TP=2 starts verify
   260 E4M3 decoder linears and pass raw, English/Chinese chat, and
   2047/2048/2049 SWA-boundary gates. Phase 3 task and cache qualification is
   also complete.

2. **The complete Onyx FP8 shape matrix is qualified.**
   `benchmark/onyx/bench_fp8_kernels.py` covers all five TP=2 shapes and all
   required small-M and prefill token counts.

3. **The production W8A16 extension path is complete.**
   `fp8_utils.py` consumes the maintained
   `custom_esimd_kernels_sglang.onednn_fp8_gemm_w8a16` packaged op. Onyx
   shape-aware gates retain the faster fallback for nine candidate cases.

4. **Fast-path dispatch is observable.**
   `SGLANG_XPU_FP8_STRICT_DISPATCH=1` fails when an eligible extension is
   unavailable and logs each selected ESIMD, W8A16, or qualified fallback shape.

5. **The initial FP16 activation gate is established.**
   Two clean pure-FP16 starts pass fixed raw parity, English/Chinese chat, and
   the 2047/2048/2049 SWA boundary. Broader task qualification remains open.

6. **HD128 attention remains generic.**
   Current Gemma4-specific ESIMD attention and QKV/norm/RoPE fusions target
   HD256/HD512. This is a performance gap, not a blocker for FP8 linear
   correctness.

7. **Image execution is implemented; video remains open.**
   The current SGLang Onyx model loads the checkpoint-faithful vision tower in
   BF16 and serves single- and multi-image requests with the online-FP8 text
   decoder. Video frame grouping, real PTS propagation, VIDEO item construction,
   and six-channel temporal-patch dispatch are not yet implemented.

## 4. TP=2 FP8 shape inventory

The table uses logical GEMM dimensions `K -> N` per TP rank. Runtime FP8 storage
may be transposed for the selected kernel.

| Module | Count | K | N per TP rank | Level A target |
|---|---:|---:|---:|---|
| `qkv_proj` | 52 | 6656 | 2304 | FP8 weight, ESIMD small-M, W8A16 prefill |
| `output_gate_proj` | 52 | 6656 | 2048 | FP8 weight, ESIMD small-M, W8A16 prefill |
| `o_proj` | 52 | 2048 | 6656 | FP8 weight, ESIMD small-M, W8A16 prefill |
| `gate_up_proj` | 52 | 6656 | 19968 | FP8 weight, ESIMD small-M, W8A16 prefill |
| `down_proj` | 52 | 9984 | 6656 | FP8 weight, ESIMD small-M, W8A16 prefill |
| `lm_head` | 1 | 6656 | 101024 | Unquantized FP16, same as Gemma4-31B |

The decoder linear total is 25,163,726,848 parameters. The input embedding and
LM head are each 1,344,831,488 parameters. Ignoring small scales, norms,
allocator fragmentation, kernel workspaces, and KV cache:

- Level A weight estimate at TP=2:
  `decoder_fp8 + embedding_fp16 + lm_head_fp16 ~= 14.22 GiB/rank`.

These are capacity estimates, not measured memory results. Actual peak and
steady-state memory must be recorded from the server.

## 5. Phase plan

### Phase 0: freeze the control and build the harness

#### Work

1. Freeze the exact BF16 architecture reference:
   - checkpoint config and index hashes
   - SGLang, kernel, and llm-scaler commits
   - container image
   - TP, dtype, attention backend, pool sizes, and environment
2. Preserve a deterministic raw-parity corpus, including the known input:
   `[200000, 954, 10810, 323, 4302, 373]`.
3. Build an Onyx chat evaluation harness that:
   - always applies the model chat template
   - handles the generated `to=user<|message|>` recipient header
   - records output IDs as well as decoded content
   - checks NaN/Inf and invalid output
4. Retain the existing BF16 task results as a reference:
   - English and Chinese generation
   - arithmetic/reasoning, including GSM8K if the model has a stable baseline
   - tool-call formatting
   - long-context/SWA boundary cases
5. Run unquantized FP16 and establish it as the direct control for all later
   FP8 work. Do not use either FP16 or FP8 performance numbers until this exact
   FP16 path passes the correctness gate.

#### Current Phase 0 evidence

- CPU HF text-only reference with all 27,854,780,928 parameter elements forced
  to FP16 returns finite FP32 logits and selects token 328 for the fixed raw
  input. Its top five token IDs are 328, 262, 290, 702, and 511.
- The earlier TP=2 SGLang FP16 token-16 result is invalid: an isolation log
  records the intended FP16 server failing to bind port 30000 while a stale
  service remained reachable.
- Two independent clean TP=2 SGLang FP16 starts select token 328. An untraced
  validation run also passes English and Chinese multi-token chat plus valid
  outputs at SWA-boundary lengths 2047, 2048, and 2049.
- Clean BF16/FP16 all-I/O captures each contain 9,969 tensors per rank. Final
  logits have relative RMSE 0.00730 and cosine similarity 0.99998 and preserve
  token 328. No FP16 correctness defect is currently reproduced.

#### Comprehensive precision trace usage

`benchmark/onyx/trace_precision.py` captures one fixed raw request with
`TENSOR_DUMP_MODE=all_io`. The mode records, in execution order, every parent
and leaf module input/output on both TP ranks. One capture therefore covers all
52 decoder layers, including block boundaries, norms, QKV and Q/K norm, RoPE,
attention, output-gate input/output, MLP stages, residual boundaries, final
hidden state, and final logits. It synchronizes tensors to CPU and must never be
used for performance measurement.

Run exactly one clean capture per dtype. From the host, start each capture
detached so loss of the terminal cannot terminate the Engine:

```bash
docker exec -d txytest_sgl_bmg bash -lc '
  export PYTHONPATH=/llm/workspace/sgl_gemma/sglang/python
  export ZE_AFFINITY_MASK=0,1
  export SGLANG_USE_SGL_XPU=1
  export SGLANG_SKIP_VISION_GPU=1
  export PYTHONUNBUFFERED=1
  cd /llm/workspace/sgl_gemma/sglang
  python3 benchmark/onyx/trace_precision.py capture \
    --dtype bfloat16 \
    --output-dir /llm/workspace/copilot_workspace/onyx_trace_bf16 \
    > /llm/workspace/copilot_workspace/onyx_trace_bf16.log 2>&1
  printf "%s\n" "$?" \
    > /llm/workspace/copilot_workspace/onyx_trace_bf16.rc
'
```

Wait for the rc file with the standard 30-second distributed-init and
60-second weight-load watchdogs, then restart the whole container and confirm
zero zombies. Repeat the same command once with:

```text
--dtype float16
--output-dir /llm/workspace/copilot_workspace/onyx_trace_fp16
```

and use `onyx_trace_fp16.log` / `onyx_trace_fp16.rc` for its log and result.
The capture refuses a non-empty output directory so tensors from different
server starts cannot be mixed.

After both successful captures, compare them without starting a model:

```bash
docker exec txytest_sgl_bmg bash -lc '
  cd /llm/workspace/sgl_gemma/sglang
  python3 benchmark/onyx/trace_precision.py compare \
    --reference-dir /llm/workspace/copilot_workspace/onyx_trace_bf16 \
    --candidate-dir /llm/workspace/copilot_workspace/onyx_trace_fp16 \
    --report /llm/workspace/copilot_workspace/onyx_trace_comparison.json
'
```

Each capture contains `metadata.json` plus one `Pass00000.pt` per TP rank. The
comparison report preserves execution order, reports shape/dtype/finite status,
RMSE, relative RMSE, cosine similarity, the first threshold-crossing tensor per
rank, and the largest divergences. Thresholds are diagnostic defaults and can
be changed with `--relative-rmse-threshold` and `--cosine-threshold`; the raw
tensors remain the source of truth. Delete both trace directories, their logs
and rc files, and the comparison JSON after the confirmed conclusion is written
to this status document.

#### Exit gate

- The existing BF16 reference remains reproducible after a clean container
  restart.
- The harness reports stable scores and token IDs.
- Pure FP16 passes raw, chat/task, SWA-boundary, and multi-token decode gates.
- If pure FP16 is incorrect, FP8 work stops and fixes FP16 first.

### Phase 1: decoder online-FP8 functional bring-up

#### Work

1. Start with the smallest behavior change:
   - `--quantization fp8`
   - `--load-format layered_fp8`
   - `--dtype float16`
   - TP=2, eager, `intel_xpu`
2. Keep the current CPU-staged loader lifecycle:
   - construct and load the full BF16 text model on CPU
   - move one module at a time to XPU
   - run `process_weights_after_loading`
   - retain only the FP8 result on XPU
3. Add inspection coverage proving that all 260 decoder linears have:
   - E4M3 FP8 weight
   - finite scalar per-tensor weight scale
   - dynamic/no persisted input scale
   - the expected TP=2 shape
4. Confirm that the input embedding, norms, KV cache, and LM head remain
   unquantized FP16.
5. Verify both packed mappings during CPU load:
   - Q/K/V checkpoint tensors into `qkv_proj`
   - gate/up checkpoint tensors into `gate_up_proj`
6. Keep the existing strided-KV writer regression in the gate. FP8 projection
   output must not reintroduce the packed-QKV cache corruption.
7. Add loader progress markers. Use a no-progress watchdog rather than treating
   a slow but progressing online quantization pass as a hang.

#### Exit gate

- Two consecutive clean-container starts complete without OOM or loader stall.
- Exactly 260 decoder modules are FP8 and their scales are finite.
- The fixed raw prompt still selects token 328.
- Multi-token prefill/decode completes with no NaN/Inf.
- English and Chinese chat remain coherent.
- Failure cleanup leaves zero zombie processes.

### Phase 2: qualify Onyx FP8 kernels before performance integration

#### Work

1. Add standalone benchmarks for every FP8 decoder shape in Section 4. The
   unquantized LM head is listed only to make the dtype boundary explicit.
2. Test ESIMD small-M at `M={1,2,4,8,16,32,64}`.
3. Test W8A16 prefill at representative token counts beginning above the
   small-M cutoff and extending through the supported context:
   `M={65,128,256,1024,2048,4096,8192,16384}` where memory permits.
4. Compare each kernel with a dequantized FP32/FP16 PyTorch reference using:
   - max absolute and relative error
   - cosine similarity
   - NaN/Inf checks
5. Benchmark FP16 activation input only.
6. Verify actual runtime dispatch:
   - ESIMD kernel for eligible small-M linears
   - packaged oneDNN W8A16 op for prefill
   - explicit fallback identity when either extension is absent
7. Replace the hard-coded `mini_fp8_C*.so` lookup with the packaged
   `custom_esimd_kernels_sglang.onednn_fp8_gemm_w8a16` interface.
8. Add a strict diagnostic mode that fails instead of silently falling back.
   Normal production behavior may retain a correct fallback, but the release
   gate must prove which path ran.

#### Exit gate

- All Section 4 decoder shapes pass numerical comparison.
- Each selected fast path is faster than its fallback on the same XPU, in the
  same process, after shared warmup. No kernel is integrated because it is
  merely present.
- Actual kernel names/dispatch counters confirm the intended path.
- Missing extensions produce an actionable log and never a false "W8A16
  enabled" report.

### Phase 3: model-level FP8 correctness qualification

Only two runtime configurations are compared:

| Configuration | Purpose |
|---|---|
| Pure FP16 | direct correctness control |
| FP8 decoder weights + FP16 activation | only FP8 candidate |

The existing BF16 result remains a historical architecture reference, but it is
not part of an FP8 precision matrix. There is no FP8-weight + BF16-activation
branch in this plan.

#### Required suites

1. **Raw parity**
   - fixed six-token prompt
   - curated prompts covering different vocabulary regions
   - compare top-k IDs and logits, not only decoded text
2. **Chat/task**
   - same chat template and extraction for pure FP16 and FP8+FP16
   - English, Chinese, arithmetic/reasoning, and tool calling
   - zero-shot ARC-Challenge through the Chat API, using the complete test split
     and identical deterministic prompts
   - establish Onyx's pure-FP16 score before applying the FP8 threshold
3. **Hybrid attention**
   - lengths around the inclusive 2048 SWA boundary:
     2047, 2048, and 2049
   - longer cases at 4096 and 8192; the model's 16K maximum is not tested with
     a full-length input because generation requires additional token capacity
   - prompts that require retaining information across global layers
4. **Decode/cache**
   - multi-token greedy decode
   - multi-turn/prefix reuse
   - radix cache on and off
5. **Numerical safety**
   - finite scales
   - finite per-layer hidden states and final logits
   - no invalid token IDs or empty-success responses

#### Proposed acceptance gate

- The fixed raw prompt must retain token 328.
- No NaN/Inf or invalid output.
- FP8+FP16 task accuracy must be within 1 percentage point of pure FP16 on a
  sufficiently sized deterministic set. GSM8K and ARC-Challenge are evaluated
  independently; either suite may lose at most 1 percentage point. This is a
  proposed release threshold, not a measured claim.
- Long-context cases at 2047 through 8192 tokens are stability and hybrid-SWA
  gates: outputs must be finite, valid, deterministic under repetition, and work
  with radix cache both on and off. FP8 is not required to select the exact same
  token as FP16.
- A top-token change is diagnostic evidence, not by itself a correctness
  failure. Layer tracing is required only when task accuracy regresses beyond
  the threshold, values become non-finite, output is invalid/incoherent, or the
  divergence suggests a shape- or dispatch-specific defect.
- Performance measurements remain invalid until this gate passes.

### Phase 4: end-to-end performance and memory qualification

#### Method

1. Extend the runtime benchmark used for Gemma4 rather than relying on unitrace
   GPU self-time:
   - `/generate`
   - streaming
   - `ignore_eos`
   - batch size 1 initially
   - warmup 2
   - trials 3
   - report medians
2. Use real Onyx-tokenized text. Repeated BOS or other out-of-distribution dummy
   prompts are not a correctness or speculative-decoding workload.
3. Measure input lengths 1K, 2K, 4K, 8K, and a practical near-16K point with a
   fixed output length. The qualified capacity point is 16,000 input plus 64
   output tokens under the model's 16,384-token limit.
4. Record:
   - load time
   - peak and steady per-rank memory
   - TTFT
   - TPOT
   - output throughput
   - host load and competing scheduler processes
5. Keep every parameter except the intended experiment variable identical.
   Pure FP16 and FP8+FP16 require separate server starts, so record them as
   matched startup runs and use an A-B-A order when node load permits.
6. Use traces only to explain runtime results. Do not substitute kernel
   self-time for end-to-end latency.

#### Optimization order

1. Confirm packaged W8A16 prefill.
2. Confirm ESIMD decode for all five decoder linear types.
3. Investigate output-gate projection and sigmoid/multiply launch overhead.
4. Investigate HD128 attention only if end-to-end traces show material cost.
5. Treat TP allreduce and shared-node host contention separately from FP8
   linear efficiency.

#### Exit gate

- Measured memory matches the expected direction and the server retains enough
  capacity for the declared 16K configuration.
- The FP8+FP16 path is not slower than pure FP16
  without a documented, measured explanation.
- W8A16 and ESIMD wins are demonstrated first in microbenchmarks and then in
  end-to-end runtime.
- No cross-startup or cross-harness numbers are subtracted as if they were a
  controlled delta.

#### Result

Phase 4 passed with `benchmark/onyx/bench_runtime.py`: real Onyx-tokenized
text, `/generate`, streaming, `ignore_eos`, batch size 1, two warmups, three
trials, and medians. Matched eager TP=2 radix-off starts used A-B-A ordering.
Across 1,024 through 16,000 input tokens, FP8 TTFT was within +0.04% to +2.17%
of the two-run FP16 midpoint, while TPOT improved by 18.42% to 20.22%. The
64-token end-to-end request was faster at every length.

Weight loading used 26.01 GB/rank for FP16 and 14.24 GB/rank for FP8. Sampled
peak/steady device usage was 28,198/27,834 MiB on FP16 and 16,547/16,182 MiB
on FP8 for GPU0/GPU1; no larger transient peak was observed. FP8 therefore
reduces steady device usage by 41.3%-41.9% and retains the declared 16K
capacity. FP8 online conversion increases weight-load time from about 18.1 s
to 32.7 s and total startup from about 45 s to 60 s.

The measured path already satisfies the Level A performance gate. Further
kernel work is therefore an optimization investigation, not a release blocker.
Source comparison with the optimized Gemma4 path established the following
ordered queue. Every item requires a standalone correctness/performance
microbenchmark before model integration:

1. **HD128 fused QK norm + RoPE.** Onyx currently launches two scaleless norms,
   query scaling, and interleaved RoPE separately. The existing
   `sgl_kernel.fused_qk_norm_rope` supports FP16, HD128, and interleaved RoPE.
   Fold the constant Onyx query scale into its Q weight; also qualify
   `rotary_dim=0` for the 13 NoPE layers.
2. **Decoder residual/norm fusion.** Qualify `esimd_norm_add_norm` for
   post-attention norm + residual + pre-FFN norm, then
   `esimd_rmsnorm_residual_scalar` with scalar 1 for post-FFN norm + residual.
   Preserve Onyx's `(1 + weight)` semantics explicitly.
3. **Stride-aware KV scatter.** Onyx K/V remain strided views of packed QKV, so
   the existing contiguous-only `esimd_kv_scatter` is bypassed. Measure the
   native scatter first, then extend the kernel to consume source row stride
   only if it wins over both native scatter and explicit contiguous copies.
4. **QKV + output-gate decode projection.** Qualify
   `esimd_gemv_fp8_pert_fused2` at bsz=1 while keeping independent weights and
   scales. Keep sigmoid application after attention and do not introduce a
   second persistent packed-weight layout.
5. **HD128 attention.** Only after the lower-risk fusions, trace the full model.
   Gemma4's DPAS prefill and ESIMD page/split-K decode paths are gated to
   HD256/HD512, so HD128 requires kernel work. Start it only if the trace shows
   material TTFT or TPOT cost.

All five investigations are complete. The fused QK microbenchmark is
1.91-2.51x faster with Q/K cosine at least 0.99999988. Matched model A/B/A
shows 9.92%-10.79% lower TPOT in A and 12.93%-13.74% lower TPOT in A2; the
exact fused path scores 96/100 on GSM8K Chat with zero invalid responses.

The residual/norm microbenchmark is 1.55x faster for post-attention add plus
pre-FFN norm and 1.33x faster for post-FFN norm plus residual, with cosine at
least 0.99999988. Its matched A/B/A lowers TPOT by 9.26%-9.93% in A and
9.09%-10.59% in A2 without material TTFT change. The complete smoke,
2047-8192 boundary, repeatability, multi-turn, and tool-call capture passes.
The opt-outs are `SGLANG_ONYX_DISABLE_FUSED_QK_NORM_ROPE=1` and
`SGLANG_ONYX_DISABLE_RESIDUAL_NORM_FUSION=1`.

The stride-aware KV scatter is bit-exact for packed source stride `(2304, 1)`
and takes 41.75-42.01 us at M=1/64/1024, about 20%-21% below native scatter.
Matched A/B/A lowers TPOT by 2.44%-3.81% in A and 0.86%-3.99% in A2. The
native comparison opt-out is `SGLANG_XPU_DISABLE_ESIMD_KV_SCATTER=1`.

The QKV/output-gate fused2 GEMV is bit-exact with independent scales and is
1.05x faster standalone. It reuses the existing `_esimd_t` weight layouts.
Matched A/B/A lowers TPOT by 3.25%-4.34% in A and 3.44%-5.23% in A2. The
opt-out is `SGLANG_ONYX_DISABLE_FUSED_QKV_GATE_GEMV=1`.

The final full-model trace does not justify HD128 attention work:
`XeFMHAFwdKernel` accounts for 3.76% and 3.73% of GPU self-time on TP0/TP1
over one warmup plus one measured 1K-input/64-output request. No new attention
kernel is started.

Existing optimized paths that are not part of this queue are shape-qualified
FP8 ESIMD decode, W8A16 prefill, XPU SiLU-and-multiply, SYCL standalone RMSNorm,
and ESIMD decode LM head.

A pre-optional-optimization FP8 throughput matrix used the same harness with
512 output tokens, two warmups, three trials, streaming, and `ignore_eos`. It
is an FP8 capacity measurement rather than an FP16 comparison:

| Input tokens | TTFT (ms) | TPOT (ms) | Decode throughput (token/s) | E2E (s) |
|---:|---:|---:|---:|---:|
| 1,024 | 289.00 | 44.57 | 22.44 | 23.07 |
| 2,048 | 589.13 | 44.87 | 22.29 | 23.51 |
| 4,096 | 1,249.50 | 47.62 | 21.00 | 25.59 |
| 8,192 | 2,453.47 | 47.82 | 20.91 | 26.89 |

The machine-readable result is
`/home/intel/xiangyu/copilot_workspace/onyx_runtime_fp8_output512.json`.

After completing optimization items 1-4, the fully optimized FP8 path was
remeasured with the same methodology:

| Input tokens | TTFT (ms) | TPOT (ms) | Decode throughput (token/s) | E2E (s) |
|---:|---:|---:|---:|---:|
| 1,024 | 286.89 | 34.17 | 29.26 | 17.75 |
| 2,048 | 584.51 | 34.34 | 29.12 | 18.13 |
| 4,096 | 1,233.58 | 37.45 | 26.70 | 20.37 |
| 8,192 | 2,433.75 | 37.14 | 26.93 | 21.41 |

The machine-readable result is
`/home/intel/xiangyu/copilot_workspace/onyx_runtime_optimized_fp8_output512.json`.

### Phase 5: optional fused QKVG/output-gate optimization

This is not required for functional FP8 support.

The native checkpoint stores `[Q|K|V|output_gate]`, while the current SGLang
model uses fused QKV plus a separate output-gate linear. A safe optimization
must:

1. Replace, not duplicate, existing persistent parameters.
2. Load local TP shards directly; never construct the full fused GPU weight.
3. Preserve separate quantization scales for QKV and output gate unless a
   measured accuracy study approves a shared scale.
4. Prefer a fused-two-matrix ESIMD dispatch if it can keep separate weights and
   scales, rather than concatenating solely to reduce a launch.
5. Account for the gate activation lifetime: 2048 FP16 elements per token per
   TP rank.
6. Keep the gate application after attention:
   `sigmoid(output_gate) * attention_output`.
7. Pass a standalone microbenchmark before model integration.

If end-to-end runtime does not improve, keep the simpler separate projections.

### Phase 6: trace-driven HD128 attention work

Onyx HD128 currently uses the generic XPU attention path. Gemma4's HD256/HD512
ESIMD kernels and fused QKV/norm/RoPE path cannot be assumed to support HD128.

Only start this work if Phase 4 shows attention is a meaningful fraction of
TTFT or TPOT:

1. Add HD128/QH16/KVH1-per-rank microbenchmarks.
2. Cover both SWA 2048 and global attention.
3. Validate GQA16, NoPE layers, interleaved RoPE, and page size 64.
4. Benchmark the kernel before integrating it.
5. Keep attention/KV dtype independent from linear-weight FP8.

This phase must not delay Level A if generic attention is correct and the main
FP8 goals are already met.

### Phase 7: multimodal FP8

Phase 7 is active. The first image-capable target is implemented and validated:
the existing TP=2 online-FP8 text decoder runs with a replicated BF16 Onyx
vision encoder, adapter, final projection, and perception normalization on both
TP ranks. `OnyxForCausalLM` now uses SGLang's common multimodal embedding
routine, and the model-info endpoint reports `has_image_understanding: true`.

The initial BF16 check was deliberately limited to the vision subsystem rather
than a second full 31B service qualification. A reduced Onyx vision
configuration loaded identical reference/SGLang weights and produced bit-exact
BF16 features (`max_abs=0`, output shape `[4, 128]`). The final-target TP=2
FP8-text/BF16-vision service then passed:

- one-image chat: correctly identified the cat, pink hoodie, and sunglasses;
- two-image chat: correctly identified cat then dog in input order;
- fixed text raw token remained 328, and English/Chinese chat smoke passed;
- 17 affected Onyx/KV unit tests passed.

The same final-target service was rerun on the text task suites after image
integration:

| Suite | Current result | Invalid responses |
|---|---:|---:|
| GSM8K Chat API, 100 examples | 0.90 (90/100) | 0 |
| ARC-Challenge zero-shot Chat API, complete 1,172-example test split | 0.9411 (1103/1172) | 1 |

Both used temperature 0; GSM8K used the Chat harness with `max_tokens=512`.
Artifacts are
`/home/intel/xiangyu/copilot_workspace/onyx_current_fp8_gsm8k_100.log` and
`/home/intel/xiangyu/copilot_workspace/onyx_arc_current_fp8.json`. These are
current-path FP8 results, not a new matched FP16/FP8 comparison, so the formal
Phase 3 quantization deltas remain unchanged.

Current supported boundary:

- image preprocessing, variable-resolution patch expansion, sparse/global
  vision attention, pixel-shuffle downsampling, adapter/projection, multiple
  images, and image-feature insertion are supported;
- video is explicitly rejected by the SGLang processor until frame-group and
  timestamp metadata are represented as video items;
- audio is not an Onyx modality.

The validated load retained all 260 decoder linears as E4M3 FP8 while vision
modules remained BF16. Both ranks loaded in about 42 seconds and reported
17.82 GB model memory per rank. After the one- and two-image smoke requests,
`xpu-smi` reported 26,802 MiB on XPU0 and 26,053 MiB on XPU1, including
allocator-retained runtime buffers.

#### Video gap analysis

Video does not require a new text decoder, adapter, projection, or attention
kernel. The checkpoint reference already defines the token protocol and uses
the same vision encoder after converting each temporal frame group to a
`[vision_patch_temporal * 3, H, W]` tensor. The remaining work is primarily
correctness-sensitive SGLang integration:

1. **Request/processor entry point**
   - `OnyxSGLangProcessor.process_mm_data_async` currently raises
     `NotImplementedError` when `request_obj.video_data` is present.
   - It must pass video data into `load_mm_data` and preserve the distinction
     between IMAGE and VIDEO items.

2. **Training-faithful decode and timestamps**
   - The reference samples at 2 FPS, caps at 96 frames, rounds the selected
     frame count to a multiple of `vision_patch_temporal=2`, and groups every
     two consecutive frames.
   - Each group needs the real decoded PTS of its first frame. The prompt is
     rendered as
     `<|vid_start|> (Time: X.Xs <|video|>*P [separator])* <|vid_end|>`.
     Replacing PTS with a uniform synthetic timeline can differ from training.
   - The reference file-path decoder uses `torchcodec`, which is not installed
     in the current container. Either install the matching dependency or adapt
     SGLang's `VideoDecoderWrapper` while retaining real PTS; silently falling
     back to a decoder/timestamp approximation is not acceptable.

3. **Frame-group feature representation**
   - `OnyxProcessor` currently emits image tensors and all video group tensors
     through the same `pixel_values` field.
   - SGLang maps `pixel_values` to IMAGE by default. The Onyx processor must
     explicitly construct VIDEO items, or rename/register a video feature
     field, so each `[6, H, W]` group is associated with its corresponding
     `<|video|>` span.
   - One logical video creates multiple disjoint feature spans because timestamp
     text and frame separators occur between groups. Multi-video requests must
     retain both group order and video boundaries through cache splitting.

4. **Vision/model dispatch**
   - The checkpoint vision encoder already handles three-channel images and
     six-channel two-frame groups. The SGLang port currently rejects any channel
     count other than three; the reference temporal-patch branch must be
     restored.
   - `OnyxForCausalLM` currently registers only
     `Modality.IMAGE: get_image_feature`. It needs a VIDEO embedding function
     that reuses the same BF16 encoder/adapter/projection path.

5. **Qualification**
   - Compare exact prompt token layout, group count, tokens per group, and
     rendered PTS against the checkpoint processor.
   - Verify early/late event ordering on a controlled clip, then multiple videos
     and mixed image/video/text input.
   - Record encoder memory and latency separately from text prefill/decode.

The hard part is therefore frame/PTS/token-span alignment, not compute-kernel
availability. A successful import, decode, or non-crashing generation is not a
video correctness gate.

#### Bring-up order

1. [x] Run the image-capable model with all vision modules ignored by FP8:
   - `model.vision_encoder`
   - `model.vision_adapter`
   - `model.vision_projection`
   - `model.perception_emb_norm`
2. [x] Generalize `LowMemFp8ModelLoader` vision skipping from Gemma-only
   names to the actual Onyx prefixes.
3. [ ] Inventory and microbenchmark each vision linear, patch embedding,
   adapter, and projection shape.
4. [ ] Quantize one class of vision linear at a time, only after an isolated
   kernel win:
   - vision MLP
   - vision attention projections
   - adapter
   - final vision projection
5. [x] Keep patch embedding and normalization BF16 unless a dedicated
   implementation and accuracy study justify quantization.
6. [x] Maintain an explicit ignore list using actual Onyx module prefixes, not
   Gemma4 names.
7. [ ] Implement the video integration above in processor → item construction →
   six-channel vision dispatch order, then qualify temporal ordering.

#### Required gates

- Text-only results remain unchanged when multimodal code is enabled.
- Image question answering and OCR/reference tasks match the BF16 multimodal
  baseline within a predeclared threshold.
- Video temporal ordering and timestamp behavior match BF16.
- Mixed image/video/text batching is stable.
- Peak memory is measured during both model loading and encoder execution.

## 6. Test inventory

### CPU/unit tests

- Onyx passes `quant_config` to every decoder linear.
- A tiny Onyx config creates the expected FP8/unquantized module split.
- Packed QKV and gate-up weight loaders preserve shard mapping under FP8.
- Ignore-list matching works through packed-module mappings.
- Input embedding remains unquantized.
- LM head remains unquantized FP16.
- Logits multiplier and soft cap run after FP32 promotion.
- Existing strided KV-cache regression remains enabled.

### XPU kernel tests

- Every Section 4 FP8 decoder shape at all selected M values.
- FP16 activation input.
- Numerical comparison, NaN/Inf, and repeatability.
- Intended ESIMD/W8A16 dispatch and explicit fallback.
- Peak temporary allocation for each weight conversion.

### XPU model tests

- TP=2 load twice from clean container state.
- Raw fixed-token parity.
- Multi-token English and Chinese.
- SWA boundary and 16K context.
- Prefix/radix reuse.
- Prompt-logprob path.
- OpenAI chat endpoint and recipient-header post-processing.
- Zero zombies after the required container restart.

## 7. Expected source changes

| Area | Expected change |
|---|---|
| `python/sglang/srt/models/onyx.py` | Minimal Level A changes and later optional fused-QKVG integration |
| `python/sglang/srt/layers/quantization/fp8_utils.py` | Packaged W8A16 import, dispatch diagnostics, strict validation mode |
| `python/sglang/srt/model_loader/loader.py` | Progress reporting and Onyx-aware multimodal skipping |
| `test/registered/unit/models/test_onyx.py` | Quant propagation, module split, loader, and logits tests |
| `llm-scaler/sglang/custom-esimd-kernels` | Onyx shape tests/benchmarks and packaged W8A16 path |
| `sgl-kernel-xpu` | Only if trace-driven HD128 attention work is justified |
| `ONYX_SGLANG_SUPPORT_STATUS.md` | Record each measured result and final launch block |

Formal reusable tests belong in the repositories. One-off logs and diagnostic
scripts must be removed after their evidence is summarized in the status
document.

## 8. Risk register

| Risk | Mitigation |
|---|---|
| FP16 activation is faster but numerically wrong | Qualify unquantized FP16 before FP8 performance work |
| Online quantization changes output-gate behavior | Trace gate projection and post-sigmoid output against pure FP16 |
| Packed projection scales are merged incorrectly | Inspect logical widths and preserve scale boundaries |
| LM head is accidentally quantized | Assert it remains unquantized FP16, matching Gemma4-31B |
| W8A16/ESIMD silently falls back | Positive dispatch evidence and strict diagnostic mode |
| A kernel is fast in isolation but slows the model | Microbenchmark first, then inspect host/API integration overhead |
| `.item()`, `.cpu()`, or `.numpy()` enters forward | Prohibit device-to-host reads in the hot path |
| Loader appears hung during real quantization | Progress markers plus no-progress watchdog |
| Repeated runs inherit bad Level Zero/CCL state | Restart the entire container after every Onyx diagnostic run |
| Shared-node CPU load distorts TPOT | Record load and use matched A-B-A runs |
| HD128 generic attention is blamed without evidence | Trace full runtime before starting a new attention kernel |
| Vision is quantized before BF16 multimodal parity | Keep it ignored until the multimodal baseline passes |

## 9. Candidate bring-up configuration

This is a starting point, not a shippable claim:

```bash
export ZE_AFFINITY_MASK=0,1
export SGLANG_USE_SGL_XPU=1
export SGLANG_XPU_FP8_W8A16_PREFILL=1
export ONYX_SERVER_PORT=31888

python3 -m sglang.launch_server \
  --model-path /llm/workspace/model/onyx-hf \
  --device xpu \
  --tp 2 \
  --dtype float16 \
  --quantization fp8 \
  --load-format layered_fp8 \
  --attention-backend intel_xpu \
  --page-size 64 \
  --max-total-tokens 16384 \
  --swa-full-tokens-ratio 0.25 \
  --context-length 16384 \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --trust-remote-code \
  --model-impl sglang \
  --port "${ONYX_SERVER_PORT}"
```

Run and approve pure FP16 first, then use this FP8+FP16 configuration. Re-enable
overlap and radix features one at a time after correctness; do not combine
several behavioral changes in the first FP8 run.

Port 31888 is reserved for Onyx development in this workspace. Onyx validation
tools default to this port; port 30000 must not be used for new Onyx runs.

## 10. Completion checklist

### Text FP8

- [x] Pure FP16 control frozen and correct
- [x] 260 decoder linears verified FP8
- [x] Input embedding and LM head verified unquantized FP16
- [x] `layered_fp8` completes two clean TP=2 starts
- [x] Raw, chat, task, SWA-boundary, and near-16K gates pass
- [x] Onyx FP8 kernel shape matrix passes
- [x] Packaged W8A16 and ESIMD dispatch proven
- [x] Runtime and memory matrix recorded
- [x] Current FP8 1K/2K/4K/8K input, 512-output matrix recorded
- [x] Candidate launch block reproduced from a clean zero-residual server state

### Multimodal FP8

- [x] BF16 vision subsystem matches the checkpoint reference bit-exactly
- [x] Onyx-specific vision ignore/quantization policy exists
- [x] Single- and multi-image correctness smoke passes on FP8 text + BF16 vision
- [ ] Video correctness and timestamp-ordering gates pass
- [x] Multimodal load/runtime memory is measured
- [x] Current supported and ignored module list is documented
