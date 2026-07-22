# Onyx SGLang Support Status

Last updated: 2026-07-22

## Scope

The current target is the Onyx dense text and image model on Intel XPU/BMG:

- BF16 inference
- TP=2, eager execution
- hybrid sliding-window/global attention
- Hugging Face-format checkpoint loading
- image understanding with a BF16 vision tower and online-FP8 text decoder

Image execution is enabled. Video remains explicitly unsupported pending
frame-group and timestamp integration. XPU graph remains disabled. BF16, FP16,
the online-FP8 text path, and FP8-text/BF16-vision image inference are covered
from the same converted checkpoint.

## Architecture contract

- 52 decoder layers, hidden size 6656, FFN size 19968
- 32 query heads, 2 KV heads, head dimension 128
- GQA group size 16 at TP=2
- 39 sliding-window layers and 13 global/NoPE layers
- inclusive HF window 2048, represented as exclusive RadixAttention offset 2047
- interleaved-pair RoPE (`is_neox_style=False`)
- scaleless Q/K RMSNorm followed by query scale
  `43.7840518911 / sqrt(head_dim)`
- attention output gate `sigmoid(gate) * attention_output`
- offset layer norms use `(1 + weight)`; final norm uses direct `weight`
- independent token embedding and LM head
- logits scaling `0.19611613513818404` and tanh soft cap 20

## Implementation progress

| Item | Status | Notes |
|---|---|---|
| Architecture/checkpoint audit | Complete | Native reference, HF converter, checkpoint metadata, and XPU HD128 backend reviewed |
| Text model implementation | Implemented | `onyx.py` added with exact norms, Q/K scaling, interleaved RoPE/NoPE, output gate, and FP32 logits transform |
| Hybrid SWA model config | Implemented | Onyx architecture and `layer_types` connected to the dual KV-pool path without heterogeneous-head compression |
| HF converter/config metadata | Implemented | Hybrid pattern, context length, and logits metadata exported |
| Targeted unit tests | Complete | 17 affected Onyx/KV tests cover text architecture, TP loading, vision behavior, FP8 validation scope, multimodal normalization, and strided KV writes |
| Strided KV-cache write | Fixed | Packed-QKV K/V views now bypass the stride-blind ESIMD scatter; a shared memory-pool regression test covers this case |
| Real checkpoint conversion | Complete | `/home/intel/xiangyu/model/onyx-hf`: 2 safetensor shards, 1,437 tensors, about 56 GiB |
| TP=2 XPU validation | Complete | BF16 eager `intel_xpu` prefill and multi-token decode match the reference first token and produce correct English/Chinese answers |
| Pure FP16 control | Initial gate passed | Two independent clean TP=2 starts select reference token 328; English, Chinese, multi-token decode, and 2047/2048/2049 SWA-boundary checks pass |
| Comprehensive precision trace | Implemented | `benchmark/onyx/trace_precision.py` and `TENSOR_DUMP_MODE=all_io` capture ordered parent/leaf module inputs and outputs on both TP ranks |
| Decoder online FP8 | Phase 1 complete | Two clean TP=2 starts validate exactly 260 E4M3 decoder linears and pass raw, English/Chinese chat, and SWA-boundary gates |
| Onyx FP8 fast kernels | Phase 2 complete | All 75 required cases pass numerical/repeatability gates; shape-aware runtime dispatch selects only measured wins |
| Model-level FP8 correctness | Phase 3 complete | GSM8K delta is -1.00 point; ARC-Challenge delta is -0.17 point; radix on/off and 2047-8192-token stability gates pass |
| Optional FP8 LM head | Qualified | `--enable-fp8-lm-head` preserves current accuracy and lowers decode TPOT by 2.5%-3.5% in matched A-B-A measurements |
| Multimodal image support | Phase 7 image bring-up complete | Checkpoint-faithful BF16 vision tower feeds the TP=2 online-FP8 decoder; single- and two-image chat smoke plus current text-task reruns passed |
| Multimodal video support | Not implemented | Processor rejects video explicitly until frame grouping and timestamps are carried as video items |

## Multimodal FP8 image bring-up

The initial target keeps the vision path unquantized while retaining the
qualified online-FP8 decoder:

- BF16: `model.vision_encoder`, `model.vision_adapter`,
  `model.vision_projection`, and `model.perception_emb_norm`;
- E4M3 FP8: all 260 text decoder linears;
- FP16: text embeddings, decoder norms, activations, and LM head.

Implementation:

- `onyx_vision.py` implements the checkpoint's patch projection, interpolated
  position embedding, 2D RoPE, sparse/global attention schedule, pixel-shuffle
  downsampling, and adapter.
- `onyx.py` constructs the vision modules, loads their checkpoint weights
  without applying text packed-QKV remapping, and inserts features through
  `general_mm_embed_routine`. Text embeddings are normalized before feature
  insertion so image spans receive only `perception_emb_norm`, matching the
  checkpoint reference rather than applying a second RMSNorm.
- `multimodal/processors/onyx.py` maps the `<|image|>` sentinel to expanded
  `<|patch|>` spans and rejects unsupported video/audio inputs explicitly.
- `model_config.py` now reports Onyx image understanding.
- `LowMemFp8ModelLoader` recognizes the actual Onyx vision prefixes when
  `SGLANG_SKIP_VISION_GPU=1`.

Evidence:

- Reduced reference/SGLang BF16 vision towers with identical weights were
  bit-exact: shape `[4, 128]`, `max_abs=0`, cosine `0.99999994`.
- Both TP ranks loaded in 41.80-41.84 seconds and reported 17.82 GB model memory
  per rank. The live endpoint returned `has_image_understanding: true`.
- A cat image produced: `A cat is shown wearing a pink hoodie and sunglasses.`
- A two-image request produced cat then dog in the supplied order.
- The same service preserved text behavior: fixed raw output token 328 and
  correct France/China chat answers.
- Post-smoke `xpu-smi` memory was 26,802 MiB on XPU0 and 26,053 MiB on XPU1,
  including allocator-retained runtime buffers.
- 17 affected Onyx and KV unit tests passed.

Current-path text accuracy was rerun on this exact TP=2 online-FP8-text plus
BF16-vision service:

| Suite | Current result | Invalid responses |
|---|---:|---:|
| GSM8K Chat API, 100 examples | 0.90 (90/100) | 0 |
| ARC-Challenge zero-shot Chat API, complete 1,172-example test split | 0.9411 (1103/1172) | 1 |

GSM8K used `sglang.test.run_eval`, the Chat API, temperature 0, and
`max_tokens=512`. ARC used `benchmark/onyx/eval_arc_challenge.py`, temperature
0, and the answer-letter-only prompt. Raw artifacts are
`/home/intel/xiangyu/copilot_workspace/onyx_current_fp8_gsm8k_100.log` and
`/home/intel/xiangyu/copilot_workspace/onyx_arc_current_fp8.json`.

These measurements qualify the current optimized multimodal-capable FP8 path;
they are not a new matched FP16/FP8 pair. The Phase 3 table below remains the
formal quantization-delta comparison. The current GSM8K result is therefore not
combined with the old FP16 score to claim a new delta.

The final warm one-image smoke used 894 prompt tokens and completed in
0.819 seconds. The two-image smoke used 1,773 prompt tokens. These are
functional smoke observations, not a controlled performance benchmark.

Validated launch additions:

```bash
export SGLANG_SKIP_VISION_GPU=0
export SGLANG_FP8_IGNORED_LAYERS=\
model.vision_encoder,model.vision_adapter,model.vision_projection,model.perception_emb_norm
```

### Remaining video gap

The checkpoint already contains the required video behavior; no new decoder or
vision-attention kernel is currently indicated. It samples at 2 FPS (up to 96
frames), groups two consecutive RGB frames into each `[6, H, W]` input, and
places the real first-frame PTS before every group:

```text
<|vid_start|>
  Time: X.Xs <|video|> ... <|video|> <|vid_frame_separator|>
  ...
  Time: Y.Ys <|video|> ... <|video|> <|vid_end|>
```

SGLang is missing four integration pieces:

- `OnyxSGLangProcessor` currently rejects `video_data` instead of forwarding it
  through multimodal loading.
- The training-faithful path depends on `torchcodec`, which is absent from the
  current container. An adapter from SGLang's decoder is acceptable only if it
  preserves real PTS and the same sampling/grouping behavior.
- The checkpoint processor returns image tensors and flattened video groups in
  `pixel_values`, while SGLang classifies that field as IMAGE. Video groups need
  explicit VIDEO items and correct mapping from each group tensor to its
  disjoint `<|video|>` token span.
- The current SGLang vision port rejects six-channel frame groups, and
  `OnyxForCausalLM` registers only an IMAGE embedding function. The reference
  temporal-patch branch and VIDEO dispatch still need to be restored.

Required evidence is exact processor token/PTS parity, controlled early/late
event ordering, multi-video and mixed image/video/text stability, and separate
vision-encoder memory/latency. A non-crashing video generation alone is not a
correctness result.

## TP=2 XPU validation

Validated configuration:

- `dtype=bfloat16`, `tp_size=2`, `attention_backend=intel_xpu`
- eager execution, `max_total_tokens=16384`
- hybrid SWA memory enabled with `swa_full_tokens_ratio=0.25`
- CPU-staged `layered_fp8` loader, used without quantization

Correctness results:

- The native checkpoint and converted HF model have bit-exact CPU FP32
  prefill/decode logits.
- For input IDs `[200000, 954, 10810, 323, 4302, 373]`, both the reference
  and SGLang select token 328 (`" to"`). Before the KV fix, SGLang selected
  token 2649 (`"ano"`).
- Layer 0 `intel_xpu` attention output now matches direct PyTorch SDPA at BF16
  trace precision on both TP ranks.
- Multi-token hybrid-SWA decode answers the France-capital and China-capital
  prompts correctly and terminates on the expected end-of-turn token.
- The 9 Onyx model tests and the shared strided-KV regression test pass.

## FP16 control qualification and precision tracing

The FP8 target uses FP16 activations, so an unquantized FP16 run is the direct
correctness control. A CPU HF text-only reference with every parameter converted
to FP16 returns finite FP32 logits and selects token 328 for the fixed raw input.
The earlier report that TP=2 SGLang FP16 selected token 16 is withdrawn. It could
not be reproduced after clean container restarts, and the earlier isolation log
shows a new FP16 server failing to bind port 30000 because another service still
owned it. That run was not an isolated FP16 measurement.

New Onyx server runs use dedicated port 31888. The Onyx validation harness
defaults to `http://localhost:31888`; port 30000 is no longer used for Onyx
development.

Two independent clean TP=2 SGLang FP16 starts now select token 328. The second,
untraced run also passes:

- English multi-token chat: France capital is Paris.
- Chinese multi-token chat: China capital is Beijing.
- Valid one-token outputs at input lengths 2047, 2048, and 2049.

These results qualify the initial pure-FP16 functional control. They are not an
FP8 result and do not approve FP16 or FP8 performance numbers.

The tensor dump forward hook now supports `TENSOR_DUMP_MODE=all_io`. In this
mode it records positional and keyword inputs plus outputs for every parent and
leaf module in execution order, handles repeated module calls without overwriting
keys, detaches and copies tensors to CPU, and flushes one complete pass only
after the root model returns. `ModelRunner` invokes the root `forward` method
directly, so a root `nn.Module` hook does not fire; all-I/O mode therefore wraps
the root `forward` method while retaining ordinary hooks on child modules. The
legacy leaf-output mode remains the default. The focused test covers this direct
root-forward invocation.

`benchmark/onyx/trace_precision.py` drives one deterministic raw request and
captures `Pass00000.pt` for both TP ranks plus run metadata. Its compare command
reports shape, dtype, finite status, RMSE, relative RMSE, cosine similarity,
first threshold crossing, and largest divergences. BF16 and FP16 must be
captured in separate clean-container runs; the dumps synchronize every observed
tensor to CPU and are diagnostic artifacts, not performance measurements.

The clean BF16 and FP16 captures each contain 9,969 tensors per rank and both
select token 328. With diagnostic thresholds of relative RMSE 0.05 and cosine
similarity 0.995:

- TP rank 0 has no threshold crossing.
- TP rank 1 has one crossing at
  `model.layers.43.self_attn.o_proj.input.args.0`: relative RMSE 0.07572 and
  cosine similarity 0.99714. The tensor is finite in both runs.
- Final logits have relative RMSE 0.00730 and cosine similarity 0.99998, remain
  finite, and preserve the reference top token.

There is therefore no reproduced FP16 correctness defect to patch. FP8 work may
advance to decoder online-FP8 functional bring-up, while the broader FP16 task
baseline remains the direct comparison for later accuracy qualification.

## Online FP8 Phase 1 result

On 2026-07-17, the unquantized FP16 control was changed only by adding
`--quantization fp8`. The run retained TP=2, eager execution, `intel_xpu`
attention, FP16 activations, the CPU-staged `layered_fp8` loader, and dedicated
port 31888. W8A16 was not explicitly enabled because its Onyx shape matrix is a
Phase 2 qualification item.

The model now keeps the independent LM head outside `quant_config`. After
online quantization, the loader calls an Onyx-specific strict validator before
the server becomes ready. On both TP ranks and on two consecutive clean starts,
it verified:

- Exactly 52 each of `qkv_proj`, `output_gate_proj`, `o_proj`,
  `gate_up_proj`, and `down_proj`: 260 decoder linears total.
- Every decoder weight is E4M3 FP8 with the expected transposed TP=2 storage
  shape, one finite per-tensor weight scale, and no persisted input scale.
- Input embedding, independent LM head, and all weighted norms remain FP16.
- Per-rank weight memory is 14.24 GB; 17.65 GB remains after weight loading.
- Weight loading and online quantization complete in 32.37-33.02 seconds.

Both clean starts pass the same deterministic validation suite:

- Fixed raw input selects token 328 (`" to"`).
- English chat answers Paris and Chinese chat answers Beijing coherently.
- Inputs of length 2047, 2048, and 2049 return valid tokens 328, 15718, and
  290 respectively, matching the qualified pure-FP16 run.
- Explicit PID cleanup leaves no launch server, scheduler, or detokenizer
  process.

Validated functional launch:

```bash
export ZE_AFFINITY_MASK=0,1
export SGLANG_USE_SGL_XPU=1
export SGLANG_SKIP_VISION_GPU=1

python3 -m sglang.launch_server \
  --model-path /llm/workspace/model/onyx-hf \
  --device xpu --tp 2 \
  --dtype float16 --quantization fp8 --load-format layered_fp8 \
  --attention-backend intel_xpu --page-size 64 \
  --mem-fraction-static 0.95 --max-total-tokens 16384 \
  --swa-full-tokens-ratio 0.25 --chunked-prefill-size 1024 \
  --disable-radix-cache --max-running-requests 1 --context-length 16384 \
  --disable-cuda-graph --disable-custom-all-reduce \
  --disable-overlap-schedule --skip-server-warmup \
  --watchdog-timeout 300 --trust-remote-code --model-impl sglang \
  --random-seed 0 --host 0.0.0.0 --port 31888
```

This completes functional decoder online FP8 only. It does not qualify the
task-level accuracy, 16K correctness, or performance.

## Online FP8 Phase 2 kernel qualification

`benchmark/onyx/bench_fp8_kernels.py` exercises the five TP=2 decoder shapes at
ESIMD `M={1,2,4,8,16,32,64}` and W8A16
`M={65,128,256,1024,2048,4096,8192,16384}`. Each case uses FP16 activation,
online-quantized E4M3 weight, a dequantized FP16 PyTorch reference, repeated
finite/error/cosine checks, and three interleaved fast/fallback timing samples
after shared warmup on one BMG XPU.

All 75 cases pass the numerical and repeatability gates. Sixty-six candidate
fast paths beat the fallback. The nine losing candidates are intentionally not
selected:

- `output_gate_proj` W8A16 at M=65 and 128.
- `gate_up_proj` ESIMD at M=64.
- `down_proj` ESIMD at M=2, 4, 8, 16, 32, and 64; M=1 remains an ESIMD win.

The runtime now imports
`custom_esimd_kernels_sglang.onednn_fp8_gemm_w8a16` instead of a hard-coded
benchmark `.so`, applies the measured shape gates, and supports
`SGLANG_XPU_FP8_STRICT_DISPATCH=1`. A clean strict TP=2 model run logged actual
ESIMD, packaged oneDNN W8A16, and qualified-fallback selections on both ranks.
The same run validated 260 E4M3 linears and retained token 328, coherent
English/Chinese chat, and the 2047/2048/2049 boundary outputs. This completes
Phase 2; the following section records the completed Phase 3 qualification.

## Online FP8 Phase 3 correctness qualification

Pure FP16 and FP8-weight/FP16-activation runs used the same TP=2 eager
`intel_xpu` configuration, deterministic Chat API prompts, and port 31888.
Measured task results:

| Suite | FP16 | FP8 + FP16 | Delta |
|---|---:|---:|---:|
| GSM8K Chat API, 100 examples | 0.88 | 0.87 | -1.00 percentage point |
| ARC-Challenge zero-shot Chat API, complete 1,172-example test split | 0.9411 (1103/1172) | 0.9394 (1101/1172) | -0.17 percentage point |

ARC uses `benchmark/onyx/eval_arc_challenge.py`, temperature 0, and an
answer-letter-only prompt. Both runs have zero invalid responses. Only 9 of
1,172 predictions differ: FP8 loses five FP16-correct cases, gains three cases,
and changes one answer that remains incorrect. The net two-question loss is
consistent with ordinary FP8 quantization noise and is well inside the
predeclared one-percentage-point gate.

The deterministic raw corpus retains the same top token in every case, with
top-five overlap of 4-5. The ten-case task smoke remains 0.9 in both modes.
Multi-turn chat and tool calling pass; the latter returns a structured
`get_weather({"city": "Paris"})` call with the Onyx parser and chat template.

Hybrid-SWA and cache qualification covers 2047, 2048, 2049, 4096, and 8192
input tokens. All outputs are finite and valid, radix-off passes all lengths,
and radix-on also passes all lengths plus deterministic repeated-prefix reuse.
The former 2049 radix hang was an admission-budget bug: after locking a
2048-token cached prefix, `PrefillAdder` reserved the complete sliding window a
second time and rejected the request forever. SWA budgeting now subtracts the
window already protected by the matched prefix; the focused unit test covers
zero, partial, and complete prefix coverage. The fixed 2049 request reuses all
2048 cacheable tokens and returns the same token as a cold request.

At 4096 tokens FP16 selects token 1394 (`" This"`) while FP8 selects token 3231
(`"This"`). FP16 log probabilities are -0.1754 and -1.9574 respectively; FP8
gives -0.7588 and -0.6787. This is recorded as a quantization-induced token
flip, not a release failure: the output remains valid and deterministic, and
both independent task suites satisfy their accuracy thresholds. Long-context
tests are stability/SWA gates and do not require exact FP16/FP8 token identity.

Phase 3 is complete. Performance measurements may proceed under Phase 4.

## Online FP8 Phase 4 performance and memory qualification

Phase 4 used `benchmark/onyx/bench_runtime.py` with real Onyx-tokenized text,
`/generate`, streaming, `ignore_eos`, batch size 1, 64 output tokens, two
warmups, three trials, and medians. Radix cache was disabled. Pure FP16 and
FP8-weight/FP16-activation services were restarted in A-B-A order with all
other runtime parameters matched. The table uses the midpoint of the two FP16
medians and the intervening FP8 run:

| Input tokens | FP16 TTFT (ms) | FP8 TTFT (ms) | FP8 TTFT delta | FP16 TPOT (ms) | FP8 TPOT (ms) | FP8 TPOT improvement |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 284.03 | 288.99 | +1.75% | 56.68 | 45.40 | 19.90% |
| 2,048 | 577.33 | 589.88 | +2.17% | 56.40 | 45.00 | 20.22% |
| 4,096 | 1,232.48 | 1,256.40 | +1.94% | 59.82 | 47.74 | 20.19% |
| 8,192 | 2,438.62 | 2,456.55 | +0.74% | 59.59 | 47.94 | 19.55% |
| 16,000 | 4,866.40 | 4,868.28 | +0.04% | 59.23 | 48.32 | 18.42% |

The 16,000-token point leaves room for generation under the model's 16,384
maximum. A 16,320-input plus 64-output request at the exact boundary was not
accepted, so it is not used as a performance result. FP8 end-to-end latency is
lower at every measured length because the 18.4%-20.2% decode gain outweighs
the at-most 2.17% TTFT increase. Host load was approximately 2 during the
matched runs, and process inspection found only the measured TP=2 schedulers.

Weight loading takes 18.1 s and 26.01 GB/rank for FP16 versus 32.7 s and
14.24 GB/rank for FP8. The FP8 startup cost is online weight conversion.
Independent startup sampling found no transient above steady device usage:
FP16 uses 28,198/27,834 MiB and FP8 uses 16,547/16,182 MiB on GPU0/GPU1.
This is a 41.3%-41.9% steady-memory reduction, while preserving the declared
16K capacity. Phase 4 is complete.

A subsequent FP8-only run measured the pre-optional-optimization TP=2 service
with 512 output tokens. It retained the same real-tokenized input, `/generate`,
streaming, `ignore_eos`, radix-off, two warmups, three trials, and median
methodology:

| Input tokens | TTFT (ms) | TPOT (ms) | Decode throughput (token/s) | E2E (s) |
|---:|---:|---:|---:|---:|
| 1,024 | 289.00 | 44.57 | 22.44 | 23.07 |
| 2,048 | 589.13 | 44.87 | 22.29 | 23.51 |
| 4,096 | 1,249.50 | 47.62 | 21.00 | 25.59 |
| 8,192 | 2,453.47 | 47.82 | 20.91 | 26.89 |

Host load remained approximately 1.8-2.1 and process inspection found only the
measured TP=2 schedulers. The raw report is
`/home/intel/xiangyu/copilot_workspace/onyx_runtime_fp8_output512.json`.

Level A remains complete, but a source audit against Gemma4 identified an
ordered optional optimization queue:

1. qualify the existing `sgl_kernel.fused_qk_norm_rope` for Onyx FP16/HD128,
   interleaved RoPE, constant query scale, and NoPE;
2. qualify `esimd_norm_add_norm` and `esimd_rmsnorm_residual_scalar` for the two
   decoder residual/norm boundaries;
3. measure and, only if beneficial, make `esimd_kv_scatter` consume the packed
   QKV source row stride;
4. qualify `esimd_gemv_fp8_pert_fused2` for bsz=1 QKV plus output-gate decode;
5. trace the resulting model before deciding whether new HD128 prefill/decode
   attention kernels are justified.

This ordering prefers existing kernels and low-risk local fusion before new
attention work. It does not reopen Level A correctness. All five investigations
are complete; the trace did not justify a new HD128 attention kernel.

The item-1 standalone HD128 microbenchmark passes RoPE and NoPE at M=1/64/1024:
Q/K cosine is at least 0.99999988, relative RMSE is at most 0.000515, and the
fused kernel is 1.91-2.51x faster than the composed path. Matched A/B/A model
runs with `SGLANG_ONYX_DISABLE_FUSED_QK_NORM_ROPE=1` as B show 9.92%-10.79%
lower TPOT in A and 12.93%-13.74% lower TPOT in A2 across 1K-8K inputs. The
fused exact path scores 96/100 on GSM8K Chat with zero invalid responses and
passes the complete Onyx smoke/boundary/repeatability/tool capture. Raw reports
are `onyx_fused_qk_norm_rope_microbench.json`,
`onyx_runtime_fused_qk_a.json`, `onyx_runtime_qk_baseline_b.json`, and
`onyx_runtime_fused_qk_a2.json` under
`/home/intel/xiangyu/copilot_workspace/`.

For item 2, standalone M=1 hidden-size-6656 measurements show cosine at least
0.99999988 and relative RMSE at most 0.000499. `esimd_norm_add_norm` is 1.55x
faster than post-attention norm + residual + pre-FFN norm, while
`esimd_rmsnorm_residual_scalar` with scalar 1 is 1.33x faster than post-FFN norm
+ residual. Matched A/B/A integration lowers TPOT by 9.26%-9.93% in A and
9.09%-10.59% in A2, with no material TTFT change. The fixed token remains 328,
task smoke remains 0.9, all 2047/2048/2049/4096/8192 boundaries are valid,
4096 still selects token 1394, and repeatability, multi-turn, and tool calling
pass. The opt-out is `SGLANG_ONYX_DISABLE_RESIDUAL_NORM_FUSION=1`; raw reports
are `onyx_residual_norm_microbench.json`,
`onyx_runtime_residual_norm_fused_a.json`,
`onyx_runtime_residual_norm_baseline_b.json`, and
`onyx_runtime_residual_norm_fused_a2.json`.

For item 3, the original native strided K/V scatter takes 53.44 us at M=1.
Explicit contiguous copies plus the existing fused kernel take 42.27 us, which
proved that a stride-aware kernel was worth implementing. The ESIMD kernel now
consumes the independent K and V source row strides directly. It is bit-exact
for Onyx stride `(2304, 1)` and takes 41.75-42.01 us at M=1/64/1024, about
20%-21% below native scatter. Matched A/B/A integration lowers TPOT by
2.44%-3.81% in A and 0.86%-3.99% in A2. The complete model capture passes.
`SGLANG_XPU_DISABLE_ESIMD_KV_SCATTER=1` selects the native comparison path.

For item 4, `esimd_gemv_fp8_pert_fused2` preserves independent QKV and
output-gate scales and is bit-exact against two separate ESIMD GEMVs. The
standalone result is 51.55 us versus 54.22 us (1.05x). Integration reuses the
existing per-layer `_esimd_t` layouts rather than allocating an additional
packed-weight format. Matched A/B/A lowers TPOT by 3.25%-4.34% in A and
3.44%-5.23% in A2, with the full model capture passing. The opt-out is
`SGLANG_ONYX_DISABLE_FUSED_QKV_GATE_GEMV=1`.

For item 5, a final 1K-input/64-output unitrace captured one warmup and one
measured request on both TP ranks. Generic HD128 `XeFMHAFwdKernel` attention
accounts for 161.9 ms of 4,308.5 ms GPU self-time on TP0 (3.76%) and 162.0 ms
of 4,340.8 ms on TP1 (3.73%). GEMV and PCIe all-reduce dominate instead.
Attention therefore fails the predeclared material-cost gate, so no new HD128
prefill/decode kernel is started. Traces are under
`/home/intel/xiangyu/copilot_workspace/onyx_unitrace_final2/`.

After completing items 1-4, the fully optimized FP8 path was remeasured with
the same 512-output methodology:

| Input tokens | TTFT (ms) | TPOT (ms) | Decode throughput (token/s) | E2E (s) |
|---:|---:|---:|---:|---:|
| 1,024 | 286.89 | 34.17 | 29.26 | 17.75 |
| 2,048 | 584.51 | 34.34 | 29.12 | 18.13 |
| 4,096 | 1,233.58 | 37.45 | 26.70 | 20.37 |
| 8,192 | 2,433.75 | 37.14 | 26.93 | 21.41 |

The machine-readable report is
`/home/intel/xiangyu/copilot_workspace/onyx_runtime_optimized_fp8_output512.json`.

### Optional FP8 LM head

`--enable-fp8-lm-head` is default-off and applies only to single-token XPU
decode. Each TP rank lazily quantizes its `[101024, 6656]` independent LM-head
shard to E4M3 with one FP16 scale per output channel, then dispatches
`esimd_gemv_fp8_pern`. Multi-token logits retain the original FP16 path, and the
original FP16 weight remains resident.

An isolated BMG microbenchmark at the exact per-rank Onyx shape measured
1.1367 ms for FP8 versus 2.2614 ms for FP16 (1.989x). Across five random inputs,
the minimum cosine similarity was 0.99964762, maximum relative L2 error was
0.02654906, and all five argmax results matched.

The Onyx correctness qualification passed:

- fixed raw output token 328 and coherent Paris/Beijing chat;
- exact 2047/2048/2049 boundary tokens 328/15718/290;
- task smoke 0.9, valid 4096/8192 outputs, deterministic prefix reuse,
  multi-turn chat, and tool calling;
- GSM8K Chat API 0.90 (90/100), equal to the current-path result;
- ARC-Challenge 0.9420 (1104/1172), one correct answer above the current-path
  result, with the same one invalid response.

Matched A-B-A runtime measurements used the qualified launch script,
`benchmark/onyx/bench_runtime.py`, 512 output tokens, two warmups, three trials,
and medians. `ON midpoint` is the midpoint of the two FP8-LM-head runs:

| Input tokens | FP16 LM-head TPOT (ms) | FP8 LM-head ON midpoint (ms) | Improvement |
|---:|---:|---:|---:|
| 1,024 | 34.693 | 33.540 | 3.32% |
| 2,048 | 34.942 | 33.728 | 3.47% |
| 4,096 | 37.517 | 36.490 | 2.74% |
| 8,192 | 37.545 | 36.598 | 2.52% |

The complete current-path matrix, including a 16,000-token input and 512-token
output, was measured in one server run with
`ONYX_CONTEXT_LENGTH=16896`, `ONYX_MAX_TOTAL_TOKENS=17152`, and
`ONYX_ALLOW_LONG_CONTEXT=1`:

| Input tokens | TTFT (ms) | TPOT (ms) | Decode throughput (token/s) | E2E (s) |
|---:|---:|---:|---:|---:|
| 1,024 | 285.49 | 34.55 | 28.94 | 17.94 |
| 2,048 | 582.46 | 34.51 | 28.97 | 18.22 |
| 4,096 | 1,187.29 | 37.28 | 26.83 | 20.24 |
| 8,192 | 2,421.31 | 37.44 | 26.71 | 21.56 |
| 16,000 | 4,859.20 | 37.30 | 26.81 | 23.92 |

This is an absolute FP8-LM-head-ON matrix, not an A-B-A comparison; the OFF
path was not rerun with the long-context allocation.

TTFT changed by -0.37% to -0.58%, with no regression. After the same clean
startup and smoke, device memory increased by 753-754 MiB per rank; the FP8
weight and scale account for about 641.5 MiB, with the remainder attributable to
allocator retention and quantization temporaries. The declared 16,384-token
capacity remains unchanged.

Raw artifacts are:

- `onyx_fp8_lm_head_qualification.json`
- `onyx_fp8_lm_head_gsm8k_100.log`
- `onyx_fp8_lm_head_arc_challenge.json`
- `onyx_runtime_fp8_lm_head_on_output512.json`
- `onyx_runtime_fp8_lm_head_off_output512.json`
- `onyx_runtime_fp8_lm_head_on_a2_output512.json`
- `onyx_runtime_fp8_lm_head_on_1k_16k_output512.json`

All are under `/home/intel/xiangyu/copilot_workspace/`.

The former multimodal prerequisite is now satisfied for images:
`OnyxForCausalLM` loads the BF16 vision modules and reports
`has_image_understanding: true` while retaining the optimized online-FP8 text
path. Video integration and vision-linear FP8 qualification remain open.

## Resolved KV-cache corruption

Onyx splits K and V from a packed QKV projection. The resulting token rows have
stride `(2304, 1)`, even though each K/V row contains only 128 elements. The XPU
ESIMD KV scatter addressed every source row as `token * row_dim` and did not
consume tensor stride, so it copied unrelated packed-QKV data into the cache.
Both `intel_xpu` and `torch_native` attention use this common writer, explaining
why both backends produced the same wrong token.

The original correctness fix restricted the fused writer to contiguous K/V.
The optimized writer now reads each tensor's source row stride, so packed-QKV
views and contiguous inputs both use one bit-exact ESIMD scatter. The native
indexed scatter remains available when the kernel or shape is unsupported and
through `SGLANG_XPU_DISABLE_ESIMD_KV_SCATTER=1`.

## Converted checkpoint

The TP=1 native BF16 checkpoint was converted to:

`/home/intel/xiangyu/model/onyx-hf`

Validation results:

- 2 safetensor shards, 59,553,431,312 bytes total
- 1,437 unique tensor keys: 628 text tensors and 809 vision tensors
- representative text embedding, Q/K/V/gate, final-layer MLP, final norm, and
  LM-head shapes match the architecture
- all 806 vision encoder tensors, 2 vision adapter tensors, and the vision
  projection tensor are present in BF16; representative shapes were checked
- `OnyxForCausalLM`, complete model (`has_vision=true`), context length 16384
- 39 sliding-window layers and 13 full/NoPE layers
- SGLang resolves the native `sglang.srt.models.onyx.OnyxForCausalLM`
- hybrid SWA enabled and heterogeneous-head compression disabled
- tokenizer vocabulary size 202048, canonical control-token IDs preserved,
  and BOS is inserted automatically
- the converter's missing `model.rotary_emb.freqs` notice is expected: this
  derived buffer is recomputed before saving and SGLang skips it while loading
- config SHA256:
  `b37e118de9a51d901656e7c711f8406b96035aee534e3d049f8441d3856a9372`
- safetensors index SHA256:
  `5df81addbcdaf2504fc6f309a0d18654d8a5ec6969239790d061f143591d0440`
- conversion log: `/home/intel/xiangyu/model/onyx-hf-conversion.log`

## Current limitations

- Image inference loads the vision encoder, adapter, projection, and perception
  normalization in BF16. Video remains explicitly rejected until real PTS,
  frame-group VIDEO items, disjoint token spans, and six-channel temporal patch
  dispatch are implemented.
- Decode fuses the QKV and output-gate FP8 GEMVs while preserving their separate
  scales and reusing the existing per-layer packed layouts; the checkpoint
  parameters remain logically separate without an additional persistent copy.
- Onyx head dimension 128 currently uses the generic XPU attention path rather
  than Gemma4-specific HD256/HD512 ESIMD kernels.
- `is_hybrid_swa_compress` remains disabled because Onyx uses two KV heads in
  both sliding and global layers; it does not have Gemma4's heterogeneous KV
  head layout.
- The default GPU-resident loader can intermittently block in a large embedding
  H2D copy. The CPU-staged `layered_fp8` loader completes in about 23 seconds per
  TP rank and is the validated path.
- Distributed initialization can still intermittently block before weight
  loading. Diagnostic runs use a 30-second phase watchdog and restart the whole
  container on this condition.
- OpenAI-compatible tool calling uses the tracked
  `benchmark/onyx/onyx_tool_chat_template.jinja` plus
  `--tool-call-parser onyx`; the launch script passes that template explicitly
  so a converted checkpoint cannot retain a stale integration prompt. The
  template renders standard
  `assistant.tool_calls` history, resolves a subsequent `tool_call_id` to its
  function name, and retains the legacy `recipient` input for compatibility.
- `tool_choice=auto`, native `required`, and named-function constraints are
  qualified on the TP=2 online-FP8 service. Streaming emits the function name
  followed by incremental argument fragments. A complete standard OpenAI
  round trip returns `get_weather({"city":"Paris"})`, accepts a tool result
  carrying only `tool_call_id`, and produces the final answer
  `The current temperature in Paris is 18°C.` with `finish_reason=stop`.
- `tool_choice=auto` supports one or more native recipient calls in one OpenAI
  response and honors the protocol default `parallel_tool_calls=true`.
  `required` and named choice retain the native single-call structural
  constraint and therefore require `parallel_tool_calls=false`; when that field
  is omitted, the server preserves compatibility by selecting single-call mode.
- Native calls are separated by tokens that the checkpoint also declares as
  EOS. Auto-parallel requests ignore EOS for tool-recipient output, but honor
  the first EOS when the generated output starts with the direct-user prefix.
  After one or more calls, bounded transition regexes stop when the model moves
  to either a user response or a tool-result block. This avoids an
  answer-length regex and full-output re-decode on every decode step. The parser
  accepts repeated assistant recipient blocks, assigns response-local indexes
  `0..N-1`, and suppresses the generated transition header. A zero-call
  response preserves the complete native `to=user` body.
- The final deterministic OpenAI matrix passes 8/8 cases: four zero-call
  arithmetic/translation/explanation/stable-knowledge prompts, two single-call
  weather/time prompts, one repeated-function two-call prompt, and one
  two-function parallel prompt. The streaming subset passes zero, one, and two
  calls with unique IDs, incremental arguments, and the expected
  `stop`/`tool_calls` finish reasons. A 3,359-character zero-call response
  stops on token 200008 without protocol leakage, and a client-provided stop
  string remains trimmed normally.
- Incoming OpenAI history may also contain parallel calls: the server expands
  each multi-call assistant message and its matched tool results into sequential
  native recipient/tool pairs before rendering the Onyx prompt.
- The unmodified `/llm/workspace/20k.json` is the long-history delivery gate. A
  direct streaming `/v1/chat/completions` request passes in 8.87 seconds at
  19,258 prompt plus 74 completion tokens, returning one response-local index-0
  `run_shell_command` with the expected foreground TypeScript/Phaser dependency
  install. With the default `sampling_defaults=model`, the model's explicit
  `do_sample=false` generation default is honored as greedy decoding when the
  request omits `temperature`.
- The tool-selection rules remain an integration prompt, not a recovered Onyx
  training prompt. They explicitly reserve tools for requested external
  information/actions, reject unrelated calls for arithmetic, translation,
  explanations, and stable knowledge, and tell the model to emit each necessary
  independent recipient call before replying.
