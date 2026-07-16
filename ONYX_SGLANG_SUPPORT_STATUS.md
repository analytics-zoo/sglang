# Onyx SGLang Support Status

Last updated: 2026-07-17

## Scope

Phase 1 targets the Onyx dense text model on Intel XPU/BMG:

- BF16 inference
- TP=2, eager execution
- hybrid sliding-window/global attention
- Hugging Face-format checkpoint loading

The current implementation excludes execution of the vision/video tower, XPU
graph, and fused TP `[Q|K|V|output_gate]` projection. BF16, FP16, and the
functional online-FP8 text path are covered. The converted checkpoint still
packages all text and vision weights so only one model copy is required.

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
| Targeted unit tests | Complete | 9 tests cover pattern, iRoPE, query scale, RMSNorm, logits dtype, TP loaders, and skipping packaged vision weights |
| Strided KV-cache write | Fixed | Packed-QKV K/V views now bypass the stride-blind ESIMD scatter; a shared memory-pool regression test covers this case |
| Real checkpoint conversion | Complete | `/home/intel/xiangyu/model/onyx-hf`: 2 safetensor shards, 1,437 tensors, about 56 GiB |
| TP=2 XPU validation | Complete | BF16 eager `intel_xpu` prefill and multi-token decode match the reference first token and produce correct English/Chinese answers |
| Pure FP16 control | Initial gate passed | Two independent clean TP=2 starts select reference token 328; English, Chinese, multi-token decode, and 2047/2048/2049 SWA-boundary checks pass |
| Comprehensive precision trace | Implemented | `benchmark/onyx/trace_precision.py` and `TENSOR_DUMP_MODE=all_io` capture ordered parent/leaf module inputs and outputs on both TP ranks |
| Decoder online FP8 | Phase 1 complete | Two clean TP=2 starts validate exactly 260 E4M3 decoder linears and pass raw, English/Chinese chat, and SWA-boundary gates |
| Onyx FP8 fast kernels | Phase 2 complete | All 75 required cases pass numerical/repeatability gates; shape-aware runtime dispatch selects only measured wins |
| Model-level FP8 correctness | Phase 3 complete | GSM8K delta is -1.00 point; ARC-Challenge delta is -0.17 point; radix on/off and 2047-8192-token stability gates pass |

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

A subsequent FP8-only run measured the current TP=2 service with 512 output
tokens. It retained the same real-tokenized input, `/generate`, streaming,
`ignore_eos`, radix-off, two warmups, three trials, and median methodology:

| Input tokens | TTFT (ms) | TPOT (ms) | Decode throughput (token/s) | E2E (s) |
|---:|---:|---:|---:|---:|
| 1,024 | 289.00 | 44.57 | 22.44 | 23.07 |
| 2,048 | 589.13 | 44.87 | 22.29 | 23.51 |
| 4,096 | 1,249.50 | 47.62 | 21.00 | 25.59 |
| 8,192 | 2,453.47 | 47.82 | 20.91 | 26.89 |

Host load remained approximately 1.8-2.1 and process inspection found only the
measured TP=2 schedulers. The raw report is
`/home/intel/xiangyu/copilot_workspace/onyx_runtime_fp8_output512.json`.

Optional QKVG/output-gate fusion is not required for the Level A target because
the qualified FP8 path is already faster end to end. HD128 kernel work is not
started because no full-runtime evidence identifies attention as a material
regression. Multimodal FP8 remains blocked at its explicit prerequisite:
`OnyxForCausalLM` sets `has_vision = False`, skips the packaged vision weights,
and the live endpoint reports `has_image_understanding: false`; there is no
BF16 SGLang image/video baseline against which FP8 could be qualified.

## Resolved KV-cache corruption

Onyx splits K and V from a packed QKV projection. The resulting token rows have
stride `(2304, 1)`, even though each K/V row contains only 128 elements. The XPU
ESIMD KV scatter addressed every source row as `token * row_dim` and did not
consume tensor stride, so it copied unrelated packed-QKV data into the cache.
Both `intel_xpu` and `torch_native` attention use this common writer, explaining
why both backends produced the same wrong token.

`_set_kv_buffer_impl` now uses the fused ESIMD writer only when reshaped K and V
are contiguous. Strided sources use the native indexed scatter. An XPU
microbenchmark confirms bit-exact cache contents for both strided and contiguous
sources.

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

- The single converted checkpoint includes vision weights. Phase 1 SGLang
  inference instantiates only the text model and explicitly skips the vision
  encoder, adapter, and projection weights while loading.
- The first implementation keeps `output_gate_proj` separate from QKV. A later
  fused implementation must replace these parameters rather than retaining both
  layouts, or it would duplicate persistent weight memory.
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
- OpenAI-compatible tool calling requires
  `benchmark/onyx/onyx_tool_chat_template.jinja` and
  `--tool-call-parser onyx`. `tool_choice=auto` is qualified;
  `tool_choice=required` structural constraints are not advertised.
