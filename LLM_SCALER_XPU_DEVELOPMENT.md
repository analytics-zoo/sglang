# Intel XPU Development Notes for sglang (llm-scaler)

Scope: running sglang on Intel Arc Pro B60 (Battlemage, Xe2) with focus on
Qwen3.5 hybrid linear-attention models. This doc consolidates the architecture
understanding and open issues discovered while bringing up this stack end-to-end.

Related reference docs (outside this repo, but cross-linked here):
- `/home/intel/xiangyu/docker_sgl_xpu/qwen35_xpu_notes.md` — Qwen3.5 XPU issues inventory
- `/home/intel/xiangyu/docker_sgl_xpu/qwen35_pipeline.md` — Qwen3.5-0.8B forward pipeline
- `/home/intel/xiangyu/sgl-kernel-xpu/KERNELS.md` — sgl-kernel-xpu op inventory
- `/home/intel/xiangyu/llm-scaler/sglang/custom-esimd-kernels-sglang/KERNELS.md` — ESIMD kernels inventory

---

## 1. Platform snapshot

```
Host                : Linux 6.17 · 7× Intel Arc Pro B60 (Battlemage / Xe2 / bmg-g21)
Driver              : libze-intel-gpu (intel-graphics PPA)
oneAPI              : 2025.3.3 (icpx, sycl-ls)
PyTorch             : 2.11.0+xpu
triton-xpu          : 3.7.0
Python              : 3.12

Container image     : amr-registry.caas.intel.com/intelanalytics/llm-scaler-omni:torch211-sglang
                      (our local build at /home/intel/xiangyu/docker_sgl_xpu/Dockerfile)
Dev container       : txytest_sgl_0427   (mounts /home/intel/xiangyu -> /llm/workspace,
                                          /home/intel/LLM -> /llm/models,
                                          /dev/dri, group-add video)
```

Models available (`/home/intel/LLM/...`) include `Qwen3.5-0.8B`, `Qwen3-0.6B`,
`Qwen3-32B`, `Qwen3.5-27B-GPTQ-Int4`, `Qwen3.5-35B-A3B*`, etc.

---

## 2. Kernel-stack architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ sglang   (python/sglang/srt/…)  — model code, scheduler, memory pool         │
├──────────────────────────────────────────────────────────────────────────────┤
│ Python-level kernel packages (all installed into the same venv):             │
│                                                                              │
│ ┌───────────────────────┐  ┌─────────────────────────┐ ┌───────────────────┐ │
│ │ sgl_kernel            │  │ vllm_xpu_kernels._xpu_C │ │ custom_esimd_     │ │
│ │ (sgl-kernel-xpu)      │  │  (vllm-xpu-kernels)     │ │ kernels_sglang    │ │
│ │                       │  │                         │ │                   │ │
│ │ • cutlass-sycl FMHA   │  │ • gdn_attention         │ │ • ESIMD decode    │ │
│ │   (decode/prefill/    │  │   (conv1d + chunked-    │ │   hot paths       │ │
│ │    chunkprefill)      │  │    GDR, Xe2 AOT)        │ │ • esimd_gdn_conv  │ │
│ │ • MLA decode          │  │ • cutlass-based group   │ │   _fused (decode) │ │
│ │ • Grouped MoE GEMM    │  │   gemm, flash-attn var  │ │ • eagle_gdn       │ │
│ │ • fused_qk_norm_rope  │  │ • cutlass_*_scaled_mm   │ │   (≤4 tok spec)   │ │
│ │ • FP8/INT4 GEMM/GEMV  │  │ • bgmv_* (LoRA)         │ │ • rms_norm_gated  │ │
│ │ • RMSNorm/Gemma/RoPE  │  │                         │ │ • MoE batch ops   │ │
│ │ • MoE helpers         │  │                         │ │   (FP8 / INT4)    │ │
│ │ • FlashInfer samplers │  │                         │ │ • qkv_split_norm  │ │
│ │ • Speculative / tree  │  │                         │ │   _rope           │ │
│ │                       │  │                         │ │ • FP8 GEMM M>=1   │ │
│ └───────────────────────┘  └─────────────────────────┘ └───────────────────┘ │
│       torch.ops.sgl_kernel       torch.ops._xpu_C       torch.ops.*          │
│                                                        (custom_esimd_       │
│                                                         kernels_sglang,     │
│                                                         eagle_ops, moe_ops, │
│                                                         moe_int4_ops)       │
├──────────────────────────────────────────────────────────────────────────────┤
│ triton-xpu 3.7.0   (sglang's default XPU GDN/MoE/gating kernels)             │
│ torch.xpu          (PyTorch upstream XPU path — empty_cache, get_device_…)   │
│ oneDNN / oneMKL / oneCCL / IPEX                                              │
│ cutlass-sycl (FetchContent'd by sgl-kernel-xpu + vllm-xpu-kernels)           │
├──────────────────────────────────────────────────────────────────────────────┤
│ Level Zero → compute-runtime → libze_intel_gpu → hardware                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Division of labor (practical)

| Kernel class                          | Who owns it                                            |
|---------------------------------------|---------------------------------------------------------|
| Full attention (paged/chunked/decode) | `sgl_kernel` (cutlass-sycl) + `vllm_xpu_kernels` (paged)|
| MLA decode (DeepSeek-style)           | `sgl_kernel` (cutlass-sycl MLA)                         |
| Linear-attn prefill (GDN chunked)     | `vllm_xpu_kernels.gdn_attention` (Xe2 chunked)          |
| Linear-attn decode (GDN recurrent)    | Triton-XPU ( `fused_sigmoid_gating_delta_rule_update` ) |
| Linear-attn decode (optimized shapes) | ESIMD `esimd_gdn_conv_fused[_seq]` (Qwen3-Next / 3.5-35B)|
| MoE grouped GEMM                      | `sgl_kernel.moe_grouped_mm_nt_xe20` (cutlass)           |
| MoE fused E2E pipelines               | ESIMD `moe_forward_full[_int4]`                         |
| Quantization (FP8/INT4/FP4)           | `sgl_kernel` (point ops) + ESIMD (GEMM/GEMV variants)   |
| RMSNorm / Gemma-RMSNorm / RoPE        | `sgl_kernel` + ESIMD fused variants                     |
| Sampling / speculative                | `sgl_kernel`                                            |

### Fork / upstream status

| Repo                                          | Upstream                                          | Ours / state |
|-----------------------------------------------|---------------------------------------------------|--------------|
| `sglang`                                      | `github.com/analytics-zoo/sglang` (branch `xpu_main`, HEAD `32c351381`) | Editable-installed into `txytest_sgl_0427`. Several XPU-specific patches live in the working tree (see §6). |
| `sgl-kernel-xpu` (new fork)                   | `github.com/analytics-zoo/sgl-kernel-xpu` (remote `origin`) + `github.com/sgl-project/sgl-kernel-xpu` (remote `upstream`, base HEAD `c668bb6`) | Branch `intel/esimd-gdn-integration`, uncommitted working tree carrying ESIMD + gdn_attn vendor-in (§5). |
| `vllm-xpu-kernels`                            | `github.com/vllm-project/vllm-xpu-kernels` @ `4c83144` | Used as vendor source; we also have a parallel in-container build in progress to run its `tests/gdn_attn/test_gdn_attn.py`. |
| `custom-esimd-kernels-sglang`                 | fork of `custom-esimd-kernels-vllm` under `llm-scaler/sglang/` | Renamed from `-vllm` → `-sglang`; build artifacts exist but we are phasing it out in favor of folding into `sgl-kernel-xpu`. |

---

## 3. Qwen3.5-0.8B forward pipeline (text-only)

Full trace in `docker_sgl_xpu/qwen35_pipeline.md`. Short version:

```
Engine.generate
 └─ Scheduler → ModelWorker → ModelRunner.forward_extend
     └─ Qwen3_5ForConditionalGeneration.forward         (multimodal wrapper)
         └─ general_mm_embed_routine(language_model=Qwen3_5ForCausalLM)
             └─ for layer in layers[0..23]              (24 layers for 0.8B)
                 ├─ layer_types[i] == "linear_attention" (18 layers)
                 │    Qwen3_5LinearDecoderLayer.forward
                 │     ├─ input_layernorm (GemmaRMSNorm)
                 │     ├─ Qwen3_5GatedDeltaNet.forward
                 │     │    ├─ in_proj_qkvz, in_proj_ba                    (Linear)
                 │     │    ├─ fused_qkvzba_split_reshape_cat_contiguous   (Triton)
                 │     │    ├─ RadixLinearAttention                        (attn layer)
                 │     │    │    └─ GDNAttnBackend.forward_extend
                 │     │    │         ├─ causal_conv1d_fn                  (Triton)
                 │     │    │         ├─ fused_gdn_gating(A_log, a, b, dt) (Triton)
                 │     │    │         └─ chunk_gated_delta_rule(q,k,v,g,β) (⚠ Triton-XPU fails → PyTorch fallback)
                 │     │    ├─ RMSNormGated                                (Triton; _get_sm_count patched)
                 │     │    └─ out_proj                                    (Linear)
                 │     ├─ post_attention_layernorm (GemmaRMSNorm)
                 │     └─ Qwen2MoeMLP (dense: gate_up_proj → silu_and_mul → down_proj)
                 │
                 └─ layer_types[i] == "full_attention" (6 layers at idx 3/7/11/15/19/23)
                      Qwen3_5AttentionDecoderLayer.forward
                       ├─ qkv_proj → split (Q|gate|K|V) (attn_output_gate=True)
                       ├─ GemmaRMSNorm on q and k (_apply_qk_norm)
                       ├─ RoPE (NeoX style)
                       ├─ RadixAttention → XPUAttentionBackend.forward_extend   (cutlass-sycl)
                       ├─ attn_output *= sigmoid(gate)
                       ├─ o_proj
                       └─ Qwen2MoeMLP
```

Key numbers for Qwen3.5-0.8B:
- 24 layers = 18 linear-attn + 6 full-attn
- Linear-attn: `H_k = H_v = 16`, `head_k_dim = head_v_dim = 128`, `conv_kernel_dim = 4`
- Full-attn: 8 Q heads, 2 KV heads, `head_dim = 256`, `attn_output_gate = True`

---

## 4. Known issues & current state

### 4.1 GDN extend (linear-attn prefill) — Triton-XPU can't compile

**Failure** (from `fla/chunk_delta_h.py::chunk_gated_delta_rule_fwd_kernel_h_blockdim64`):

```
note: Pipeline failed while executing
  [`TritonIntelStrideVersioning` on 'builtin.module' operation]
RuntimeError: PassManager::run failed
```

Root cause: this kernel uses heavy `tl.make_block_ptr` + `boundary_check` +
`tl.trans` + `tl.dot` patterns. Intel Triton 3.7.0's MLIR pass pipeline fails
on `stride-versioning`, and downstream TTGIR passes fail even if that pass is
bypassed.

**Current workaround** (in place): pure-PyTorch fallback at
`python/sglang/srt/layers/attention/fla/chunk_torch_xpu.py`, wired through
`layers/attention/linear/kernels/gdn_triton.py::TritonGDNKernel.extend`'s
`is_xpu()` branch.
- Correct but **very slow** (per-token Python loop, ~108 einsums/token × 18 layers).
- Usable only for short-prompt smoke tests.

**Planned real fix**: switch to the native SYCL `torch.ops.sgl_kernel.gdn_attention`
once the `sgl-kernel-xpu` fork build is verified. Schema (vendored 1:1 from
`vllm-xpu-kernels`):
```python
torch.ops.sgl_kernel.gdn_attention(
    core_attn_out, z, qkvz, ba,
    num_k_heads, num_v_heads, head_k_dim, head_v_dim,
    conv_state, ssm_state, conv_weights, conv_bias,
    activation, A_log, dt_bias,
    num_prefills, num_decodes, has_initial_state,
    non_spec_query_start_loc, non_spec_state_indices_tensor,
    num_actual_tokens, tp_size,
)
```
Xe2 fast path goes through `chunk_gated_delta_rule_xe2` with `chunk_size=64`
(matching FLA_CHUNK_SIZE).

### 4.2 Radix cache vs Mamba vs Intel XPU — three-way incompatibility

**Symptom** at startup:
```
Qwen3_5ForConditionalGeneration with radix cache requires page_size=1 in the current
Mamba scheduling mode (no_buffer), but got 64. Automatically setting page_size=1.
Disabling overlap schedule since mamba no_buffer is not compatible with overlap schedule...
Intel XPU attention backend only supports page_size of 32, 64 or 128, changing page_size from 1 to 128.
```

**Mechanics** (`server_args.py:2306-2396`):
1. Mamba/GDN state is a compressed recurrent state → radix-prefix reuse needs checkpoints.
2. `mamba_scheduler_strategy=no_buffer` snapshots every token → forces `page_size=1`.
3. `intel_xpu` attention backend only accepts `page_size ∈ {32,64,128}` (`server_args.py:2636-2641`); page_size=1 gets bumped to 128 → silently breaks MambaRadixCache.
4. `extra_buffer` mode (`mamba_track_interval=256`, checkpoints every FLA_CHUNK_SIZE tokens) would solve it — but is gated by `assert is_cuda()` (`server_args.py:2343-2345`).

**Current workaround**: pass `disable_radix_cache=True`. Cost: no cross-request prefix reuse; acceptable for bench/offline, painful for serving.

### 4.3 `_get_sm_count` hardcoded to CUDA

`fla/layernorm_gated.py::_get_sm_count` called `torch.cuda.get_device_properties(device).multi_processor_count`. On XPU this raised
`AssertionError: Torch not compiled with CUDA enabled`.

**Fixed** (in sglang working tree): added `device.type == "xpu"` branch using `torch.xpu.get_device_properties(device).max_compute_units` (closest analog to CUDA SMs on Xe hardware).

### 4.4 Fallback performance

The PyTorch GDN extend fallback scales poorly:
- Short prompt (~10 tokens) + 96-token decode: usable (~40 s including model load).
- Anything larger: scheduler pins a CPU core busy-looping Python while the GPU idles.

ROI ordering for a real fix:
1. **Wire `torch.ops.sgl_kernel.gdn_attention`** — expected 10–50× speedup per extend kernel.
2. (If native kernel isn't ready) rewrite the fallback in chunked PyTorch (matmuls along `chunk_size=64`), still no Triton. ~20 LoC, maybe 10× improvement.
3. Upstream fix to Triton-XPU block-pointer pipeline — long term.

---

## 5. `sgl-kernel-xpu` fork integration (branch `intel/esimd-gdn-integration`)

Goal: fold ESIMD + gdn_attn kernels into a single maintained fork, so
`import sgl_kernel` alone covers everything sglang XPU needs.

**Remote layout** at `/home/intel/xiangyu/sgl-kernel-xpu`:
```
origin   → github.com/analytics-zoo/sgl-kernel-xpu   (push / pull)
upstream → github.com/sgl-project/sgl-kernel-xpu     (read-only; periodic rebase)
HEAD     → intel/esimd-gdn-integration  (branched off upstream main c668bb6)
```

**What's vendored in** (files in `src/sycl/esimd/`, `src/sycl/gdn_attn/`):
- ESIMD core / lgrf / moe / gemm / topk_v2 kernels (from `custom-esimd-kernels-sglang`).
- Eagle speculative decode (from the same).
- MoE batch FP8 / INT4 (from the same).
- gdn_attn (conv1d + gated_delta_rule, with Xe2 chunked path) from `vllm-xpu-kernels`.

**Build convention**:
- Filenames ending in `*Xe20.(cpp|sycl)` get BMG AOT (`-device bmg`) automatically.
- Renamed: `chunk_gated_delta_rule_xe2.cpp` → `…_xe2Xe20.cpp`; `esimd_kernel_lgrf.sycl` → `…_lgrfXe20.sycl`; `esimd_kernel_topk_v2.sycl` → `…_topk_v2Xe20.sycl`.
- `torch_bindings_esimd.cc` + `torch_bindings_gdn_attn.cc` register every vendored op under `torch.ops.sgl_kernel.*` (duplicate `eagle_ops`/`moe_ops`/`moe_int4_ops` namespaces kept for drop-in parity).
- `VLLM_XPU_ENABLE_XE2` is defined project-wide so `gdn_attn_interface.cpp` takes the Xe2 chunked prefill path.
- `torch/extension.h` → `torch/all.h` + `torch/library.h` in vendored `.sycl` files (per upstream README SABI guidance).

**Xe3 WA** (ported from `vllm-xpu-kernels@368f685`):
- `src/sycl/Device.cpp::query_device()` — added `intel_gpu_ptl_h / ptl_u / wcl` cases returning `(3, 0)`.
- `python/sgl_kernel/utils.py` — added `is_xe3_arch()` and `is_xe2_or_xe3_arch()`.
- `python/sgl_kernel/moe.py` — the three `is_xe2_arch()` guards relaxed to `is_xe2_or_xe3_arch()`. Xe3 runs Xe2 cutlass kernels as a WA (mirrors upstream comment "Use XE2 cutlass kernel (also used as WA for XE3/XE3P)").

**Status**: uncommitted working tree; **not yet built**. We agreed not to commit until the build+test passes on hardware.

---

## 6. sglang repo patches (editable-installed into `txytest_sgl_0427`)

These changes live in this repository's working tree (same checkout you are
reading). They are bring-up scaffolding; most can be replaced once
`sgl_kernel.gdn_attention` lands.

1. **`python/sglang/srt/layers/attention/fla/chunk_torch_xpu.py`** (new) — pure-PyTorch GDN extend fallback. Used by the `is_xpu()` branch in `gdn_triton.py`.

2. **`python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py`** — added `is_xpu()` branch that:
   - binds `chunk_gated_delta_rule` to `chunk_gated_delta_rule_torch`
   - extends the "pre-slice state, return new state" contract so the XPU path uses the same flow as CPU/NPU

3. **`python/sglang/srt/layers/attention/linear/gdn_backend.py`** — `is_xpu` added to the imports and to the write-back branch of `forward_extend` (so the last recurrent state gets scattered back into the pool like on CPU/NPU).

4. **`python/sglang/srt/layers/attention/fla/layernorm_gated.py::_get_sm_count`** — XPU branch (see §4.3).

These are documented in detail in `docker_sgl_xpu/qwen35_xpu_notes.md`.

---

## 7. Open work items (prioritized)

### P0 — End-to-end Qwen3.5 perf on XPU

- [ ] Finish `vllm-xpu-kernels` in-container build (Task #9; ocloc AOT-linking `_moe_C.abi3.so`, ~80% done).
- [ ] Run `tests/gdn_attn/test_gdn_attn.py` to verify `torch.ops._xpu_C.gdn_attention` correctness on our hardware (Task #10).
- [ ] Build `sgl-kernel-xpu@intel/esimd-gdn-integration` in container.
- [ ] Smoke-test `torch.ops.sgl_kernel.gdn_attention` produced by our fork (schema parity with `_xpu_C` version).
- [ ] Replace `chunk_gated_delta_rule_torch` dispatch in `gdn_triton.py` with a direct call to `torch.ops.sgl_kernel.gdn_attention` (adjust argument packing: sglang passes `(q, k, v, g, β, ssm_states, cache_indices, query_start_loc)`, the kernel expects raw `(qkvz, ba, A_log, dt_bias, …)` → bypass sglang's `fused_gdn_gating` on the XPU path, or wrap the call).
- [ ] Re-run `run_qwen35.py`; measure prefill+decode latency, compare vs PyTorch fallback.

### P1 — Restore radix cache for Qwen3.5

Target: remove the `disable_radix_cache=True` workaround (§4.2).

- [ ] Unlock `extra_buffer` mode on XPU — relax the `assert is_cuda()` on `server_args.py:2343-2345` to also allow XPU (requires `fla/chunk_gated_delta_rule` to return per-chunk `h`; the native `gdn_attention` kernel already does).
- [ ] Verify `track_mamba_state_if_needed_kernel` Triton kernel (`hybrid_linear_attn_backend.py:37-94`) compiles on XPU. It's a plain gather/scatter, but worth testing; if it fails, write an eager `index_copy_` fallback.
- [ ] Exercise `_track_mamba_state_extend` with radix enabled — verify state checkpoints are scattered back correctly at FLA_CHUNK_SIZE boundaries.
- [ ] Test: two sequential prompts sharing a long prefix; ensure second prompt shows shorter prefill wall time (proof of prefix hit).

### P2 — Structural cleanup

- [ ] Decide: keep `custom-esimd-kernels-sglang` as a separate package, or fully fold into the `sgl-kernel-xpu` fork and retire it.
- [ ] Add `tests/` entries under `sgl-kernel-xpu@intel/esimd-gdn-integration` for the vendored GDN attention (port `test_gdn_attn.py`) and top-level ESIMD ops.
- [ ] Expose `VLLM_XPU_ENABLE_XE3` (or equivalent) as a proper CMake option instead of assuming WA routing — blocked on upstream roadmap.
- [ ] Lift `AOT_DEVICES` list to include `ptl-h,ptl-u,wcl` when actually targeting those chips, rather than relying on SPIR-V JIT.

### P3 — Documentation / ergonomics

- [ ] Finalize this doc (and keep `qwen35_xpu_notes.md`/`qwen35_pipeline.md` in sync as we land fixes).
- [ ] Add a "run-me.sh"-style script under `/home/intel/xiangyu/sgl_offline/` that reproduces the minimum Qwen3.5 smoke test.
- [ ] CI recipe once the fork builds green.

---

## 8. Reproduction commands

```bash
# Build image (host)
cd /home/intel/xiangyu/docker_sgl_xpu && SUDO=0 bash build.sh

# Install local sglang (editable) into running container
docker exec txytest_sgl_0427 bash -lc '
  git config --global --add safe.directory /llm/workspace/sglang
  cd /llm/workspace/sglang/python
  cp -f pyproject_xpu.toml pyproject.toml
  pip install --no-deps -e .
'

# Qwen3.5-0.8B offline smoke test
docker exec txytest_sgl_0427 bash -lc \
  'cd /llm/workspace/sgl_offline && python3 run_qwen35.py'

# Sanity: Qwen3-0.6B (no GDN; exercises XPU attention path only)
docker exec txytest_sgl_0427 bash -lc \
  'cd /llm/workspace/sgl_offline && python3 run_qwen3_06b.py'
```

Minimum Engine config that currently works on Qwen3.5 (see workarounds above):
```python
sgl.Engine(
    model_path="/llm/models/Qwen3.5-0.8B",
    trust_remote_code=True,
    disable_overlap_schedule=True,   # forced by no_buffer strategy
    disable_radix_cache=True,        # forced by page_size 3-way conflict
    device="xpu",
    attention_backend="intel_xpu",
    page_size=64,
    dtype="bfloat16",
)
```

---

## 9. INT4 / AWQ / GPTQ / GGUF support gaps for Qwen3.5 on XPU

Target models on disk:

| Model                                      | Architecture                            | `quant_method` | Notes |
|--------------------------------------------|------------------------------------------|----------------|-------|
| `Qwen3.5-27B-GPTQ-Int4`                    | `Qwen3_5ForConditionalGeneration`        | `gptq` (w4g128, sym, desc_act=false) | Dense + hybrid GDN; `dynamic` excludes attn/shared_expert/mtp/visual |
| `Qwen3.5-35B-A3B-GPTQ-Int4`                | `Qwen3_5MoeForConditionalGeneration`     | `gptq` (w4g128, sym, desc_act=false) | MoE + hybrid GDN; same `dynamic` exclusion map |

### 9.1 Quantization registry landscape

sglang (`python/sglang/srt/layers/quantization/__init__.py:72-108`) registers
these method keys:

```
awq, awq_marlin, gguf, gptq, gptq_marlin, compressed-tensors, fp8, fbgemm_fp8,
modelopt, modelopt_fp4, modelopt_mixed, bitsandbytes, qoq, w8a8_int8,
w8a8_fp8, blockwise_int8, petit, moe_wna16, quark, auto_round, mxfp4, ...
```

Each has its own kernel backing. For **XPU** the story differs per method:

| Method       | Dequant / GEMM kernel       | XPU-ready? | Source of truth |
|--------------|-----------------------------|-----------|------------------|
| `awq`        | `awq_dequantize` + `torch.matmul` | ✅ partial | `awq.py:82-86` → `from sgl_kernel import awq_dequantize` (exposed by our XPU sgl-kernel-xpu; see `include/sgl_kernel_ops.h:181`) |
| `awq_marlin` | `apply_awq_marlin_linear` | ❌ | `awq.py:346-347`: `if not _is_cuda: return False` in `is_awq_marlin_compatible`. **Marlin path is CUDA-only.** |
| `gptq`       | `gptq_gemm` + `gptq_shuffle` | ❌ | `gptq.py:67-69`: **`from sgl_kernel import gptq_gemm, gptq_shuffle` gated by `if _is_cuda`**. On XPU the imports are simply skipped → running `gptq` on XPU raises `NameError: gptq_gemm is not defined` at first layer apply. |
| `gptq_marlin`| `gptq_marlin_gemm`         | ❌ | `marlin_utils.py:43-46`: `if _is_cuda: from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm`. `GPTQMarlinConfig.override_quantization_method` happily picks itself even when not CUDA if config is marlin-compatible, but `apply_gptq_marlin_linear` then crashes on `ops = None` / `gptq_marlin_gemm not defined`. |
| `gguf`       | `ggml_dequantize` etc.     | ❌ | `gguf.py:36-62`: only `_is_cuda`, `_is_musa`, `_is_npu` branches import real kernels; others hit `warnings.warn("Only CUDA, MUSA and NPU support GGUF quantization currently.")`. No XPU path. |
| `auto_round` | routes to awq/gptq         | inherits | Good pass-through if the underlying method works on XPU. For `auto_round:auto_gptq` it ends up at `GPTQLinearMethod` → broken (see above). |
| `compressed-tensors` (wNa16)| marlin           | ❌ | `compressed_tensors/schemes/compressed_tensors_wNa16.py:46` uses `jit_kernel.gptq_marlin_repack` under `_is_cuda`; no XPU branch. |
| `moe_wna16`  | w4/w8-NA16 MoE             | ❌ | Triton MoE wNA16 kernel — same Triton-XPU risk class as GDN chunk_delta_h.py; untested. |
| `awq_cpu`, `gptq_cpu` | `int4_scaled_mm_cpu`| n/a (CPU only) | IPEX AMX path. Not XPU. |

**Net**: today on XPU only the "plain" AWQ path (dequant → dense matmul) is usable. Every other INT4 family either fails to import or hits an undefined symbol at first call.

### 9.2 Why `gptq` currently fails for the Qwen3.5 GPTQ-Int4 checkpoints

`config.json` in both GPTQ-Int4 variants declares:

```json
"quantization_config": {
  "bits": 4, "group_size": 128, "sym": true, "desc_act": false,
  "quant_method": "gptq",
  "dynamic": {
    "lm_head": {}, "model.language_model.embed_tokens": {},
    "-:.*attn.*": {},               # exclude all attention (full + linear)
    "-:.*shared_expert.*": {},       # exclude shared experts (MoE)
    "-:.*mtp.*": {},                 # exclude MTP head
    "-:.*visual.*": {}               # exclude vision encoder
  }
}
```

The `dynamic` map is interpreted by `utils.py:275-293` (regex-prefix match,
`-:` = "don't quantize"). sglang honours this via `get_linear_quant_method`
→ `UnquantizedLinearMethod()` for excluded modules. So in principle **only
the MLP (dense) and the routed MoE experts are GPTQ-quantized**; attention
matrices, shared experts, MTP, visual encoder stay FP16/BF16.

Two failure modes on XPU today:

1. **Linear GPTQ apply**: `GPTQLinearMethod.apply` (`gptq.py:588-608`) calls `gptq_gemm(...)`. On XPU `gptq_gemm` was never imported (`_is_cuda` gate at `gptq.py:66-68`). First forward raises `NameError`.
2. **`override_quantization_method` auto-upgrades to gptq_marlin if allowed**. `GPTQMarlinConfig.override_quantization_method` returns `"gptq_marlin"` even on XPU (no CUDA guard in the override itself). The user won't see a warning — sglang just silently picks marlin, then crashes in `marlin_utils.py` because `gptq_marlin_gemm` is CUDA-only. Workaround today: explicitly pass `quantization="gptq"` (not `gptq_marlin`, and not left auto). Still broken at apply-time due to #1.

### 9.3 Why `awq` half-works for Qwen models on XPU

`sgl-kernel-xpu/src/sycl/awq_dequantize.cpp` provides
`torch.ops.sgl_kernel.awq_dequantize` (weight-only INT4 → fp16/bf16).
`awq.py::AWQLinearMethod.apply` does `awq_dequantize(qweight, scales, qzeros)`
then `torch.matmul(x, dequantized_weight)`. That **works end-to-end on XPU**
for AWQ-quantized models (e.g. a hypothetical `Qwen3.5-*-AWQ-Int4`), but:

- Performance is **un-fused** — dequant + matmul are two separate XPU ops, with full-FP16 weight materialised between them.
- **`awq_marlin` auto-upgrade path is blocked** (`is_awq_marlin_compatible` returns False when not CUDA).
- **MoE AWQ → MoeWNA16 route** is untested on XPU; the Triton kernels used there have the same risk profile as GDN extend.

An optimized XPU AWQ path is **already present in our workspace but not wired
into sglang**: `custom-esimd-kernels-sglang/csrc/xpu/esimd_kernels/int4_GEMV.h`
registers `esimd_gemv_int4` / `esimd_gemv_int4_fused2` under
`torch.ops.custom_esimd_kernels_sglang.*` (see
`custom-esimd-kernels-sglang/KERNELS.md §2`). It accepts the same INT4 layout
`[N, K/2]` uint8 packed + per-group fp16 scale (group_size=128) that AWQ uses.
Integrating it would require:
- a `sgl_kernel`-namespace binding (planned: via our fork `sgl-kernel-xpu@intel/esimd-gdn-integration`, §5)
- a `AWQLinearMethod.apply` XPU branch that calls the fused int4 GEMV instead of `awq_dequantize + matmul`
- layout adapter from AWQ's packed order (pairs-of-4) to the ESIMD kernel's expected packing — **nibble order differs** between AWQ (`AWQ_REVERSE_ORDER = [0,4,1,5,2,6,3,7]`) and IPEX/GGML layouts that `esimd_gemv_int4` supports today.

### 9.4 GGUF — no XPU path at all

`gguf.py:36-62` silences to `warnings.warn("Only CUDA, MUSA and NPU support
GGUF quantization currently.")` and does not register any kernel on XPU.
The CPU side of the quant family (`awq_cpu.py`, `gptq_cpu.py`) both resolve to
`torch.ops.sgl_kernel.int4_scaled_mm_cpu` (IPEX AMX) — no equivalent XPU op.

For Qwen3.5 GGUF variants (not on disk, but a likely ask): nothing works.
Options:

1. **Port `awq_cpu.py`/`gptq_cpu.py` pattern to XPU** by introducing
   `torch.ops.sgl_kernel.int4_scaled_mm_xpu` backed by our vendored ESIMD
   int4 GEMV. Cheapest path if the shape matrix is compatible.
2. **Implement GGUF K-quants on XPU** (Q2_K, Q3_K_*, Q4_K_*, Q5_K_*, Q6_K,
   Q8_K, IQ*). A full matrix; llama.cpp SYCL backend has kernels we can draw
   from (`llama.cpp/ggml/src/ggml-sycl/dequantize.cpp`). Big effort.

### 9.5 Additional Qwen3.5-specific landmines for INT4

Beyond the cross-cutting XPU issues, the Qwen3.5 model code has a few gates
that interact with quantization:

1. **`Qwen3_5GatedDeltaNet.__init__` (qwen3_5.py:539-545)**: forces `quant_config=None` for the linear-attn layers when `quant_config.get_name() == "modelopt_fp4"`. **Does NOT special-case `gptq` or `awq`** — but the `dynamic` map in the GPTQ configs above already excludes `.*attn.*`, so in practice all linear-attn projections become FP16 via `UnquantizedLinearMethod`. That's correct behavior, but it relies on regex matching working, which it does for prefixes like `model.language_model.layers.5.linear_attn.in_proj_qkvz` — ✅ matches `-:.*attn.*`.

2. **`in_proj_qkvz` / `in_proj_ba` packed weight loaders** (`qwen3_5.py:184-186`, `_bind_packed_weight_loaders`) also bind loaders for `weight_scale_inv` / `weight_scale` / `input_scale`. If the GPTQ-Int4 checkpoint routes through `UnquantizedLinearMethod` these scale params never materialize → fine. But if a future variant enables GPTQ on these layers, the `_make_packed_weight_loader` path in `qwen3_5.py:308-357` already supports `BlockQuantScaleParameter` and `PerTensorScaleParameter` — **but not the GPTQ-specific `g_idx`/`qweight`/`qzeros` tensors**. This would need a separate splitter.

3. **MoE GPTQ for Qwen3.5-35B-A3B-GPTQ-Int4**:
   - `GPTQConfig.get_quant_method` at `gptq.py:256-258`: `if isinstance(layer, FusedMoE): raise TypeError("GPTQ Method does not support MoE, please use gptq_marlin")`.
   - `gptq_marlin` is CUDA-only (see §9.1).
   - Only workable today: `moe_wna16`, which itself uses Triton kernels that may or may not compile on Intel Triton — untested.

4. **`process_weights_after_loading` expects CUDA-aligned dtypes**: `gptq.py:570-586` (marked as `exllama shuffle`) requires `gptq_shuffle` — again CUDA-only.

### 9.6 What's needed to actually run Qwen3.5-GPTQ-Int4 on XPU

Minimum viable path (linear GPTQ only, dense model like Qwen3.5-27B-GPTQ-Int4):

- [ ] Add a `gptq_gemm` / `gptq_shuffle` implementation to `sgl-kernel-xpu` (SYCL).  Needed by `gptq.py:588-608::GPTQLinearMethod.apply`. Options: wrap `awq_dequantize`-style INT4 dequant then call `torch.matmul`; or port sgl-kernel's CUDA gptq kernel from AutoGPTQ / exllama2 directly. Simpler & slower first.
- [ ] Expose through the Python `sgl_kernel` surface (add to `python/sgl_kernel/gemm.py`).
- [ ] Guard `gptq.py`: change the `_is_cuda` gate at line 66 to `_is_cuda or _is_xpu` once the imports resolve, or add a parallel `if _is_xpu:` branch.
- [ ] Ensure `GPTQMarlinConfig.override_quantization_method` **does not** silently upgrade to `gptq_marlin` on XPU — add `if not _is_cuda: return None` guard similar to `AWQMarlinConfig`.
- [ ] Confirm the `dynamic` regex exclusion leaves attention / GDN / MoE-shared / MTP / visual on FP16 (end-to-end smoke test).

Optional, for MoE variant Qwen3.5-35B-A3B-GPTQ-Int4:

- [ ] Implement `GPTQMoEMethod` (today only `GPTQMoEAscendMethod` exists) or fall back to `moe_wna16` + verify its Triton kernels compile on XPU. If they don't, same fallback pattern as GDN: PyTorch eager or ESIMD.
- [ ] The routed experts use `w13_qweight` / `w2_qweight` packed int32 layouts — match ESIMD `esimd_moe_gemm_fp8` / `moe_forward_full_int4` layout conventions (see `custom-esimd-kernels-sglang/KERNELS.md §9`).

For AWQ checkpoints (optimization, not correctness):

- [ ] Bind `esimd_gemv_int4` / `esimd_gemv_int4_fused2` into `torch.ops.sgl_kernel.*` via the fork (§5).
- [ ] Add an `awq.py::AWQLinearMethod` XPU branch that uses the fused GEMV instead of dequant+matmul, with nibble-order adapter.

For GGUF (greenfield):

- [ ] Add `int4_scaled_mm_xpu` to `sgl-kernel-xpu`, route `awq.py`/`gptq.py` through it analogously to the CPU paths.
- [ ] (Longer term) Port llama.cpp SYCL K-quant kernels.

---

## 10. Summary matrix

| # | Area                                 | Status   | Workaround                              | Real fix                                                        |
|---|--------------------------------------|----------|-----------------------------------------|-----------------------------------------------------------------|
| 1 | GDN extend Triton compile            | fixed(WA)| PyTorch fallback `chunk_torch_xpu.py`   | Wire `torch.ops.sgl_kernel.gdn_attention` (fork build in progress) |
| 2 | Radix × Mamba × XPU page_size        | mitigated| `disable_radix_cache=True`              | Unlock `extra_buffer` on XPU; requires per-chunk `h` output    |
| 3 | `_get_sm_count` CUDA hardcode        | fixed    | XPU branch using `max_compute_units`    | Replace with `torch.accelerator.current_device_properties()`   |
| 4 | Fallback performance                 | open     | Short prompts only                      | Native SYCL kernel (#1)                                         |
| 5 | Xe3 hardware support (Panther/WC)    | WA-ready | `is_xe3_arch()` → Xe2 kernels route     | Native Xe3 cutlass kernels (upstream, no ETA)                   |
| 6 | GPTQ-Int4 on XPU (Qwen3.5-27B / 35B) | broken   | Use FP16/BF16 checkpoint                | Add `gptq_gemm`/`gptq_shuffle` SYCL impl; guard marlin auto-upgrade on XPU |
| 7 | GPTQ-Marlin on XPU                   | broken   | Force `quantization="gptq"` (also broken per #6) | Port `gptq_marlin_gemm` to SYCL, or keep disabled on XPU |
| 8 | AWQ-Int4 on XPU                      | works-slow| `awq_dequantize` + `torch.matmul`       | Bind ESIMD `esimd_gemv_int4[_fused2]` in `sgl_kernel` and use fused apply |
| 9 | AWQ-Marlin on XPU                    | blocked  | Falls back to plain AWQ (slow)          | Port `awq_marlin_gemm` to SYCL, or keep disabled on XPU         |
| 10| GGUF on XPU                          | absent   | Dequantize offline to FP16              | `int4_scaled_mm_xpu` op + GGUF K-quant SYCL port                |
| 11| Qwen3.5-35B-A3B-GPTQ-Int4 MoE path   | broken   | FP16/BF16 checkpoint only               | `GPTQMoEMethod` for XPU, or compile `moe_wna16` Triton on XPU   |

This document supersedes earlier scattered notes and should be updated as
items move through the workflow.
