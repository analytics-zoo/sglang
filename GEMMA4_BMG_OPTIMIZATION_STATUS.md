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
"graph slower" here is purely a symptom of the xccl-captured-allreduce stale-replay bug
(见 Known Issue #2;⚠️ 2026-07-03 修正:captured allreduce **可行**,vLLM-xpu 即把它捕进图——
本栈 stale 特属 xccl;首选修复 = 图安全 custom SYCL allreduce,piecewise-collectives-eager 仅为保底 workaround)。

## FP8 Decode GEMV — Kernel 定位、VL 调优、刷新性能矩阵 (2026-07-02)

详见 `GEMMA4_DECODE_PERF_ROOFLINE.md`（同日已更正）。要点：

### 正确的 decode FP8 GEMV 内核
- 包：`/workspace/custom-esimd-kernels/`（`python_v2`，**非** `custom-esimd-kernels-sglang` 的
  `fp8_GEMV_v2.h`——那是 decode 死代码）。op `torch.ops.custom_esimd_kernels.esimd_gemm_fp8_pert`。
- VL 由 `csrc/xpu/esimd_kernels/fp8_GEMM_pert.h::select_vl_ks`（~L75）→ `GEMV_fp8_pert_batched_kernel<VL,KS>`。
- gemma4 decode 的 5 个 GEMV 归组（按 N_out）实为 **6 个 (N,K) shape**：gate_up (21504,5376) VL256×60；
  qkv sliding (8192,5376)/global (10240,5376) VL256；**o_proj sliding (5376,4096)×50、o_proj full-attn
  (5376,8192)×10、down_proj (5376,10752)×60 → 全 VL512**（trace 只按 N_out=5376 归成一组，误作 "同形状"）。

### VL512 → 256 A/B：kernel 微测提速未传导到 E2E → **保留 512**
> ⚠️ **本节结论已被 2026-07-03 的复测推翻，见下方「GEMV tile 复调优」。** 此处的 +2-3% "回退"
> 系 **rebuild-confound + 节点争用噪声**（旧 A/B 两侧是不同 binary、非同时刻），并非真实回退。
> 用 **同一 binary + env-gate** 复测后 VL256 实为中性偏优（−0.06ms）。保留原文如下备查。

把上面 3 个 `<512>` shape 改 VL256 重建后同 server A/B：

| ctx  | VL512 (known-good) | VL256 | delta |
|------|--------------------|-------|-------|
| 1024 | **39.03**          | 40.19 | +3.0% |
| 8192 | **39.51**          | 40.39 | +2.2% |

VL256 让 E2E decode **回退 ~2-3%（~1ms TPOT）**，正确性不变（gsm8k 0.975）。孤立 GEMV 微测的 "+6-9%"
在 bsz=1 E2E 未兑现（这 3 个 shape 本就在 90% 带宽墙）。**结论：不改 256。** 容器已还原 pristine
known-good（gemm `.so` md5 `42e28795`，sglang `.so` `3806e20f`）。

### 刷新的 clean-bench 性能矩阵（known-good VL512，eager，1k–64k × out256）

| input | TTFT (ms) | TPOT (ms) | tok/s | E2E (s) |
|-------|-----------|-----------|-------|---------|
| 1024  | 621       | **39.23** | 25.5 | 10.63 |
| 2048  | 1276      | **39.25** | 25.5 | 11.28 |
| 4096  | 2644      | **39.22** | 25.5 | 12.64 |
| 8192  | 5721      | **39.27** | 25.5 | 15.73 |
| 16384 | 12805     | **40.99** | 24.4 | 23.26 |
| 32768 | 31100     | **44.16** | 22.6 | 42.36 |
| 65536 | 84143     | **50.45** | 19.8 | 97.01 |

- TPOT 1k–8k **平坦 ~39.2ms**（spread 0.05ms，极稳），16k 起因 10 层 global split-K 扫全 KV 上升。
- out=64 对照：40.1/40.5/40.2 @ 1k/4k/8k（少 token → 首 token 暖机权重大，比 out256 高 ~1ms）。
- **更正**：clean TPOT 真实 ~39–40ms **可复现**；`ROOFLINE.md` 早前 "42–43ms / 38.9 不可复现" 系
  unitrace wall 抬高值（已在该文更正）。真实 device-busy ~91%（非 81%），XPU graph 可回收 host-gap ~3ms（非 8.6ms）。
- lm_head 未分片 = 2.82GB/rank，已在内存墙 96%；ESIMD vs oneDNN 仅差 0.13ms（原 "ESIMD 省 3ms" 系归因错误）。

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

## Known Issue #2: XPU GRAPH DECODE GARBLE (TP>1) — captured **xccl** allreduce replays STALE on this stack (⚠️ 2026-07-03 修正：并非"captured allreduce 本质不可行")

**Symptom:** with `SGLANG_XPU_ENABLE_GRAPH=1` at TP=2, decode output is garbled: token 0
is correct (logprob −0.00) then token 1+ degenerate ("Paris ( ( (", "46 a a a"). gsm8k ≈ 0.
Eager (graph off) with the SAME kernels is correct (gsm8k 0.950). Localized 2026-07-01.

**⚠️ 2026-07-03 结论修正**：此前写"captured allreduce 必然 stale、唯一正确路线 = 让 collectives 保持 eager"是**错误**的。
vLLM-xpu **确实把 per-step allreduce 捕进 XPU graph 内**（见下"vLLM-xpu 实证"）——所以"捕获 allreduce"本身**可行**。
真实问题被收窄为：**本容器的 torch+xccl（oneCCL）栈里，把 `dist.all_reduce` 直接捕进 `torch.xpu.XPUGraph` 后，2nd+ replay 返回 stale/lagged 结果**。

**现象（本栈可复现）：** `SGLANG_XPU_ENABLE_GRAPH=1` TP=2 decode 乱码：token 0 正确（logprob −0.00），
token 1+ 退化（"Paris ( ( ("、"46 a a a"），gsm8k≈0。eager（同 kernels）正确。gemma4 decode 每 step 捕获
~120 allreduce（60 层 × o_proj+down_proj 各一次）；token 0 = 首次 replay（对），token 1+ = 后续 replay（stale）→ 每层 TP 输出损坏。

**隔离复现（2026-07-03 重跑 `cc_workspace/gemma_splitk/test_ar_*.py`，GPU0,1，xccl）：**
- `dist.all_reduce` 捕进 XPUGraph，逐 step in-place 换输入后 replay：**首次 replay 正确、2nd+ replay stale/lagged**。
  `test_allreduce_graph`: REPLAY=30(对) → REPLAY2=30(应 300)。`test_ar_faithful`: replay0=3(对), replay1=3(应6,stale), replay2=6(应9)…滞后一拍并冻结。
- 四种捕获方式均 stale：side-stream、default-stream（`capture_begin/end`）、每 replay 加 `dist.barrier()+synchronize()`、**以及 in-graph producer**（`test_ar_producer`：图内 `y=x*2` 再 `all_reduce(y)`，仍 step0 对、step1+ stale）→ 排除"缺少图内生产者"假设。
- 对照（captures fine）：纯 torch 计算 op replay 3× 全新（对）；split-K ESIMD kernel in-place 换 seqlen/q replay cos=1.0（对）。**故 stale 特属 xccl 集合通信，非通用图捕获缺陷。**

**已排除的其它可疑点：** fresh `torch.empty` attention out/scratch（改持久 `_decode_buf`）；`swa_page_table`/`swa_out_cache_loc`（改持久 in-place）；split-K 两阶段乱序（加 `h.depends_on(evA)`）——均只扰动、非根因。

**vLLM-xpu 实证（`vllm/platforms/xpu.py` NOTE gemma4-xpu-graph + `xpu_model_runner.py:53`）：**
- vLLM-xpu 把 `torch.cuda.CUDAGraph → torch.xpu.XPUGraph`（与 sglang 同一 device graph），FULL_DECODE_ONLY 下
  **allreduce 位于捕获区内、随 decode step 一起 replay**（allreduce **不是** split point；仅 attention(sycl-tla FMHA 不可捕) 与 ESIMD-MoE 是 FX split point）。注释称"validated on Qwen3-Coder-Next TP=4"。
- **两点关键澄清（决定能否照搬）：**
  1. 该 "validated" 引用的是 **perf POC（`xpu_graph_vs_eager_perf`）**，是"能跑 + 更快"，**未见 gsm8k 正确性门**——即 vLLM 很可能带有**同样的 stale 隐患但只测了性能**。
  2. vLLM 的图安全 **custom/quick allreduce**（`ca_comm`/`qr_comm`：`ops.qr_all_reduce`、`symm_mem.two_shot_all_reduce_`）
     **只在 `cuda_communicator.py` 装配，`xpu_communicator.py` 没有** → XPU 路径 `all_reduce` 落到 `dist.all_reduce`（xccl），
     与 sglang 相同。故 vLLM-xpu 在**本栈**大概率同样 stale（除非其验证栈的 oneCCL/driver 版本行为不同）。

**修正后的正确修复方向（按稳健度排序）：**
- **Option B（首选，vLLM 已有代码可借鉴）**：换用**图安全的 custom SYCL allreduce**（plain kernel、无 oneCCL 内部 queue）——
  即 sgl-kernel-xpu 的 `python/sgl_kernel/allreduce.py`（当前未在此 XPU 构建），或移植 vLLM 的 `qr_all_reduce`/`symm_mem two-shot`。
  这类 kernel 像 split-K 一样可被图捕获（隔离已证 ESIMD kernel replay cos=1.0）。**这是让"captured allreduce"真正正确的路径。**
- **Option A（保底 workaround）**：piecewise capture、把 `tensor_model_parallel_all_reduce` 留 eager（段间跑集合通信）。
  能绕开 stale，但保留 120 次/step 集合 launch，收益受限。
- **必做验证**：无论哪条路，落地后必须过 gsm8k 门（≥0.975），不能只看 perf（vLLM 的教训）。
- **待厘清**：本栈 xccl-captured-allreduce 的 stale 是 oneCCL 版本/配置问题还是根本限制——可对照 vLLM 验证栈的 oneCCL/torch/driver 版本，或直接在本容器起 vLLM-xpu graph 做 gsm8k 正确性复核。

### XPU graph decode gap 量化（unitrace 实测，2026-07-03）

> 目的：回答"decode 打开 XPU graph 到底能省多少 TPOT"。方法：eager server 挂 unitrace，
> bsz=1 打 30-token decode，取主 decode burst（31 步）解析 `cat=gpu_op` 逐核 device 时间。
> ⚠️ unitrace 会同时**放大** device 核时延与 host-gap，绝对值仅供比例分析（trace TPOT 68ms/step vs 非-trace 37.7ms）。

**主 decode burst：31 步，wall/step=68.1ms（traced），device-busy%（union）=79.7%。**

per-step device 时间构成（traced，按占比）：

| kernel 类 | 数量/step | us/step | %dev | 可否被 graph 消除 |
|-----------|----------|---------|------|-------------------|
| GEMV_fp8`<256/512>` | 240 | 24537 | 45.2% | ❌ 带宽墙（DRAM-bound，~92% roofline） |
| allreduce (xccl) | 120 | 11769 | 21.7% | ❌ 集合通信;本栈 captured-replay stale(Known Issue #2,非本质不可捕) |
| oneDNN `gemm_kernel` | 7 | 6079 | 11.2% | ❌ 计算/带宽 |
| lm_head `GEMV_fp16<256>` | 1 | 4657 | 8.6% | ❌ 带宽墙（2.82GB 权重） |
| norms（融合） | 211 | 2770 | 5.1% | ⚠️ 仅省 launch |
| paged_attn 3-phase | 145 | 2024 | 3.7% | ❌ 计算 |
| gelu_mul / splitK / elementwise / KVScatter / rope / index | ~500 | ~1600 | ~3% | ⚠️ 仅省 launch |

**结论：graph 的收益上限很小。**
- **不可回收部分 ≈ 90% device 时间**：GEMV_fp8 45% + allreduce 22% + oneDNN GEMM 11% + lm_head 8.6% + paged_attn 3.7% 全是真实带宽/计算/通信，graph 不改变它们。
- **graph 只能回收 host-launch 空隙**（device-idle）：~1400 核/step，trace 下 device-idle ≈ 20%（放大态），生产态更小（旧测 decode ~91% busy）→ **实际可回收 ≈ 单位数 %（几 ms/step）**，与本文档"对 TPOT 的现实预期：量级几个%"一致。
- **额外封顶**:本栈 xccl captured-allreduce replay stale(见 Known Issue #2)。修复有二——图安全 custom SYCL allreduce(首选,可把 120 allreduce 也捕进图)或 piecewise-collectives-eager(保底,保留 120 次/step 集合 launch)。无论哪种,可回收的都只是 host-launch 空隙,不动 GEMV/allreduce 本体带宽/通信。
- **ROI 判断**：decode 已带宽/通信受限（GEMV+allreduce+GEMM+lm_head = 87% device 时间），XPU graph 即便修好正确性，TPOT 收益预期仅单位数 %。**优先级应低于**减少 DRAM 流量类优化（权重/KV 复用、量化、算子融合削 traffic）。真正修 graph 的价值更多在**去除 CPU 争用敏感场景下的 host 抖动**，而非稳态 TPOT。

## PREFILL 阶段 Kernel 分发分析（源码追踪，2026-07-02）

> 本文档其余部分只覆盖 **decode**。本节补 **prefill（M>1）** 的逐 kernel 分发。分发链**全部来自源码追踪**（file:line 可查），**相对占比已用 `utrace_4k` trace 实测验证**（2026-07-02，见本节末 "### 实测验证"）。默认配置 = 可发布的 fp16+ESIMD+intel_xpu+layered_fp8（bf16 差异单列）。

模型复核（config.json）：dense、无 MoE、无 PLE（`hidden_size_per_layer_input=0`）；60 层 = 50 sliding（hd256/16kv）+ 10 global（hd512/4kv，`global_head_dim=512`）。

### Prefill 单层 kernel 全景

| 步骤 | 走的 kernel | 后端 | 调优 | 源码 |
|---|---|---|---|---|
| input_layernorm（无 residual） | `rmsnorm` | **SYCL** | ✅ | `layernorm.py:542` |
| qkv/o/gate_up/down **FP8 激活量化** | `_per_token_group_quant_8bit` | **Triton** | ❌ 有 SYCL 版未接 | `fp8_kernel.py:704`,`:343` |
| qkv/o/gate_up/down **FP8 GEMM** | `torch._scaled_mm` + 反量化 elementwise | **oneDNN+torch** | ⚠️ vendor fallback，非 ESIMD | `fp8_utils.py:1575`,`:1586` |
| QKV norm+RoPE（sliding, fp16） | `esimd_qkv_split_norm_rope` | ESIMD | ⚠️ prefill 无 M gate，存疑 | `gemma4_causal.py:449` |
| QKV norm（sliding, **bf16**） | `gemma_qkv_rmsnorm` | **Triton** | ❌ | `gemma4_causal.py:510`；kernel `gemma4_fused_ops.py:244` |
| q/k/v norm（global hd512） | `Gemma4RMSNorm`→`rmsnorm` ×3 | **SYCL 但未融合** | ⚠️ 3 次独立 launch | `layernorm.py:916`；`gemma4_causal.py:543-554` |
| RoPE（global，及 bf16 sliding） | `fused_qk_rope_with_cos_sin_cache_inplace` | **SYCL** | ✅ | `rotary_embedding/base.py:442` |
| attention（prefill） | `flash_attn_varlen_func` | **SYCL FMHA** | ✅ | `xpu_backend.py:781/795` |
| post_attn + pre_ff（含 residual） | `fused_add_rmsnorm` | **SYCL** | ✅ | `layernorm.py:540` |
| MLP 激活 | `gelu_tanh_and_mul` | **SYCL** | ✅ | `activation.py:169` |
| post_ff norm+residual+scalar | `gemma_rmsnorm_residual_scalar` | **Triton** | ❌ | `gemma4_causal.py:839`；kernel `gemma4_fused_ops.py:54` |

### 仍然是 Triton 的 kernel（3 个）

1. **FP8 激活量化 `_per_token_group_quant_8bit`** — prefill 频率最高（每层 4 次 × 60 = 240/step）。
   分发链：`fp8_utils.py:1633`（定制 ESIMD GEMM 只在 `M<=64` 生效，prefill M>64 被排除）→ `:1715` 调 `per_token_group_quant_fp8` → `fp8_kernel.py:701-704`（**非 cuda 一律绑定 Triton `_per_token_group_quant_8bit_raw`**）→ Triton launch `:343`。
   ⭐ sgl_kernel **有** SYCL 版 `sgl_per_token_group_quant_8bit`（`fp8_kernel.py:87`），但顶层 `per_token_group_quant_fp8` 只在 `_is_cuda` 分支用 SYCL-capable 版，XPU 落到 Triton → **"有现成 SYCL 但没接上"**。
2. **`gemma_rmsnorm_residual_scalar`**（post_ff：norm+残差加+layer_scalar，融合）— 每层 1 次。prefill 因 `shape[0]!=1` 走 Triton（`gemma4_causal.py:839`）；ESIMD 版 gate 死 `shape[0]==1`（decode-only，`:822`）。
3. **`gemma_qkv_rmsnorm`**（sliding QKV 融合 norm）— **仅 bf16 配置**；fp16 走 ESIMD。`gemma4_fused_ops.py` 全文纯 Triton，无 SYCL/ESIMD 替身。

### 非 Triton、但走 vendor / 未融合（非本套定制 ESIMD）

1. **prefill FP8 GEMM = `torch._scaled_mm`（oneDNN），不是 ESIMD**。定制 ESIMD GEMM（`esimd_gemm_fp8_pert`）被显式 gate 在 `M<=64`（`fp8_utils.py:1626-1659`，注释：其 M≥64 weight-stationary 路径 "14000us vs 156us @M=4096" 太慢，故 prefill 让给 `_scaled_mm`）。→ prefill 主算力走 oneDNN，不是本文其余部分调过的那套 decode ESIMD。
   > 口径提醒：早期笔记 profile 里的 `FP8_GEMM_DPAS_V9` 极可能就是 `torch._scaled_mm` 的 oneDNN kernel，而非 ESIMD——按当前 gate，prefill 根本不进 ESIMD。旧笔记把它标成 ESIMD 可能是误判，待一次 prefill profile 对齐。
2. **反量化是未融合的 elementwise**（`_apply_fallback_scaled_mm`，`fp8_utils.py:1575-1589`）：gemma4 权重 per-tensor、激活 per-token（`per_token_group_quant_fp8` group=全行 → 每 token 一个 scale）→ 命中 fallback：fp8 matmul 输出 **fp32**，再 `output * x_scale * weight_scale` 两次 elementwise + cast。相比 per-tensor 融合进 `_scaled_mm` epilogue（`:1793`），每个 FP8 linear 多一份 fp32 输出读写 + 两趟 elementwise。
3. **global 层（hd512）q/k/v norm 未融合**：`head_dim==256` gate 全不满足（`gemma4_causal.py:451`/`504`）→ 10 个 global 层每层 3 次独立 `Gemma4RMSNorm→rmsnorm`（SYCL 但 3 次 launch），无 sliding 的融合 kernel。

### 实测验证（`utrace_4k` trace，TP0 python3.38621.json，2026-07-02）

用之前 decode 复盘用的同一条 4k trace（4096 input，`--chunked-prefill-size 1024` → 4 chunks×1024），
隔离出**单次测量 prefill**（trace 内含 3 次 prefill run = 2 warmup + 1 measured；取最后一次干净窗口
`[98.618..101.263]s`，其后才出现 decode 的 `gemv_fp8`）。交叉校验：该窗口 device self-time **2597 ms ≈
实测 TTFT@4k 2644 ms** → **prefill ~98% device-busy（compute-bound，几乎无 host-gap）**，与 decode 的 ~91% 形成对比。

**逐 family 占比（单次 4096-token prefill，2597 ms；per-chunk = ÷4）：**

| family | kernel | self-time | %prefill | 次数(/chunk) | 结论 |
|---|---|---|---|---|---|
| **FP8 GEMM** | `gemm_kernel[SIMD16 {32;1;1}{128;4;1}]` (**oneDNN**) | 1474 ms | **56.7%** | 964 (241/chunk) | ✅ **确认非 ESIMD**；~~compute 墙~~ → 见 opt#3：实测仅 18–21% XMX 利用率，非墙 |
| **未融合反量化** | `ElementwiseGlobalRangeKernel<float,3>` + CopyScalar (fp32) | 442 ms | **17.0%** | 3999 (≈1000/chunk) | ✅ **确认**；**最大可优化项** |
| TP allreduce | `oneccl_allreduce_pcie<half>` | 282 ms | 10.9% | 480 (120/chunk=2/层) | comm；PCIe |
| attention | `cutlass::fmha::XeFMHAFwdKernel` (**SYCL FMHA**) | 237 ms | 9.1% | 480 (120/chunk) | ✅ 确认 |
| act | `gelu_tanh_and_mul` (SYCL) | 36 ms | 1.4% | 240 (60/chunk) | ✅ |
| SYCL norm | `rmsnorm`/`FusedAddRmsNorm` | 36 ms | 1.4% | 1044 (261/chunk) | ✅ |
| FP8 act 量化 | `_per_token_group_quant_8bit` (**Triton**) | 35 ms | 1.3% | 960 (240/chunk=4/层) | ✅ 确认 Triton；**占比低** |
| post_ff norm | `_gemma_rmsnorm_residual_kernel` (**Triton**) | 16 ms | 0.6% | 240 (60/chunk) | ✅ 确认 Triton；**占比低** |
| QKV norm+RoPE | `esimd_qkv_split_norm_rope` (**ESIMD, fp16 sliding**) | 10 ms | 0.4% | 200 (50/chunk=全 sliding 层) | ✅ prefill 确实调用；**占比极低** |
| RoPE(global) | `fused_qk_rope...` (SYCL) | 2 ms | 0.1% | 40 (10/chunk) | ✅ |

**验证结论：**
1. **所有分发映射（表格 10 行）与 trace 一致，无一处误判。** 尤其 **prefill FP8 GEMM = oneDNN `gemm_kernel`，
   ESIMD `gemv_fp8_pert`/`gemm_fp8_pert` 在纯 prefill 窗口出现 **0 次**（首个 `gemv_fp8` 恰好标记 decode 起点）。
2. **解决 `FP8_GEMM_DPAS_V9` 存疑**：本 trace 里 prefill FP8 GEMM 的真实内核名 = `gemm_kernel[SIMD16 {32;1;1}{128;4;1}]`
   （oneDNN 的 "fast variant"，与 §ROOFLINE §3.1 lm_head 同族命名；另有 `{32;2;8}` slow variant 仅 x5）。旧笔记
   标成 ESIMD 属误判——**prefill 根本不进本套定制 ESIMD**，坐实。
3. **补齐相对开销排序**（此前标 TODO）：prefill 成本 = oneDNN FP8 GEMM 56.7%（compute 墙，不可动）+ **未融合 fp32
   反量化 17%（可优化）** + allreduce 11% + FMHA 9%，**其余每项 <1.5%**。

### 需实测确认（trace 已部分回答）

- **ESIMD `esimd_qkv_split_norm_rope` 在 prefill 是否高效**：trace 证实 **prefill 确实调用它**（50/chunk = 全部
  sliding 层，M=1024），但**仅 0.4%（~10ms/run）** → 即便未针对 M≫1 优化也**无性能价值可挖**；仅需确认正确性
  （gate `gemma4_causal.py:449-456` 无 M 限制、注释写 "decode"，语义存疑）。原 "必须 microbench 比性能" 降级：ROI 太低。
- **相对开销排序**：已由本 trace 给出（见上表）。**最值得动手 = 17% 的未融合反量化**（候选 #3），而非候选 #1/#2
  （它们各 <1.5%）。

### 候选优化（源码可行性，**已按 trace ROI 重排**）

1. **【最高 ROI，17%】prefill FP8 GEMM 的未融合反量化** — ✅✅ **已实现 + server 级验证全通过（gsm8k 0.975 精度中性、TTFT −13.5~15.3%、trace 确认反量化 kernel 消失，见下 "### opt#1 实现 & kernel 验证"）**。
   现状 = fp8 matmul 输出 fp32 → `output*x_scale*weight_scale`
   两趟 fp32 elementwise + cast（`ElementwiseGlobalRangeKernel<float,3>` 实测 480/chunk、占 prefill 17%）。若激活改
   per-tensor 量化即可命中 `_scaled_mm` 融合 epilogue（省 fp32 输出 + 2 趟 elementwise），代价是精度（per-tensor vs
   per-token），需 gsm8k gate。**trace 证实这是 prefill 唯一值得动的大项**（oneDNN GEMM 56.7% 是 compute 墙、动不了）。
2. **【低 ROI，1.3%】** FP8 激活量化 Triton → 已存在的 SYCL `sgl_per_token_group_quant_8bit`（改 `fp8_kernel.py` 的 XPU
   绑定）— ⛔ **未做，且已被 #1 绕开**：`fp8_kernel.py:704` 的 Triton 分发原封未动，但 #1 后 gemma4 prefill 改走
   SYCL **per-tensor** quant（`sgl_per_tensor_quant_fp8`），根本不再调用这条 Triton per-token-group quant → 对 gemma4
   基本无独立价值（详见下 "### opt#1 实现 & kernel 验证"）。trace 显示原也仅占 1.3%。
3. **【低 ROI，0.6%】** `gemma_rmsnorm_residual_scalar` 的 ESIMD 版放开到 prefill（去 `shape[0]==1` gate）。trace 仅 0.6%，
   基本可忽略。
4. **不可动**：~~oneDNN FP8 GEMM（56.7%，compute 墙）~~（**已修正**：实测仅 18–21% FP16-XMX 利用率，非 compute 墙，见 opt#3；但 native FP8-XMX 在 BMG 不支持，改善需 oneDNN tuning / 换 GEMM 实现）、TP allreduce（11%，comm/PCIe，需算法或互联层）、FMHA（9%，已 SYCL）。

### opt#1 实现 & kernel 验证（2026-07-02，GPU1 单卡 kernel 级，未起 server）

**改动**（`fp8_utils.py`，2 处，**UNSTAGED**）：
1. 模块级开关 `_XPU_FP8_PERTENSOR_PREFILL = _is_xpu and get_bool_env_var("SGLANG_XPU_FP8_PERTENSOR_PREFILL","true")` —
   **默认 ON**，`SGLANG_XPU_FP8_PERTENSOR_PREFILL=0` 即时回退到 per-token。
2. `apply_fp8_linear` 的 dynamic-activation `else` 分支新增 XPU 支路：当权重 per-tensor（`weight_scale.numel()==1`）时，
   激活也用 `scaled_fp8_quant(input_2d, None, use_per_token_if_dynamic=False)` 做 **per-tensor** 量化 →
   `per_tensor_activations=True` → 命中 `:~1800` 的**融合 `torch._scaled_mm` epilogue**，去掉未融合 fp32 反量化。
   **decode（M<=64）不受影响**：其在 `:1626-1659` 的 ESIMD 快路早退，根本不到此分支。

**kernel 级验证结果**（`ZE_AFFINITY_MASK=1`，gemma4 prefill shapes，M=1024）：

| 检查项 | 结果 |
|---|---|
| 融合路径在 XPU 可行 | ✅ `torch._scaled_mm(scale_a,scale_b,out=fp16)` 在 torch2.12/oneDNN 正常运行，5 shapes 无异常 |
| 融合 vs 手动反量化数值 | ✅ **位级一致**（同 scale，maxreldiff=0.0）——融合 epilogue 精确 |
| GEMM-path 收益 | 5 个 linear 合计 unfused **9.57ms → fused 7.78ms = −18.7%**（逐 linear 1.10–1.41×），与 trace 17% 一致；M=512/2048 同趋势 |
| per-tensor vs per-token 精度代理 | ✅ 即便 1000× token 幅度 + 30× 通道离群，cos 差 **<5e-5**（fp8 是浮点，相对精度~与 scale 粒度无关，不同于 int8） |
| 真实 `apply_fp8_linear` 集成测 | ✅ OFF/ON 切换无崩溃，输出 fp16 shape 正确，ON vs per-token 基线 **cos=0.9993** 全 shape |

脚本（host=容器）：`copilot_workspace/{fp8_dequant_microbench,fp8_quant_accuracy_proxy,fp8_apply_integration}.py`。

**✅ server 级 A/B gate 完成（2026-07-02，TP2 双卡 GPU0+1，容器重启后干净环境）**：

*正确性*——用**正确的 harness `gsm8k_chat_eval.py`**（chat 模板 `/v1/chat/completions`，baseline=0.975）。
⚠️ **不要用 `sglang.test.few_shot_gsm8k`**：它是 raw 5-shot completion，对 instruct 模型天然偏低（同一 ON server 上只有 0.775），
两种方法**不可比**，误用会造成"回退"假象（本次排查即因此走了弯路）。

| gsm8k (chat eval, n=40) | Accuracy | Invalid |
|---|---|---|
| baseline | 0.975 | — |
| **opt#1 ON** | **0.975 (39/40)** | 0.000 |
→ **精度中性**，与 baseline 完全一致。默认 ON 转为**正式**（不再是暂定）。

*性能*——同一 session 干净 A/B（`bench_bsz1.py`，bsz1 warmup2 trials3，output256）：

| input | OFF TTFT | ON TTFT | Δ TTFT | TPOT (both) |
|---|---|---|---|---|
| 1k | 621.5ms | **526.3ms** | **−15.3%** | ~40ms（不变，decode 未动）|
| 4k | 2641.0ms | **2255.8ms** | **−14.6%** | ~39ms |
| 8k | 5689.5ms | **4919.6ms** | **−13.5%** | ~40ms |

*trace 验证*——unitrace 干净单次 4096 prefill（4 chunk），baseline vs ON：

| kernel | baseline (OFF) | ON (opt#1) |
|---|---|---|
| **未融合反量化 `ElementwiseGlobalRange<float,3>`** | **299.2ms (11.5%) x1920** | **消失 (0)** ✅ |
| Triton `per_token_group_quant` | 34.8ms (1.3%) x960 | **消失 (0)** ✅ |
| fp32 中间 elem/copy/cast | 142.4ms x2083 | **4.3ms** ✅ |
| 新 `PerTensorQuantFP8Kernel`(+AbsMax) | — | 76.6ms |
| oneDNN GEMM | 1473.7ms | 1519.5ms |
| **prefill self-time 合计** | **2597.0ms** | **2275.2ms (−12.4%)** |

→ **确认**：未融合 fp32 反量化 kernel 被彻底消除、折叠进 `_scaled_mm` 融合 epilogue；prefill self-time −321.8ms/−12.4%，与 TTFT A/B（−13.5~−15.3%）一致。

**⚠️ 距离 target 仍有 gap（opt#1 单独不够）**：文档 target = 4k 1500ms / 8k 3500ms；当前 ON = 4k 2255.8ms / 8k 4919.6ms。
opt#1 拿下了 17%（未融合反量化）这一项，但 prefill 大头是 oneDNN GEMM（56.7%→66.8%，compute 墙）+ oneCCL collective（~12%）+ cutlass FMHA（~11%），
要达标需另找 GEMM/通信/attention 侧的优化，超出 opt#1 范围。

### opt#3 候选调查：oneDNN GEMM roofline + vLLM 对照 + fused-norm-quant（2026-07-02，standalone build GPU1）

用 `~/xiangyu/sgl_gemma/fp8gemm_bench/`（**独立最小 build**，只抽 W8A8/W8A16 oneDNN GEMM + fused-norm-quant，
`SyclExtension`+icpx 编成 `mini_fp8_C.so`，**未** build 整个 vllm `_xpu_C`）实测：

**① GEMM roofline —— 推翻"compute 墙"结论**：
- gemma prefill GEMM per-rank = 117.7 TFLOP / 1519.5ms = **77.5 TF/s**（in-server trace）；standalone 复现 65.5 TF/s（同量级）。
- BMG（32 Xe @2.8GHz）**FP16-XMX 峰值 367 TF/s**（⚠️ **FP8-XMX 本平台不支持**，oneDNN 把 fp8 上转到 fp16 XMX 计算，没有 native FP8 路径）。
- → **仅 18–21% XMX 利用率**。此前文档说"oneDNN GEMM 56.7% 是 compute 墙、动不了"**不准确**：远未打满，是 kernel/tile/占用率问题，不是硬件墙。（但 native FP8 翻倍那条路在 BMG 上**不存在**。）

**② sglang vs vLLM-xpu 分发路径对照**（两条都用 oneDNN，见 `analytics-zoo/vllm-xpu-kernels@add_gemma_xpu_graph`）：
| | sglang（opt#1 ON） | vLLM-xpu |
|---|---|---|
| 调用 | `torch._scaled_mm`（W8A8，激活也量化 fp8）| `fp8_gemm_w8a16`（**激活保持 fp16**，仅权重 fp8）|
| 实测 gemma shapes | W8A8 **65.5 TF/s** | W8A16 **69.4 TF/s** |
- W8A16 仅比 W8A8 快 **~6%**（聚合 1.06×，o_proj 最好 1.25×）→ **换 W8A16 不是 GEMM 层面的实质增益**。W8A16 的真正价值是**省掉整条激活量化链**。

**③ fused-norm-quant（vLLM `layernorm_quant.cpp` #267，已验证可移植）**：
把 fp8 激活量化**融进前面的 RMSNorm kernel**。实测 gemma H=5376：
| M | 未融合 (norm+quant) | FUSED | speedup |
|---|---|---|---|
| 1024 | 50.6us (28.3+21.6) | **25.7us** | **1.97×** |
| 2048 | 147.0us | 66.9us | 2.20× |
→ **融合后 quant 几乎免费**（fused ≈ norm-alone）。这能消除 opt#1 引入的独立 `PerTensorQuantFP8Kernel`+`AbsMax`（ON trace 里 76.6ms ≈ prefill 3.4%）。

**结论/下一步方向**（均**未实现**，待决策）：
- **opt#1 后的自然延伸**：把 opt#1 的独立 per-tensor quant 换成 **fused-norm-quant**（省 76.6ms/~3.4% prefill，且去掉一个 kernel）。
- **或走 W8A16**：prefill 激活不量化（GEMM 反正上转 fp16），省掉整条 quant/dequant 链——但 GEMM 本身只快 ~6%。
- **GEMM 本体**：18–21% XMX 利用率有空间，但需 oneDNN tuning / 换 GEMM 实现（cutlass-xe？），非小改；native FP8-XMX 在 BMG 不可用。

---

## ✅✅ opt#2（W8A16）—— 已实现并设为 XPU prefill 默认路径（2026-07-02，取代 opt#1）

**决策**：走 W8A16。实测证明它**整体优于 opt#1** 且**首次命中 TTFT target**，已设为**默认**（`SGLANG_XPU_FP8_W8A16_PREFILL` 默认 `true`；`fp8_utils.py`，UNSTAGED）。

**实现**：prefill 保持激活 **fp16 不量化**，跑 mixed `fp16 × fp8` oneDNN matmul（`fp8_gemm_w8a16`，从 `analytics-zoo/vllm-xpu-kernels@add_gemma_xpu_graph` 抽出，独立编成 `~/xiangyu/sgl_gemma/fp8gemm_bench/mini_fp8_C.so`，**未** build 整个 `_xpu_C`）。在 `apply_fp8_linear` 中**优先于 opt#1** 判断（M>64、per-tensor weight）；decode（M≤64）仍走 ESIMD 快路不变。`.so` 缺失时**优雅回退**到 opt#1 → per-token（三级 fallback）。

**server 级 A/B（TP2，正确 harness `gsm8k_chat_eval.py`）**：
| | gsm8k | 4k TTFT | 8k TTFT |
|---|---|---|---|
| OFF (per-token) | — | 2641ms | 5690ms |
| opt#1 (W8A8) | 0.975 | 2256ms | 4920ms |
| **opt#2 (W8A16)** | **0.975 (39/40)** | **1474ms** ✅(<1500) | **3358ms** ✅(<3500) |
→ W8A16 vs opt#1 **TTFT −32~37%**，精度中性，**命中 4k/8k target**。

**trace 验证**（unitrace，同 prefill 窗口 2880 gemm）：opt#1→W8A16 两个效果——
1. 激活量化链 `PerTensorQuant`+`AbsMax` **230ms → 0**（W8A16 不量化激活）。
2. oneDNN GEMM 本身 **4576ms → 2449ms（1.87× 更快）**：`torch._scaled_mm`(tile{128;4;1}) vs `fp8_gemm_w8a16`(tile{64;8;1})。
prefill self-time **8108ms → 5782ms（−28.7%）**。

**⚠️ follow-up（未解）**：in-server W8A16 GEMM=144 TF/s，但 standalone microbench 仅 69 TF/s（2× gap）。说明 TTFT 大胜**同时来自**去量化**和**更快的 oneDNN GEMM dispatch，非 W8A16 语义本身（microbench 里 w8a16 vs w8a8 仅 1.06×）。含义：即便留在 W8A8 换用 mini_fp8 的 GEMM dispatch 可能也能拿到大部分 GEMM 加速——**归因待厘清**（决定最优实现是"W8A16"还是"换 GEMM kernel"）。

**fused-norm-quant —— 在 W8A16 下已多余**：它是"把 fp8 激活量化融进 RMSNorm"，但 W8A16 **不量化激活**，无事可融。且已验证 **decode 也不量化激活**（ESIMD gemv 拿 fp16 直接算，trace 纯 decode tail 里 `PerTensorQuant`/`AbsMax`=0）→ **fused-norm-quant 对 prefill(W8A16) 和 TPOT 均无收益**，仅对 opt#1(W8A8) 有用，而 W8A16 已取代 opt#1。（standalone 实测 fused vs unfused = M1024 1.97×，仅供 W8A8 路径参考。）

### W8A16 默认路径完整性能矩阵（TP2 eager，bsz1，output=256，2026-07-02）

| input | TTFT (ms) | TPOT (ms) | decode tok/s | E2E (s) | vs 旧 baseline TTFT |
|-------|-----------|-----------|--------------|---------|---------------------|
| 1024  | 329.6 | 43.6 | 23.0 | 11.44 | 621 → **−47%** |
| 2048  | 694.1 | 43.4 | 23.0 | 11.76 | 1276 → **−46%** |
| 4096  | 1473.8 | 43.0 | 23.3 | 12.43 | 2644 → **−44%** ✅target1500 |
| 8192  | 3354.3 | 43.8 | 22.9 | 14.51 | 5721 → **−41%** ✅target3500 |
| 16384 | 8089.5 | 43.3 | 23.1 | 19.13 | 12805 → **−37%** |
| 32768 | 21719.2 | 44.8 | 22.3 | 33.15 | 31100 → **−30%** |
| 65536 | 65324.7 | 50.4 | 19.8 | 78.19 | 84143 → **−22%** |

（TTFT 收益随 input 增大而收敛：prefill 里去掉的量化+反量化是固定比例开销，input 越大 oneDNN GEMM/attention 占比越高。）

#### TPOT 的 43 vs 39ms 差异——已实测定因（非本优化引入，已证）

用户质疑不能凭空归为"噪声"，故做了**同一 server 的运行时 A/B**（无 unitrace，warmup2/trials3）：

| 配置 | 1k TPOT | 4k TPOT |
|------|---------|---------|
| W8A16 **ON**（默认） | 43.3 | 42.5 |
| W8A16 **OFF**（opt#1） | 43.4 | 44.3 |

- **W8A16 对 TPOT 中性**：ON≈OFF（OFF 甚至略高）。代码层面 decode（M=1）在 `apply_fp8_linear` 的 **ESIMD 早退**（`input_2d.shape[0]<=64`，line ~1673-1706）就 return，**根本走不到** W8A16 分支（gated `>64`，line ~1712）。故 opt#2 不可能影响 decode。trace 亦证 `GEMV_fp8_pert <256>/<512>` 逐 step self-time 在 baseline/opt#1/W8A16 三者**逐字节相同**（16.57/8.87ms）。
- **39→43 的绝对漂移是共享节点的 host-CPU 争用**，不是本优化：旧 39ms baseline 测于 2026-07-01 05:30（清晨、节点空闲）；今日实测于 07-02 白天，`uptime` load≈5、9 users，`ps` 显示**多个他人的 sglang::scheduler 进程钉在 ~100% CPU**（root 所有，PID 与本 server 的 11309/11310 不同，etimes 596s–28164s = 8+ 个并存 server）。bsz=1 decode 的 TPOT 含大量 host 侧逐 kernel 派发，CPU 被同租户占满时 per-step host 时间被均匀抬高 → 各 ctx 平坦 +3~4ms（今日 43 平坦 vs 旧 39 平坦），且 ON/OFF 同等受影响。
- **教训（存档）**：unitrace 下的 allreduce/timing 不可靠（会被显著放大），运行时才稳定；结论一律以运行时 bench 为准，勿据 unitrace 的 AR 数字下判断。

---

**基建教训（本次排查踩坑）**：① server 反复"崩溃"= 另一并行 session 的孤儿 launcher（`srv_ON_v2`）+ 中断遗留的 gsm8k 客户端抢同一 30000 端口和 GPU0/1，非 opt#1 bug；干净做法 = 容器重启清场 + `docker exec -d` 启动（`setsid`/同-exec 长 babysit 会被 TTY teardown 误杀）。② 容器重启后**第一次** TP2 启动 TP0 可能 weight-load 挂死，重启一次即好。③ **unitrace flush 必须 `kill -2` 直接发给 `sglang::scheduler_TP0/TP1` 子进程 PID**（发给主进程会被 kill_process_tree SIGKILL 掉子进程、只落 54KB 空 trace）。

**opt#2（1.3%，quant Triton→SYCL）**：**BLOCKED / 大概率作废**——#1 后 prefill 走 per-tensor 量化，per-token Triton
quant 已不在热路径。


## DECODE 阶段 Kernel 融合分析（trace 单层时序 + 源码，2026-07-02）

> 数据源：W8A16 默认 decode trace（`utrace_w8a16` TP0）单层内核链 + `gemma4_causal.py` decoder 层源码交叉确认。
> decode 内核在 W8A16 ON/OFF 下已证逐字节相同，故该分析与 prefill 优化无关，是独立的 decode 优化线。

### 单 decoder 层内核链（融合状态）

| # | 内核（trace） | 源码 | 融合状态 |
|---|--------------|------|---------|
| 1 | RmsNorm 7.6us | `input_layernorm(x)` L741 | ❌ **独立未融** → ②目标 |
| 2 | gemv_fp8`<256>` | qkv_proj | — |
| 3 | copy | qkv 连续化 | ❌ 冗余 → ④目标 |
| 4 | qkv_split_norm_rope | ESIMD 融合核 L467 | ✅ split+q/k norm+rope |
| 5 | RmsNorm 2.7us | `v_norm(v)` L479（核注释 L476 明说不做 V）| ❌ **未融** → ③目标 |
| 6 | index/scatter ×2 + elem ×3 | KV 写入 `save_kv_cache=True` | ❌ **未融**（融合路径 L556-571 存在但 `can_fuse=False` 硬关，注释"causes accuracy regression"）→ ①目标 |
| 7 | paged_attn ph1/ph2/ph3 | attention | — |
| 8 | gemv_fp8`<512>` | o_proj（含 AllReduce）| — |
| 9 | AllReduce | TP | ⚠️ 集合通信难融；unitrace 下时延不可信 |
| 10 | RmsNorm 7.3us | `post_attention_layernorm` L747 | ❌ **独立未融** → ②目标 |
| 11 | FusedAddRmsNorm 16us | `pre_feedforward_layernorm(h,res)` L807 | ✅ residual-add+norm（ESIMD）|
| 12 | gemv_fp8`<256>` 196us | gate_up_proj（最大）| — |
| 13 | gelu_mul | `GeluAndMul` | ✅ gelu+gate*up |
| 14 | gemv_fp8`<512>` | down_proj（含 AllReduce）| — |
| 15 | AllReduce | TP | ⚠️ |
| 16 | RmsNormResidual 13us | `post_feedforward_layernorm` L834 | ✅ residual-add+norm+scalar（ESIMD）|

### 待融合项（按实施顺序 ③→④→②→①）

- **③ v_norm 融进 qkv_split_norm_rope**（~60 核/step）：核目前只做 Q/K norm（L476 注释），V norm 是 `with_scale=False` 纯 RMSNorm，另起一核。扩展 ESIMD 核顺带做 V。改动小、风险低。 **✅ 已实现（见下）**
- **④ elem/copy/cast 清理**（~237 核/step = 4/层）：qkv 后 contiguous、v reshape、fp8/fp16 cast 等零碎核，纯 launch-overhead，Python 层可去冗余。 **✅ 已实现（见下）**
- **② input_layernorm / post_attention_layernorm 融合**（~120 核/step）：16 个 norm 里仅这 2 个未融（pre_ff/post_ff 已 ESIMD 融了 residual-add）。`input_layernorm` 可折进 qkv_proj GEMV prologue；`post_attn` 因 o_proj 内 AllReduce 阻隔较难。需 kernel 改动。 **✅ 已实现（post_attn 部分，见下）**
- **① KV-cache 写入融合**（~120 scatter/step → 60）：⚠️ 原 scope（re-enable `can_fuse`）对 decode **无效**（decode 走 ESIMD 快路径，不经过该块）。真实目标 = `_set_kv_buffer_impl` 的 XPU naive 双 scatter（`k_cache[idx]=k; v_cache[idx]=v`）合成 1 个 ESIMD 融合 scatter 核。详见下方"① 实现方案"。**✅ 已实现 + 验证通过（2026-07-03，gsm8k_chat 0.975=baseline、微基准位级一致；md5 9dd0586e）。**

### 对 TPOT 的现实预期
真实 decode 由 240 个 fp8 GEMV（bandwidth-bound）主导，上述均为**小核**，GPU 时间占比低。融合收益主要是**削减 kernel launch 数**（当前 ~1351 核/step）→ 在 host-dispatch / CPU 争用敏感场景压低 per-step host 开销，量级"几个 %"，**不改变 GEMV 带宽下限**。真正大头仍需从 GEMV 本体带宽利用率入手。
每项改动遵循规则：**rebuild → gsm8k 门（≥0.975）先过，再报 perf**。

### ✅ ③ v_norm 融合 — 已实现并验证（2026-07-02）

**改动**（UNSTAGED）：
1. `custom-esimd-kernels-sglang/csrc/xpu/esimd_kernels/qkv_split_norm_rope.h` V 分支：由 "copy only" 改为在核内做纯 RMSNorm（`with_scale=False`：`x * rsqrt(mean(x²)+1e-6)`，无 weight、无 RoPE）。
2. `gemma4_causal.py` ESIMD 路径（L476-479）：删除独立 `self.v_norm(v_3d)` 调用（核已代做），保留 unflatten reshape。
   非 ESIMD 融合路径（`gemma_qkv_rmsnorm`）本就已含 V norm，不受影响。
3. Rebuild：`custom-esimd-kernels-sglang` `custom_esimd_kernels` .so（`-j1` 串行，避免 xpu 扩展共享 `sycl_dlink.o` 的并行冲突）；新 md5=739539ad（旧 3806e20f）。

**验证**：
- 数值：微基准 V 输出 vs 参考 RMSNorm **位级一致**（maxabs=0.0，cos=1.0），且确非 copy（与原始输入差 0.41）；Q-norm 仍 cos=1.0。
- 正确性：gsm8k_chat_eval n=40 = **0.975（39/40）= baseline**，中性。
- 性能：TPOT 1k=42.3ms / 4k=42.1ms，TTFT 331/1471ms — 与 W8A16 默认矩阵一致，**无回退**（节点争用态 ~42-43ms band 内）。
- 收益：按构造去掉 60 个 standalone RmsNorm 核/step（1/层×60 层）。带宽受限 decode 下 TPOT 不可见变化（符合预期），价值在 launch-count 削减。


### ✅ ④ elem/copy/cast 清理（positions int32 缓存）— 已实现并验证（2026-07-02）

**改动**（UNSTAGED，纯 Python，无 rebuild）：
- `gemma4_causal.py` Gemma4Attention.forward ESIMD 路径：`positions.to(torch.int32)` 原来**每层都做一次**（60 层 → 60 次 UnrolledElementwiseKernel/step），但 `positions` 在整个 step 内所有层完全相同。改为在 `forward_batch` 上按 `positions` 对象身份缓存 int32 结果（每 step 新建 `forward_batch` → 缓存自动失效，无跨 step 污染）。60 次 cast → 1 次/step。

**验证**：
- 正确性：gsm8k_chat_eval n=40 = **0.975（39/40）= baseline**，中性。
- 收益：去掉 ~59 个 int32 cast 核/step。同为 launch-count 削减，TPOT 带宽受限不可见。

### ✅ ② post_attention_layernorm 融合（`esimd_norm_add_norm`）— 已实现并验证（2026-07-02）

**关键发现**：`esimd_norm_add_norm` 核**早已编译在 `custom_esimd_kernels` .so 内**（`norm_add_norm.h`，docstring 明写 "for gemma4 attn-output fuse points… Replaces 2 launches"），语义正是 `residual = rms_norm(attn_out)*w1 + residual; out = rms_norm(residual)*w2`。**因此 ② 是纯 Python wiring，无需 rebuild。**

**改动**（UNSTAGED，纯 Python）：
- `gemma4_causal.py`：导入 `_esimd_norm_add_norm`（L69-82）。
- `DecoderLayer.forward`：原先 `post_attention_layernorm` 在 moe/dense 分叉**前无条件**执行。重构为：MoE 分支保留 standalone `post_attention_layernorm`（语义不变）；**dense 分支**（gemma4-31B dense 全走此路）在满足 `bsz==1 / fp16 / contiguous` 时用 `_esimd_norm_add_norm(attn_out, residual, w1, w2, out=attn_out, eps1, eps2)` **一个核**同时完成 `post_attn_norm(attn_out)` + residual-add + `pre_ff_norm`，替代了原来的 standalone post_attn RmsNorm **和** `_esimd_fused_add_rms_norm` 两个核。权重按 RMSNorm 约定用**原始** `.weight.data`（无 +1）缓存为 `self._nan_w1/_nan_w2`。`out` 可安全 alias `attn_out`（核 3 遍顺序流式，pass 3 不再读 h2_raw）。保留 `_esimd_fused_add_rms_norm`→纯 Python 作为逐级 fallback。

**验证**：
- 数值：微基准 `esimd_norm_add_norm` 输出 cos≈1.0、residual in-place 正确（K=2560/2816/4096；hidden_size=5376 → VL=256 路径）；差异仅 fp16 舍入。
- 正确性：gsm8k_chat_eval n=40 = **0.975（39/40）= baseline**，中性（融合路径在 bsz=1 decode 触发；gsm8k parallel=16 的 batched decode 走 fallback，同样正确）。
- 性能：bsz=1 bench_bsz1 TPOT 1k=39.5ms / 4k=39.3ms（节点当前空闲态），无回退。
- 收益：dense decode 每层省 1 个 standalone post_attn RmsNorm 核（60/step），并把 pre_ff 的 fused-add 也并入同一核。
- 注：**② 只融了 `post_attention_layernorm`**（折进 pre_ff 的 fused-add 路径）。另一个未融的 `input_layernorm` 喂 qkv GEMV prologue，需改 gemm-package 核，更难，未做。

### 🔑 ① KV-cache 写入融合 — 关键发现 + ESIMD 实现方案（✅ 已完成 2026-07-03）

**关键发现（原 scope 无效）**：原计划的 "re-enable `can_fuse=False`（`create_fused_set_kv_buffer_arg` 的 RoPE+KV 融合路径，`gemma4_causal.py` L574）" 对 **decode 完全无效**。因为 decode（head_dim==256 / fp16 / contiguous / non-kv-shared）**总是走 ESIMD 快路径**（`gemma4_causal.py` L459-502），在 L502 就 `return`，**根本不进入** `can_fuse` 所在的非-ESIMD fallback 块（L555-603）。ESIMD 路径里 RoPE 已在 `qkv_split_norm_rope` 内融合，KV 写发生在 `self.attn(..., save_kv_cache=True)` 内。

**真实的 ① 目标**：decode 的 KV 写 = `xpu_backend.forward_decode` → `token_to_kv_pool.set_kv_buffer` → `_set_kv_buffer_impl`（`memory_pool.py` L97）的 **XPU naive fallback**（L141-143）：
```python
k_cache[indices] = k   # advanced-index scatter #1
v_cache[indices] = v   # advanced-index scatter #2
```
即 trace row6 的 "index/scatter ×2"。CUDA/HIP 有单发 `store_cache` JIT 核（`StoreKVCacheKernel`，`.cuh`，**CUDA-only**）把两者合一；XPU 无对应实现，故落到两个独立的 torch 高级索引 scatter（每层 2 发 → 60 层 × 2 = **120 scatter/step**）。

**路由确认**：gemma4 用 SWA 内存池（sliding + full 层），但 `SWAMemoryPool.set_kv_buffer` 只是把 full 层 / SWA 层分别委托给两个 `MHATokenToKVPool` 子池，二者都最终调用 `_set_kv_buffer_impl` → 同一 XPU naive fallback。因此 **`_set_kv_buffer_impl` 是唯一的统一改造点**。

**实现方案（ESIMD，非 Triton — 遵循"不新增 Triton 核"约束）**：
1. **新核** `csrc/xpu/esimd_kernels/kv_scatter.h`（仿 `norm_add_norm.h` 流式风格，纯 copy 无数学）：
   ```cpp
   template<int VL> struct KVScatter_kernel {
     const fp16 *k_ptr, *v_ptr;   // [T, row_dim]  连续
     fp16 *k_cache, *v_cache;     // [S, row_dim]
     const int64_t* idx_ptr;      // [T]  token 槽位（out_cache_loc）
     int T, row_dim;
     void operator()(nd_item<1> it) const SYCL_ESIMD_KERNEL {
       int t = it.get_global_id(0); if (t >= T) return;   // 每线程 1 token
       int64_t dst = idx_ptr[t];
       const fp16* ks = k_ptr + (int64_t)t*row_dim; fp16* kd = k_cache + dst*row_dim;
       const fp16* vs = v_ptr + (int64_t)t*row_dim; fp16* vd = v_cache + dst*row_dim;
       for (int c=0;c<row_dim/VL;c++){ int o=c*VL;
         block_store<fp16,VL>(kd+o, block_load<fp16,VL>(ks+o));
         block_store<fp16,VL>(vd+o, block_load<fp16,VL>(vs+o)); }
     }};
   ```
   - gemma4 形状：kv_heads=16, head_dim=256 → **row_dim = 4096 fp16 (8KB)**，4096 % 256 == 0 → VL=256、16 chunk，无 tail。bsz=1 decode → T=1（单线程单 WG，与 norm_add_norm 同量级）。
   - host launcher 用 `nd_range`（T 线程，round-up 到 WG size），仿 `fused_add_rms_norm_batched.h` 的多-token 发射惯例。
2. **host 包装** 加进 `csrc/xpu/esimd_kernel.sycl`：`void esimd_kv_scatter(Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, Tensor indices)`（reinterpret 到 fp16*/int64*，`row_dim = k.size(-1)`，`T = k.size(0)`）。
3. **注册** 进 `csrc/xpu/torch_extension.cc`：`m.def("esimd_kv_scatter(Tensor k, Tensor v, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor indices) -> ()")` + `m.impl(..., torch::kXPU, &esimd_kv_scatter)`。
4. **Python binding** 进 `python/custom_esimd_kernels_sglang/ops.py`（仿 `esimd_norm_add_norm`）。
5. **接线** `memory_pool.py::_set_kv_buffer_impl` 在 naive fallback 前加 XPU 支路（带优雅回退）：
   ```python
   if _is_xpu and same_kv_dim and store_dtype.itemsize == 2 and _esimd_kv_scatter is not None:
       row_dim_ = k.shape[1] * k.shape[2] if k.dim()==3 else k.shape[-1]
       if row_dim_ % 256 == 0:
           _esimd_kv_scatter(k.reshape(-1,row_dim_), v.reshape(-1,row_dim_),
                             k_cache.view(-1,row_dim_), v_cache.view(-1,row_dim_),
                             indices if indices.dtype==torch.int64 else indices.to(torch.int64))
           return
   # else -> 现有 naive k_cache[indices]=k; v_cache[indices]=v
   ```
   - 语义与 naive 逐位一致（纯 copy，无 div/cast：gemma4 KV cache dtype=fp16=模型 dtype，`cache_k.dtype==self.dtype` → 不进 scale/cast 分支）。`same_kv_dim=(head_dim==v_head_dim)=True`。
   - `indices` 若为 int32 需转 int64（小张量，一次微小 cast；净省 1 发 scatter）。实现时确认 `out_cache_loc` 实际 dtype，若已 int64 则零额外核。

**风险 / 预期**：纯 copy scatter，**预期位级一致、gsm8k 中性**（低精度风险，明显低于原"RoPE+KV 融合致精度回退"的 scope）。收益仅 **launch-count（120→60 scatter/step）**；decode 带宽受限 → **TPOT 预期无可见变化**（与 ③④② 一致）。**需 rebuild（`-j1`，~8-10min）+ gsm8k 门控（≥0.975）后才报 perf**。

**✅ 已实现 + 验证通过（2026-07-03，容器 `txytest_sgl_bmg`，GPU 0,1）**：
- **落地**：新核 `csrc/xpu/esimd_kernels/kv_scatter.h`（每 WG 1 token，纯 `block_load→block_store` fp16 copy，VL 自适配：SWA row_dim=4096→VL512、full row_dim=2048→VL512）；host 包装 `esimd_kv_scatter` + include 进 `esimd_kernel.sycl`；原型进 `include/kernel_ops.h`；注册进 `csrc/xpu/torch_extension.cc`（`m.def/m.impl kXPU`）；binding 进 `python/.../ops.py` + 导出进 `__init__.py::_EXPORTS`；接线 `memory_pool.py::_set_kv_buffer_impl` XPU 支路（守卫 `_is_xpu ∧ same_kv_dim ∧ store_dtype.itemsize==2 ∧ row_dim%32==0 ∧ numel>0`，int32→int64 自动 cast，不满足则优雅回退 naive）。
- **rebuild**：`MAX_JOBS=1 python3 setup.py build_ext --inplace` 成功；`custom_esimd_kernels.so` md5 `739539ad → 9dd0586e`。
- **(a) 微基准位级一致**：4 shape（T=1/4 × SWA 4096；T=1/8 × full 2048）+ int32-indices 路径，`esimd_kv_scatter` 结果 vs `k_cache[indices]=k` 参考 **全部 `torch.equal`=True**。
- **(c) server gsm8k**：**正确 harness `gsm8k_chat_eval.py`（chat 模板 /v1/chat/completions）n=40 = 0.975（39/40），Invalid 0.000 = baseline，中性 ✅**。（⚠️ 教训：`sglang.test.few_shot_gsm8k` 是 raw 5-shot、instruct 模型天然偏低——本次误用只得 0.800，与 baseline 无关；门控必须用 chat harness。）
- **支路生效核实**：server env `_is_xpu=True, _esimd_kv_scatter loaded=True`（每 decode step bsz=1 触发融合路径）；相干性 sanity（"Paris" / 17×24=408 正确）。

**收益 / 结论**：launch-count 120→60 scatter/step（如预期）；decode 带宽受限 → **TPOT 无可见变化**（与 ③④② 一致）。价值在 host-dispatch 削减与代码整洁。

### DECODE 融合小结（③④②①）

| 项 | 内容 | 状态 | rebuild | gsm8k | launch 削减/step |
|---|------|------|--------|-------|-----------------|
| ③ | v_norm 融进 qkv_split_norm_rope（ESIMD 核内） | ✅ 完成 | 是(-j1) | 0.975 | ~60 RmsNorm |
| ④ | positions int32 缓存（纯 Python） | ✅ 完成 | 否 | 0.975 | ~59 cast |
| ② | post_attn_norm 融进 pre_ff（复用已编译 esimd_norm_add_norm） | ✅ 完成 | 否 | 0.975 | ~60 RmsNorm |
| ① | KV 写 2 scatter → 1（新 ESIMD kv_scatter 核） | ✅ 完成 | 是(-j1) | 0.975 | ~60 scatter |

共性结论：decode 由 240 fp8 GEMV（bandwidth-bound）主导，四项均为**小核 launch-count 削减**，**不改变 GEMV 带宽下限 → TPOT 不可见变化**（③②实测印证）；价值在 host-dispatch/CPU 争用敏感场景的 per-step 开销与代码整洁。真正的 TPOT 大头仍需从 GEMV 本体带宽利用率入手。

### ✅ ①②③④ 全生效 trace 核实（unitrace，bsz=1 30-step decode，2026-07-03）

> 数据源：server 挂 unitrace（`--chrome-call-logging --chrome-device-logging`），bsz=1 打 30 token decode，`kill -2` 直发 `sglang::scheduler_TP0/TP1` 子进程 flush（发主进程会被 kill_process_tree SIGKILL → 空 trace，见基建教训③），逐 rank `python3.<pid>.json` 解析 `cat=gpu_op` 事件、按 step 归一。**四项融合全部落地、无回退旧路径。**

| 融合项 | 融合前（naive/独立核） | 融合后（trace 实测） | 削减 |
|-------|----------------------|---------------------|------|
| ① KV 写 scatter | `IndexPut` 2/层 = **120/step** | `KVScatter` **60/step**（1/层）；旧 KV `IndexPut` 从 4560→140/trace（残留=非-KV 用途）| **−60/step** |
| ③ V RmsNorm | standalone `RmsNorm` **60/step** | 0（并入 `qkv_split_norm_rope`，**50/step** = SWA 50 层）| **−60/step** |
| ② post_attn RmsNorm | standalone `RmsNorm` **60/step** | 0（并入 `NormAddNorm` **60/step**）| **−60/step** |
| ④ positions int32 cast | `CopyScalarFunc<int>` **60/step** | **~1/step**（forward_batch 缓存）| **−59/step** |

- 核实要点：新核 `KVScatter`/`NormAddNorm`/`qkv_split_norm_rope` 逐 step 计数干净稳定；旧 `IndexPut` KV-scatter 已消失（4560→140/trace，残留为非-KV 高级索引）；int32 cast 从 60/step 降到 ~1/step（非 60）。**四项确认全部生效**。

### 刷新的性能矩阵（①②③④ 全生效，VL512，eager TP2，bsz1，out=**512**，2026-07-03）

| input | TTFT (ms) | TPOT (ms) | decode tok/s | E2E (s) | TPOT vs baseline(out256) |
|-------|-----------|-----------|--------------|---------|--------------------------|
| 1024  |   356.7   | **37.75** | 26.5 | 19.65 | −1.48 (39.23) |
| 2048  |   712.1   | **37.37** | 26.8 | 19.81 | — |
| 4096  |  1482.3   | **37.76** | 26.5 | 20.78 | −1.46 (39.22) |
| 8192  |  3334.8   | **38.54** | 25.9 | 23.03 | −0.73 (39.27) |
| 16384 |  8010.6   | **40.11** | 24.9 | 28.51 | −0.88 (40.99) |
| 32768 | 21492.4   | **43.23** | 23.1 | 43.58 | −0.93 (44.16) |
| 65536 | 64966.0   | **49.47** | 20.2 | 90.25 | −0.98 (50.45) |

- **TPOT 全程略优于 clean baseline（~1ms，节点空闲态）**，符合"带宽受限 + 纯 launch-count 削减 → 无回退、边际收益"的预期。1k–4k 稳定 ~37.5ms（比旧 39.2ms 低 ~1.5ms），16k 起因 10 层 global split-K 扫全 KV 平滑上升至 64k=49.5ms。
- TTFT 随 input 近线性（64k≈65s），与 baseline 一致（本轮矩阵为 out=512，baseline 表为 out=256，仅 TPOT 直接可比；TTFT 与 output 长度无关，量级相符）。
- 结论：①②③④ 在 decode 削减 ~239 小核/step（60+60+60+59），**TPOT 中性偏优、精度中性（gsm8k_chat 0.975）**，已 commit（`eb7fd8a609`，`memory_pool.py`+`gemma4_causal.py`）。真正 TPOT 大头仍在 240 fp8 GEMV 带宽。


## GEMV tile 复调优：VL512→256（采纳）、VL128_KS2（否决）(2026-07-03)

推翻上文 2026-07-02「保留 512」的旧结论。旧 A/B 是 **不同 binary + 非同时刻**（rebuild-confound + 节点争用），
本轮改用 **同一 binary + env-gate**（`SGLANG_GEMV_VL_CAP`、`SGLANG_GEMV_KS_MODE`，`select_vl_ks` 内 `getenv`
一次性缓存 + 首调 GEMV 打印生效值），彻底消除混淆，并逐项用 unitrace device-trace 坐实。

### 关键方法论：孤立 kernel 微基准会**低估** in-server 带宽
- 孤立 `bench_gemv_vlks`（fresh-DRAM pool > cache，cos=1.0 gate）测 VL512 down_proj=533GB/s(85%)，
  但 **in-server** VL512 down_proj=576GB/s(91%)。差 ~6pt。故微基准预测的 kernel 提速在 E2E 大幅缩水/消失。
- 这正是旧「保留 512」误判之源：微基准显示 VL256 快，实机两者都已贴墙。

### ① VL512 → VL256（3 个 `<512>` shape：o_proj slide/full、down_proj）→ **采纳**
- kernel 微测（fresh-DRAM，cos=1.0）：这 3 个 K∈{4096,8192,10752} shape VL512=84-85% → VL256=90-93%
  （VL512 占用减半 = large-GRF）。K=5376 的 gate_up/qkv 因 5376 不被 512 整除，divisibility loop 本就落到 256，不受默认改动影响。
- 同-binary env A/B（out=512，节点空闲）：VL512 vs VL256 = 1k/4k/8k **−0.06ms（中性偏优，非回退）**。
- unitrace 坐实：`GEMV<512,1>` 120→**0/step**，`<256,1>` →234.5/step；in-server down_proj 91→93%、44M band 86→90%。
- gsm8k_chat **0.975**（cos=1.0 位级一致）。**默认改 256**（`SGLANG_GEMV_VL_CAP` 默认 256，`=512` 可复现旧 prod）。

### ② VL128_KS2（K=5376：gate_up + qkv）→ **否决（默认 OFF）**
- kernel 微测：K=5376 shape VL256=86-92% → VL128_KS2=91-94%（gate_up 92→94%，qkv 86→91%）。
  但 VL128_KS2 **伤** o_proj/down_proj（K≠5376，跌到 83-87%）→ 只能按 K==5376 keying。
- 同-binary env A/B（`SGLANG_GEMV_KS_MODE` 0 vs 1，out=512）：

  | ctx | KS=0 (VL256-only) | KS=1 (K5376→VL128/2) | Δ |
  |-----|------|------|------|
  | 1k | 37.69 | 37.79 | **+0.10** |
  | 4k | 37.68 | 37.79 | **+0.11** |
  | 8k | 38.46 | 38.56 | **+0.10** |

- unitrace 逐 shape 坐实**零收益**：gate_up in-server KS0 `<256,1>`=195.6us→**591GB/s(94%)** vs
  KS1 `<128,2>`=196.0us→**590GB/s(94%)**，**完全相同**。微测的 94% vs 92% 优势在实机消失（实机 VL256 本就 94%）。
  total GEMV/step KS0=24715us → KS1=24814us，**+99us/step**，与 E2E +0.10ms 精确吻合。
  `<128,2>` 的额外 k-split reduction 开销纯是负担 → 造成回归。
- gsm8k_chat **0.975**（cos=1.0）。**默认 OFF**（`SGLANG_GEMV_KS_MODE` 默认 0，仅 env 显式 `=1` 开启作实验）。

### 落地
- `custom-esimd-kernels/csrc/xpu/esimd_kernels/fp8_GEMM_pert.h::select_vl_ks`：默认 VL0=256（env `SGLANG_GEMV_VL_CAP`）；
  新增 env-gated `if(SGLANG_GEMV_KS_MODE==1 && N>512 && K==5376){vl=128;ks=2;}`（默认 0）。备份 `.bak_vltune`。
- 重建 gemm ext（`touch csrc/xpu/esimd_kernel_gemm.sycl` + `MAX_JOBS=2 setup.py build_ext --inplace`），
  出厂 `.so` md5 `f8f84e70`（VL256-only + KS OFF）。KNOWN_GOOD 旧 `.so` `42e28795` 仍在。
- **净收益**：VL256 采纳后 1k/4k=37.7ms（旧 VL512 clean-bench ~39.2ms 中大部分差异实为噪声；同-binary 对比仅 −0.06ms），
  TPOT 中性偏优、精度 0.975。GEMV 带宽已贴墙（gate_up 94%、总 92%），tile 层面无更多可榨空间。

**共性结论重申**：bsz=1 decode 的 240 fp8 GEMV 已在 ~92% roofline 带宽墙，tile（VL/KS）微调在实机无可见收益；
孤立微基准的"提速"是低估 in-server 带宽的假象。真正 TPOT 大头需从**减少 DRAM 流量**（如 KV/权重复用、算子融合削 traffic）而非 tile 入手。


## ★★★ PREFILL FMHA (HD512 full-attn) tile 调优 — 采纳 TILED_Q 256→128（2.3× kernel，TTFT −29~38%）(2026-07-03)

### 关键发现：full-attention 层 head_dim = **512**（不是 256），此前所有 HD256 FMHA 工作打错了目标
`SGLANG_FMHA_DEBUG` dump `forward_extend` 全注意力层（L5/11/17…59，共 10 层）：
`q=(tok,16,512)  k_cache=(pages,64,2,512)  fp16  causal  softcap=0  scale=1`。
配置坐实为**设计如此**（`gemma4_causal.py` L622-628）：
- `global_head_dim=512`（10 个 full-attention 层，O(n²)）、`num_global_key_value_heads=4`(/rank=2) → GQA 16:2
- `head_dim=256`（50 个 sliding 层，window=1024）、`num_key_value_heads=16`(/rank=8)

→ prefill FMHA 大头 = **HD512 full-attn 核**（32k prefill 占 43.9% device time / ≈51% TTFT，64k 占比更高）。
此前的「孤立微基准 78 TF/s、HD256 tile 调优」全测的是 **sliding 核（仅占 prefill 1.8%）**，
且「10× 孤立 vs in-server 之谜」纯是**微基准 head_dim 打错**——改成 HD512 后 replay 32k = 11078ms/34.6ms-per-call
与 in-server unitrace 11255ms/35.2ms **精确吻合**，无 gap、无谜团。

### HD512 tile sweep（`FMHAPrefillXe20.cmake` L37-39 + `ninja …_512` + copy .so + replay 32k）
基线 TILED_Q=256/TILED_KV=64/NUM_SG=32（与 HD256 相同、**未针对 2× head_dim 调**）→ 4.3% XMX peak（367 TF/s BMG FP16）。
根因：Q tile [256×512] + O 累加器 256×512 寄存器足迹巨大 → 溢出 / 低占用。

| TILED_Q | TILED_KV | NUM_SG | replay 32k | %peak | cos | 结论 |
|---------|----------|--------|-----------|-------|-----|------|
| 256 | 64 | 32 | 11078 ms | 4.3% | 1.00000 | 基线 |
| **128** | **64** | **32** | **4763 ms** | **10.1%** | **1.00000** | **✅ 采纳（2.33×）** |
| 64  | 64 | 32 | 8481 ms | 5.7% | 1.00000 | 过小、欠利用，否决 |
| 128 | 128 | 32 | 17088 ms | 2.8% | 0.06 | **破坏正确性**，否决 |
| 128 | 64 | 16 | 11241 ms | 4.3% | 1.00000 | SG 减半回落基线，否决 |

- 微测逐 shape（cos=1.0 全通过）：full 2k/4k/8k 由 ~4.2% → **10.8-11.3%**；chunk 1k×{8k,16k,32k} → **10.0-10.4%**。
- TILED_KV 必须保持 64（128 破坏正确性，疑与 page_size=64 / mask 逻辑绑定）。

### E2E 验证（TP2 eager，bsz1，out=512，new .so，chunk=1024）+ gsm8k gate
| input | TTFT 基线(ms) | TTFT HD512-tuned(ms) | Δ TTFT | TPOT(ms) |
|-------|--------------|----------------------|--------|----------|
| 32768 | 21492 | **15265** | **−29.0%（−6.2s）** | 43.1（中性） |
| 65536 | 64966 | **40364** | **−37.9%（−24.6s）** | 49.4（中性） |

- **gsm8k_chat = 0.975（39/40）位级一致**，与基线相同。TPOT 中性（decode 非 prefill-FMHA 受限）。
- 收益随 input 增大而放大（FMHA O(n²) 在 TTFT 占比随 input 上升）。
- 落地：`FMHAPrefillXe20.cmake` L37-39 设 `TILED_Q_512=128/TILED_KV_512=64/NUM_SG_512=32`，
  `ninja sgl-ops-sycl-xe_fmha_fwd_prefill_kernel_512` → copy `.so` 至 `dist-packages/sgl_kernel/` + `python/sgl_kernel/`。
  备份 `FMHAPrefillXe20.cmake.bak_hd512tune`（原 256/64/32）。sliding HD256 核（独立 TU）不受影响。
- **后续可探**：mainloop K-block double-buffer prefetch（`xe_fmha_fwd_mainloop.hpp`，当前仅 V 预取）可能再进一步。


### 刷新的完整性能矩阵（HD512 FMHA tile-tuned + GEMV VL256 全生效，TP2 eager，bsz1，out=512，2026-07-03）

| input | TTFT (ms) | TPOT (ms) | decode tok/s | E2E (s) | TTFT vs FMHA调优前 |
|-------|-----------|-----------|--------------|---------|--------------------|
| 1024  |    347.1  |   37.66   | 26.6 | 19.59 | −2.7% (356.7) |
| 2048  |    682.6  |   37.26   | 26.8 | 19.72 | −4.1% (712.1) |
| 4096  |   1385.5  |   37.66   | 26.6 | 20.63 | −6.5% (1482.3) |
| 8192  |   2915.4  |   38.43   | 26.0 | 22.55 | −12.6% (3334.8) |
| 16384 |   6434.3  |   40.00   | 25.0 | 26.88 | **−19.7%** (8010.6) |
| 32768 |  15309.7  |   43.12   | 23.2 | 37.34 | **−28.8%** (21492.4) |
| 65536 |  40458.5  |   49.38   | 20.2 | 65.69 | **−38.2%** (64966.0) |

- TPOT 全程中性（与 baseline 一致，decode 非 prefill-FMHA 受限）。
- TTFT 收益随 input 单调放大（FMHA O(n²) 占 TTFT 比例随 input 上升）：短 prompt（1k-4k）由固定 GEMM/量化开销主导、收益有限；16k 起显著，64k 近 −38%。
- 现存 TTFT 大头：短 input 为 oneDNN/W8A16 GEMM，长 input 为 HD512 FMHA（现 10.1% peak，仍有 XMX 余量）+ allreduce。

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
- `src/FMHAPrefillXe20.cmake` — **HD512 full-attn tile TILED_Q 256→128（NUM_SG=32/TILED_KV=64）→ 2.3× kernel、TTFT −29~38%**（2026-07-03）
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
- `python/sglang/srt/layers/logits_processor.py` — esimd_gemv_fp16 for lm_head **已启用**（提交 `ac94971b3c`，
  移除 `and False` 守卫）。注：lm_head 未分片 2.82GB 已在内存墙，ESIMD 相对 oneDNN 仅 +0.13ms，启用无害但非 3ms 提升。
