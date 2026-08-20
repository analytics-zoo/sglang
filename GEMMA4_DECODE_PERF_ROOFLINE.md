# Gemma4-31B Decode Perf + Roofline (BMG TP=2, eager)

Config: **gemma-4-31B-it, TP=2, fp16 + ESIMD + intel_xpu + layered_fp8, split-K
`SGLANG_SPLITK_G=64` with in-kernel-chunk, ESIMD fp16 GEMV lm_head, XPU graph OFF
(eager)**. Container `txytest_sgl_bmg`, GPU 0,1, freq locked 2800 MHz. bsz=1.
Kernel breakdown measured 2026-07-01 (unitrace); E2E perf matrix + VL A/B
**refreshed 2026-07-02** on the pristine known-good build (gemm `.so` md5
`42e28795`, sglang `.so` md5 `3806e20f`, ESIMD lm_head active).

All decode-kernel numbers below come from a preserved unitrace capture (4k in /
64 out), analyzed over the **verified 64-step single-request decode window**
(gate_up GEMV = 3840/60 = 64.0 steps; allreduce = 7681/120 = 64.0; 0 intra-window
>0.5s gaps → one request). Trace kept at:
`/home/intel/xiangyu/cc_workspace/gemma_splitk/utrace_4k/python3.38621.json` (TP0,
690MB) + `python3.38622.json` (TP1). Analysis scripts: `/tmp/utrace_fam2.py`,
`/tmp/utrace_gemv2.py`.

---

## ⚠️ CORRECTION / ADDENDUM（2026-07-01 首版，**2026-07-02 clean-bench 复核后修订**）

本文原 §3.1 / §2 / §4 关于 **lm_head** 的分析基于一个**错误前提**（lm_head 被 TP 分片到 `V/TP`），
经对当前 build 的 fresh unitrace 逐内核复核，更正如下（原文其余部分——FP8 GEMV/attention/norm 等——已验证准确）：

- **lm_head 未分片**：gemma4 用 `Gemma3TextScaledWordEmbedding(nn.Embedding)`（未分片）+ `lm_head = embed_tokens`
  （tied）。每个 TP rank 都持有**完整 262144 vocab**（git 确认自模型引入即如此）。当前 ESIMD 内核实名
  `GEMV_fp16_kernel<256,1>[SIMD_ANY {262144;1;1}]` 直接坐实 N=**262144**，非 `V/TP=131072`。
- **正确字节数 = 2.82 GB/rank**（262144×5376×2），非本文他处所写的 1.41 GB。floor@630GB/s = **4.47 ms**。
- **lm_head 在内存墙上，且 oneDNN≈ESIMD**：oneDNN 4.78ms=590GB/s=**94%** roofline；ESIMD `GEMV_fp16` 4.65ms=606GB/s=**96%**。
  ESIMD 相对 oneDNN **只省 0.13 ms/step，不是 §3.1 声称的 ~3 ms**。§3.1 "slow variant 48% roofline / ESIMD 1.8ms / -3.9ms E2E"
  的因果链不成立（其 48% 只有在把字节当成 1.41GB 时才出现）。
- **~~"38.9 ms current 列不可复现，真实 TPOT 42-43ms"~~ —— 此更正本身有误，已被 2026-07-02 clean bench 推翻**：
  clean `bench_bsz1`（无 unitrace）在当前 build 上**可复现** ~39-40 ms：out=256 → 39.2 ms（1k-8k 平坦），
  out=64 → 40.3 ms。**从未测到 42-43 ms**。§2 的 42.9 ms 是 **unitrace 采集值**（instrumentation 抬高 host-gap/
  collective-wait），不是 clean wall-time。真实 non-compute 开销 = clean TPOT − device compute self-time
  (36.65) ≈ 40.3 − 36.65 = **~3.6 ms（out=64）/ ~2.6 ms（out=256）**，**远小于原文声称的 ~8.6 ms host-gap**。
  → device-busy 实际 ~91%（非 81%），XPU graph 可回收的 host-gap 上限约 ~3 ms，不是 8-9 ms。
- **但 lm_head 只省 0.13ms 的 roofline 结论正确** → 因此 §3.1/§4/§5 "ESIMD lm_head 省 ~3ms E2E" 是
  **归因错误**（oneDNN 与 ESIMD 应几乎等速；"41.9 oneDNN baseline" 一列是瞬态/异常测量，非内核差异）。
  clean bench 无论 oneDNN 还是 ESIMD lm_head 都应落在 ~39-40 ms。
- **唯一 roofline 支持的 lm_head 真实优化**：把 lm_head 计算按 TP 分片（VocabParallelEmbedding，131072/rank + all_gather）
  → 1.41GB/rank → ~2.24 ms → **省 ~2.4 ms/step → ~37 ms TPOT**（相对当前 clean ~39.2ms）。需新增 vocab-parallel lm_head 路径。

> 复核方法与逐内核对比表见 session `files/unitrace_compare.md`；trace 存于
> `cc_workspace/gemma_splitk/utrace_cur/python3.1804{0,1}.json`（当前 build，ESIMD lm_head）。

---

## 1. TTFT / TPOT vs sequence length (eager, verified via `bench_bsz1.py`)

### TPOT vs context (64 output tokens) — split-K G sweep

The eager TPOT-grows-with-ctx problem was root-caused to the 10 global (hd512)
split-K attention layers running with too few KV splits. `chunk = ctx/G`; cost ∝
chunk, so a fixed G grows with ctx. Fix = compute chunk from real seqlen inside
the kernel + high G. TPOT (ms/token):

| ctx  | G=4 (old default) | G=16 | G=64 | in-kernel-chunk G=64 | **current (ESIMD lm_head, out=64)** |
|------|-------------------|------|------|----------------------|-------------------------------|
| 1024 | 42.4              | 42.1 | —    | 41.9                 | **40.1** |
| 2048 | 42.6              | 42.6 | —    | —                    | — |
| 4096 | 46.1              | 42.7 | —    | 42.9                 | **40.5** |
| 8192 | 54.8              | 54.6 | 43.4 | 41.8                 | **40.2** |
| 16384| —                 | 46.0 | 43.3 | 42.6                 | — |
| 32768| —                 | 54.6 | 44.0 | 43.9                 | — |
| 65536| —                 | 71.6 | 50.1 | 50.1                 | — |

> ✅ **末列已用 clean `bench_bsz1` 于 2026-07-02 复现**（GPU 0,1，freq 2800，known-good VL512 build，
> warmup=1 trials=3）：out=64 → 40.1/40.5/40.2 ms @ 1k/4k/8k。**"~40ms（out=64）/ ~39ms（out=256）" 是可复现的真实
> clean wall-time**；顶部 CORRECTION 早前所写的 "42-43ms" 实为 §2 unitrace 采集值（instrumentation 抬高），
> 非 clean bench。原 "38.9" 与本次 40.1 相差 ~1.4ms，属 build/run 波动（同一 G=64 in-kernel-chunk 路径）。

#### 刷新的 clean-bench 性能矩阵（2026-07-02，known-good VL512，out=256，warmup=1 trials=3）

out=256 摊薄首 token 暖机，比 out=64 低 ~1ms，是 bsz=1 稳态 TPOT 的更好估计：

| input | TTFT (ms) | TPOT (ms) | decode tok/s | E2E (s) |
|-------|-----------|-----------|--------------|---------|
| 1024  | 621       | **39.23** | 25.5 | 10.63 |
| 2048  | 1276      | **39.25** | 25.5 | 11.28 |
| 4096  | 2644      | **39.22** | 25.5 | 12.64 |
| 8192  | 5721      | **39.27** | 25.5 | 15.73 |
| 16384 | 12805     | **40.99** | 24.4 | 23.26 |
| 32768 | 31100     | **44.16** | 22.6 | 42.36 |
| 65536 | 84143     | **50.45** | 19.8 | 97.01 |

TPOT 在 1k-8k **平坦 ~39.2ms**（跨 4 个长度 spread 仅 0.05ms，极稳定），16k 起因 10 层 global split-K
扫描全 KV 而上升（16k=41.0, 32k=44.2, 64k=50.5）。TTFT 与下表一致。

- G=4 grows from 8k; G=16 flat to ~16k then grows (32k=54.6 ≈ 8k@G4, same
  chunk=2048); G=64 flat to 32k. **in-kernel-chunk + G=64 is flat 1k–32k (~42–44ms)
  and 50ms@64k** with no short-ctx regression. 64k still rises ~6ms because even
  1024 work-items (16 q-heads × 64 splits ≈ BMG's ~960 HW threads) can't fully hide
  a 64k/64 = 1024-token serial scan per split.

### TTFT vs input length (chunked prefill @ 1024)

TTFT is prefill, dominated by the chunked prefill (input/1024 chunks). Roughly
linear-plus in input length (independent of the split-K G knob).

**⚠️ 下表为旧 baseline（W8A8 per-token，opt#1/opt#2 前）。当前默认路径 = W8A16（opt#2），
TTFT 见下面 "#### opt#2" 小节的完整矩阵（1k/4k/8k 分别 −47%/−44%/−41%）。**

| input | TTFT (ms) 旧 baseline |
|-------|-----------|
| 1024  | ~623 |
| 4096  | ~2640 |
| 8192  | ~5700 |
| 16384 | ~12800 |
| 32768 | ~31100 |
| 65536 | ~84300 |

(64k prefill = 84s is the dominant wall cost of a 64k request; TPOT is the
per-token decode number after prefill. KV pool holds 181760 tokens so 64k fits.)

#### opt#1: prefill FP8 未融合反量化 → 融合 `_scaled_mm` epilogue（2026-07-02，已实现+验证）

激活改 per-tensor 量化 → 命中融合 `torch._scaled_mm` epilogue，消除未融合 fp32 反量化
（`ElementwiseGlobalRangeKernel<float,3>`，占 prefill ~11.5% self-time）。默认 ON，
`SGLANG_XPU_FP8_PERTENSOR_PREFILL=0` 回退。精度中性（gsm8k chat eval 0.975 = baseline）。

TTFT A/B（同一 session 干净测量，bsz1 warmup2 trials3，output256），并列 target：

| input | OFF TTFT | ON TTFT (opt#1) | Δ | **target** |
|-------|----------|-----------------|-----|-----------|
| 1024  | 621.5ms  | **526.3ms** | −15.3% | — |
| 4096  | 2641.0ms | **2255.8ms** | −14.6% | **1500ms** |
| 8192  | 5689.5ms | **4919.6ms** | −13.5% | **3500ms** |

TPOT 不变（~40ms，decode 走 ESIMD M≤64 快路早退，opt#1 不触及）。
trace 确认（干净单次 4096 prefill，4 chunk）：未融合反量化 `ElementwiseGlobalRange<float,3>`
299.2ms(11.5%) → **0**；Triton `per_token_group_quant` 34.8ms → **0**；fp32 中间 elem/copy/cast
142.4ms → 4.3ms；新增 `PerTensorQuantFP8Kernel`(+AbsMax) 76.6ms；prefill self-time
2597.0ms → 2275.2ms（**−12.4%**），与 TTFT A/B 一致。

**⚠️ opt#1 仍未达 target**（已被 opt#2 取代，见下）：opt#1 拿下 17% 的反量化项，但 4k/8k 距 target（1500/3500ms）还差
~750/~1420ms。

#### opt#2: W8A16 prefill（激活不量化）—— **当前默认路径，取代 opt#1**（2026-07-02）

prefill 保持激活 fp16、跑 mixed fp16×fp8 oneDNN matmul（`fp8_gemm_w8a16`），去掉整条激活量化+反量化链，
且命中一个更快的 oneDNN GEMM primitive。gsm8k 0.975（精度中性），**首次命中 target**。默认 ON，
`SGLANG_XPU_FP8_W8A16_PREFILL=0` 回退 opt#1。

W8A16 默认完整 TTFT/TPOT 矩阵（TP2 eager，bsz1，output=256）：

| input | TTFT (ms) | TPOT (ms) | decode tok/s | E2E (s) | vs 旧 baseline |
|-------|-----------|-----------|--------------|---------|----------------|
| 1024  | 329.6 | 43.6 | 23.0 | 11.44 | −47% |
| 2048  | 694.1 | 43.4 | 23.0 | 11.76 | −46% |
| 4096  | 1473.8 | 43.0 | 23.3 | 12.43 | **−44% ✅<1500** |
| 8192  | 3354.3 | 43.8 | 22.9 | 14.51 | **−41% ✅<3500** |
| 16384 | 8089.5 | 43.3 | 23.1 | 19.13 | −37% |
| 32768 | 21719.2 | 44.8 | 22.3 | 33.15 | −30% |
| 65536 | 65324.7 | 50.4 | 19.8 | 78.19 | −22% |

trace 归因（vs opt#1 同 prefill 窗口）：激活量化链 `PerTensorQuant`+`AbsMax` 230ms→0；oneDNN GEMM
4576ms→2449ms（1.87×，`torch._scaled_mm` tile{128;4;1} → `fp8_gemm_w8a16` tile{64;8;1}）；prefill self −28.7%。
⚠️ follow-up：in-server W8A16 GEMM 144 TF/s vs standalone microbench 69 TF/s（2× gap），
TTFT 大胜同时来自去量化 **和** 更快的 GEMM dispatch，归因待厘清。
（详见 `GEMMA4_BMG_OPTIMIZATION_STATUS.md` 的 opt#2 节。）

**TPOT 43 vs 旧 39ms —— 已实测定因，非本优化引入**：同 server 运行时 A/B（无 unitrace，warmup2/trials3）
W8A16 ON = 43.3/42.5ms（1k/4k）、OFF(opt#1) = 43.4/44.3ms → **W8A16 对 TPOT 中性**（ON≈OFF，OFF 略高）。
代码上 decode（M=1）走 `apply_fp8_linear` 的 ESIMD 早退（M≤64）即 return，走不到 W8A16 分支（>64）；trace
里 `GEMV_fp8_pert <256>/<512>` 逐 step self-time 三配置逐字节相同。**39→43 的绝对漂移 = 共享节点 host-CPU
争用**：旧 39ms 测于 07-01 05:30（节点空闲），今日 load≈5、`ps` 见多个他人 sglang::scheduler 钉 ~100% CPU；
bsz=1 decode 的 host 侧逐 kernel 派发被 CPU 争用均匀抬高 ~+3~4ms（各 ctx 平坦），ON/OFF 同等受影响。
教训：unitrace 的 allreduce/timing 会被放大、不可据以下判断，结论以运行时 bench 为准。



### Correctness (eager, unaffected by the G knob)
gsm8k chat eval (n=40, chat-templated, `#### N` extraction): G=4 = 0.950 (38/40),
G=16 = 0.975 (39/40), 0% invalid. bf16 baseline = 0.990. Coherent output at all G.

---

## 2. Decode kernel breakdown (4k ctx, verified 64-step window)

> ⚠️ **2026-07-02 更正（host-gap 被 unitrace 抬高）**：本节 "TPOT 42.9 ms" 是 **unitrace 采集下的 wall-time**，
> 不是 clean 值。2026-07-02 clean `bench_bsz1`（同 build，无 trace）实测 **~40.3 ms（out=64）/ ~39.2 ms（out=256）**。
> device compute self-time 36.65 ms 是 HW 时间戳、**准确**；因此真实 non-compute 残余 = 40.3 − 36.65 ≈ **~3.6 ms**
> （out=256 时 ~2.6 ms），**而非下面表/文所写的 ~6.2 ms 或 8.6 ms**。→ 真实 **device-busy ≈ 91%（非 81%）**，
> XPU graph 能回收的 host-gap 上限约 ~3 ms。下表的 42.9/6.2/%TPOT 均按 trace wall 计，仅作 trace 快照参考。

Device self-time = GPU-hardware-timestamped = **accurate for compute kernels**.
unitrace inflates wall/host-gaps and **collective sync-wait** — the oneCCL
allreduce self-time is UNRELIABLE (varied 36.7% → 9.8% between two captures of the
same config), so it is tagged and EXCLUDED from roofline / conclusions (per the
skill's known-issue note + observed variance).

Sanity: raw compute self-time (excl. allreduce) sums to 36.65 ms/step < real
TPOT 46 ms → raw compute times are physically consistent.

Columns: `ms/step` = raw device self-time (accurate); `%TPOT` = share of the real
42.9 ms/step TPOT; `bytes/step` = DRAM traffic (weight/KV = read; small = rw);
`floor` = bytes ÷ measured BW (630 read / 640 copy-rw); `roofline%` = floor/ms;
`bound` = what limits it.

| kernel (disaggregated)              | ms/step | %TPOT | calls/step | µs/call | bytes/step | floor ms | roofline% | bound |
|-------------------------------------|---------|-------|------------|---------|------------|----------|-----------|-------|
| **FP8 GEMV** (4 modules, see §3)    | 25.44   | 59.3  | 240        | 106     | 14.70 GB   | 23.33    | **92%**   | **MEM-read (wall)** |
| **ESIMD fp16 GEMV lm_head** ⚠️corrected | **4.65** | 10.8 | 1 | **4655** | **2.82 GB** | 4.47 | **96%** | **MEM-read (wall)** |
| oneCCL allreduce `[UNRELIABLE]`     | (3.98)  | (9.3) | 120        | —       | comm       | —        | — (n/a)   | comm (PCIe); not roofline'd |
| attention: page_attn sliding (3-ph) | 1.49    | 3.5   | 150        | ~10     | 419 MB     | 0.67     | 45%       | MEM-read (small abs) |
| norm: FusedNorm (RMSNorm)           | 1.16    | 2.7   | 201        | 5.7     | 6.5 MB     | 0.010    | **0.9%**  | **launch/latency** |
| attention: split-K global (2-ph)    | 1.02    | 2.4   | 22         | ~50     | 168 MB     | 0.27     | 26%       | MEM-read (small abs) |
| norm: RmsNormResidualScalar         | 0.78    | 1.8   | 60         | 12.9    | 1.9 MB     | 0.003    | **0.4%**  | **launch/latency** |
| norm: FusedAddRmsNorm               | 0.57    | 1.3   | 60         | 9.6     | 1.9 MB     | 0.003    | **0.5%**  | **launch/latency** |
| act: silu/gelu-mul                  | 0.50    | 1.2   | 60         | 8.3     | 3.9 MB     | 0.006    | **1.2%**  | **launch/latency** |
| SYCL index/scatter                  | 0.49    | 1.1   | 135        | 3.6     | 2.2 MB     | 0.003    | **0.7%**  | **launch/latency** |
| SYCL elem/copy/cast                 | 0.38    | 0.9   | 235        | 1.6     | 5.1 MB     | 0.008    | **2.1%**  | **launch/latency** |
| ESIMD qkv_split_norm_rope           | 0.08    | 0.2   | 50         | 1.7     | 1.6 MB     | 0.003    | 3.2%      | launch/latency |
| RoPE                                | 0.03    | 0.1   | 10         | 3.3     | 0.3 MB     | 0.001    | 1.7%      | launch/latency |
| **compute subtotal (excl allreduce)** | **36.65** | 85.4 | —        | —       | 16.73 GB   | **26.6** | **73%**   | — |
| **+ real allreduce + host-gap**    | ~6.2    | 14.4  | —          | —       | —          | —        | —         | comm + idle |
| **= measured TPOT @4k**             | **42.9**| 100   | —          | —       | —          | —        | **62% (E2E)** | — |

Reading the two roofline regimes:
- **MEM-read rows** (FP8 GEMV, lm_head, attention): roofline% is meaningful.
  FP8 GEMV 92% = at the bandwidth wall. lm_head 47% and split-K 26% have headroom
  but lm_head's is the only one worth chasing (4.8 ms abs vs attention's 2.5 ms).
- **launch/latency rows** (all norms, act, index, elem, rope, qkv_norm_rope):
  roofline% is **0.4–3%** — they move almost no data; their ms/step is host-launch +
  kernel-invocation latency of many µs-scale kernels (201+60+60+135+235… calls/step),
  NOT bandwidth. Their combined ~3.9 ms/step is recovered by **kernel fusion or a
  working XPU graph** (host-gap reclaim), not by any memory optimization.

(compute subtotal 36.65 ms at 73% of its 26.6 ms memory floor; the E2E 62% in §4
is lower because it also carries the comm + host-gap residual.)

Grouped roll-ups: **FP8 GEMV 25.44 ms**, lm_head 4.65 ms, [allreduce ~unreliable],
**norm TOTAL (3 kernels) 2.51 ms/step**, **attention TOTAL 2.50 ms/step**, act/
index/elem/rope the rest. Device-busy ≈ 81% in the *trace* (≈8.6 ms/step trace host-gap)
— but per the 2026-07-02 note above, **clean device-busy ≈ 91%** (real host-gap ~3 ms);
the 8.6 ms is unitrace-inflated. What a working XPU graph could reclaim is ~3 ms, not
8.6 ms (graph currently deadlocks, see `GEMMA4_XPU_GRAPH_STATUS.md`).

Over-packing note: an earlier pass lumped all norms into one "norm" bucket and all
FP8 GEMV into one row. Disaggregated here: norm = 3 distinct kernels; FP8 GEMV = 4
(§3); attention = page_attn(3 phases)+split-K(2 phases).

---

## 3. FP8 GEMV per-module roofline

Measured memory bandwidth on THIS BMG (`bw_probe2.cpp`, 512MB working-set ≫ L2,
min-of-30 device-event): **READ-only 630 GB/s**, COPY(r+w) 1283 GB/s, WRITE-only
1008 GB/s. (Device-reported "64-bit bus" is misreported — ignored. Roofline uses
the measured 630 GB/s read ceiling, not any spec.)

FP8 GEMV is weight-streaming (reads the fp8 weight, 1 byte/elem, once per step).
The single "FP8 GEMV" row is actually **4 distinct kernel instantiations** by shape
`{1; N_out; 1}`, mapped to modules via calls/step (= layers × per-layer count):

| module (kernel)                       | shape N | calls/step | wt GB/step | ms/step | achieved | **% of 630 GB/s roofline** |
|---------------------------------------|---------|------------|------------|---------|----------|-----------------------------|
| gate_up_proj `<256>`                  | 21504   | 60         | 6.94       | 11.78   | 589 GB/s | **94%** |
| o_proj + down_proj `<512>` (3 shapes†)| 5376    | 120        | 5.01       | 8.87    | 565 GB/s | **90%** |
| qkv_proj sliding `<256>`              | 8192    | 50         | 2.20       | 3.84    | 573 GB/s | **91%** |
| qkv_proj global `<256>`               | 10240   | 10         | 0.55       | 0.96    | 576 GB/s | **91%** |
| **FP8 GEMV total**                    | —       | 240        | 14.70      | 25.44   | 578 GB/s | **92%** |

- `N=21504 = 2·I/TP` → gate_up (60/step = 60 layers). `N=8192` = sliding qkv out
  `(nH·hd+2·nkv·hd)/TP` (50 layers). `N=10240` = global qkv out (10 layers).
- **† `<512>` 组不是 "o_proj(60)+down_proj(60) 同形状"，而是 3 个不同的 (N_out, K) row-projection**（用
  `select_vl_ks` 的 `[VLKS]` 打印 + safetensors 头实测确认；trace 只按 N_out=5376=H 归组，才看似一个）。
  gemma4 经 `layer_types` 有两种注意力尺寸（50 sliding : 10 full-attn，5:1），o_proj 输入维随之不同：

  | GEMV shape (N_out, K/rank) | 模块              | 层数 | VL  | wt/step |
  |----------------------------|-------------------|------|-----|---------|
  | (5376, 4096)               | o_proj — sliding  | 50   | 512 | 1.10 GB |
  | (5376, 8192)               | o_proj — full-attn| 10   | 512 | 0.44 GB |
  | (5376, 10752)              | down_proj — all   | 60   | 512 | 3.47 GB |

  o_proj 合计 1.10+0.44 = **1.54 GB**（原文 1.54 已隐含两种 K，会计正确，仅措辞把 60 层当成同 K 有误）；
  down_proj 3.47 GB；组合 5.01 GB / 120 calls，与聚合行一致。full-attn 层 q=64/kv=16 heads → o_proj 输入
  16384 → per-rank K=8192；sliding 层 q=32/kv=8 → 输入 8192 → per-rank K=4096（safetensors: o_proj
  `[(5376,8192),(5376,16384)]`）。
- **All 4 modules sit at 90–94% of the measured read roofline** → uniformly
  memory-wall-bound; no single module is an outlier with headroom. FP8 GEMV cannot
  get materially faster on this DRAM system.

### §3.1 lm_head: oneDNN slow-variant → ESIMD fp16 GEMV (**FIXED 2026-07-01**)

> ⚠️ **见顶部 CORRECTION**：本小节前提（lm_head=1.41GB/TP-sharded、oneDNN 48% roofline、ESIMD ~1.8ms、
> -3.9ms E2E）**已被 unitrace 复核推翻**。实际 lm_head 未分片 = 2.82GB/rank，oneDNN(94%)≈ESIMD(96%) 均在内存墙，
> ESIMD 仅省 0.13ms。以下原文保留作记录。

**Root cause found**: oneDNN selects a slow kernel variant `{32,2,8}` for steady-state
decode (4.69 ms/call in unitrace), while the first call and standalone benchmarks get
a fast variant `{128,4,1}` (~1.5 ms). This is NOT a bandwidth problem — the slow
variant achieves only 300 GB/s (48% roofline) due to oneDNN's kernel selection, not
memory contention.

**Fix**: enabled ESIMD fp16 GEMV (`logits_processor.py:914`, removed `and False`
guard). Same 1.41 GB fp16 weight read, but bypasses oneDNN entirely.

**E2E result** (bench_bsz1, bsz=1, warmup=2, trials=3, freq 2800):

| ctx  | before (oneDNN) | after (ESIMD fp16 GEMV) | delta |
|------|-----------------|-------------------------|-------|
| 1024 | 41.9            | **38.9**                | −3.0  |
| 4096 | 42.9            | **39.0**                | −3.9  |
| 8192 | 41.8            | **38.9**                | −2.9  |

Correctness: gsm8k 0.975 (39/40) = baseline. No extra memory (model is fp16,
lm_head weight already fp16, `.to(fp16).contiguous()` is no-op).

### §3.2 VL 512→256 调优实验（o_proj + down_proj）：kernel 微测提速 **未传导到 E2E**（2026-07-02）

> ⚠️ **2026-07-03 已被推翻——VL256 现为默认。** 本节的 "VL256 回退 +2-3%" 系 **rebuild-confound
> + 节点争用噪声**（旧 A/B 两侧是不同 binary、非同时刻）。用 **同一 binary + env-gate**（`SGLANG_GEMV_VL_CAP`）
> 复测后 VL256 实为中性偏优（−0.06ms），并经 unitrace 逐 shape 坐实（down_proj in-server 91→93%）。
> **现默认 VL256**（`SGLANG_GEMV_VL_CAP` 默认 256）。详见 `GEMMA4_BMG_OPTIMIZATION_STATUS.md`
> "GEMV tile 复调优：VL512→256（采纳）"。以下原文保留作记录。

**正确的 decode 内核位置**（此前一度找错到 `custom-esimd-kernels-sglang` 的 `fp8_GEMV_v2.h`，那是 decode 死代码）：
- 包：`/workspace/custom-esimd-kernels/`（`python_v2`），op `torch.ops.custom_esimd_kernels.esimd_gemm_fp8_pert`。
- VL 由 `csrc/xpu/esimd_kernels/fp8_GEMM_pert.h::select_vl_ks`（~line 75）决定 →
  `GEMV_fp8_pert_batched_kernel<VL,KS>`（launch ~line 3800）。
- `select_vl_ks` 逻辑：default vl=512,ks=1；K<512→128；K==512→256；N≤128&K≥2048→128,ks8；
  N≤512&K≥2048→128,ks4；再按整除性 halve vl 至 `kpt%vl==0`。gemma4 得到：gate_up/qkv → 256，
  上面 3 个 `<512>` shape（o_proj 4096/8192、down_proj 10752）→ **512**。

**实验**：把上述 3 个 shape 的 `select_vl_ks` 默认 512 改成 256，重建 gemm 扩展，同一 server 只切内核做 A/B：

| ctx  | VL512 (known-good) | VL256 | delta |
|------|--------------------|-------|-------|
| 1024 | **39.03**          | 40.19 | +3.0% (+1.16ms) |
| 8192 | **39.51**          | 40.39 | +2.2% (+0.88ms) |

- **VL256 让 E2E decode 回退 ~2-3%（~1ms TPOT）**，正确性不变（gsm8k 0.975）。
- 之前 kernel 微测得到的 "+6-9% faster @ 256" **在 bsz=1 E2E 下没有兑现**——孤立 GEMV microtest 的收益被
  端到端的 host-dispatch / 内存墙 / 其它内核淹没。§3 已证明这 3 个 shape 都在 90% roofline，本来就在墙上。
- **结论：保留 VL512，不要改 256**。容器已还原到 pristine known-good（gemm md5 `42e28795`）。
- 数据佐证 §3 的判断：FP8 GEMV 已到带宽墙，VL 微调无法再压 E2E。

### Roofline of the other BW-relevant rows
- **lm_head** (fp16 weight, **未分片 = `[V, H]` = 2.82 GB/rank read**): **~4.65 ms/step** via ESIMD
  fp16 GEMV = 606 GB/s = **96% roofline**，≈ oneDNN(4.78ms/94%)。在内存墙，换核仅省 0.13ms（见顶部
  CORRECTION；原文 "1.41 GB / 1.8 ms / FIXED" 已作废）。真正的 lm_head 优化 = vocab-parallel 分片（§5）。
- **attention** (KV read, 0.59 GB): 2.50 ms/step → 234 GB/s = 37% roofline, but only
  2.5 ms absolute — not worth optimizing.
- **norm / act / index / elem** (small activation r/w, few MB): floor at 630/640 GB/s
  is ~0.01 ms — these are **LAUNCH/LATENCY-bound, not bandwidth-bound** (they use
  0.4–1.2% of their BW ceiling). Their ~3–4 ms/step is host-launch overhead of many
  tiny kernels → fuse them or reclaim via XPU graph, NOT a bandwidth problem.

---

## 4. E2E: measured TPOT vs memory roofline (4k ctx, current config)

Bytes that MUST be read from DRAM per decode step (weight-stream + KV):

| source                    | GB/step |
|---------------------------|---------|
| FP8 GEMV weights (fp8)    | 14.70   |
| lm_head weight (fp16) ⚠️  | 2.82    |
| KV cache (attention read) | 0.59    |
| activations (norm/act rw) | ~0.03   |
| **total**                 | **18.14** |

**Memory-roofline E2E floor** = 18.14 GB ÷ 630 GB/s (measured read ceiling) =
**28.8 ms/step**. （⚠️ lm_head 更正为未分片 2.82GB，见顶部 CORRECTION；原文误作 1.41GB→16.73GB→26.6ms。）

**Measured TPOT (clean `bench_bsz1`, current: in-kernel-chunk G=64, eager, ESIMD lm_head)**
= **~39.2 ms/step (out=256) / ~40.3 ms (out=64)**，1k–8k 平坦（2026-07-02 复现）。
（原 "was 42.9 before lm_head fix" 已删：42.9 是 unitrace wall；lm_head 换核只省 0.13ms，不构成 3ms 提升。）

→ **E2E efficiency = 28.8 / 39.2 = 73% of the memory roofline**（floor 用更正后的 18.14 GB）。

Caveat: the 28.8 ms floor is a *pure memory-bound ideal* — it assumes zero TP
communication and perfect kernel/host overlap. Decode also carries an irreducible
**TP allreduce (communication-bound, NOT memory-bound)** cost and host-launch
overhead. Gap decomposition (clean TPOT ~39.2 ms; buckets = §2 device self-time + clean 残余):

| bucket                              | ms/step | note |
|-------------------------------------|---------|------|
| FP8 GEMV                            | 25.4    | 92% roofline — at the wall（VL 微调无效，见 §3.2） |
| lm_head (ESIMD fp16 GEMV)           | 4.65    | 内存墙 96%；≈oneDNN（换核仅省 0.13ms，见 CORRECTION） |
| attention (page_attn+split-K)       |  2.5    | small; 37% roofline |
| norm ×3 + act + index + elem + rope | ~4.0    | LAUNCH-bound（µs 内核 device self-time），靠 fusion / graph |
| TP allreduce(comm) + host-gap       | ~2.6    | clean 残余 = 39.2 − 36.65 compute；graph 只能回收其中 host-gap 部分 |

The **realistically recoverable** portion is the ~2.6 ms comm/host-gap residual + 部分
launch-bound 内核（kernel fusion），合计 **~3–4 ms**（需可用 XPU graph / fusion）。lm_head
不是可优化项（已在内存墙）。well-optimized decode 现实上限 ≈ **~35–36 ms/step（≈80% roofline）**，
不是原文乐观的 30–32ms；其中 ~28.8 ms（FP8 GEMV + lm_head + weight/KV 读）是本 DRAM 的硬带宽墙。

## 5. Optimization levers, by ROI

1. **TP allreduce** — clean 残余（allreduce comm + host-gap）合计仅 ~2.6 ms/step（39.2 − 36.65 compute）；
   unitrace 的 "4–7 ms" 是 instrumentation 抬高值。Rides **PCIe not XeLink** (`oneccl_allreduce_pcie`).
   杠杆：更快互联，或 fuse allreduce+RMSNorm 减少次数。
2. **Host-gap + launch-bound norms** (~3 ms clean host-gap + µs-级 norm launches) — 可用 XPU graph 回收其中
   host-gap 部分（~3 ms 上限，非原文 8.6 ms）。当前被 graph decode deadlock 阻塞（`GEMMA4_XPU_GRAPH_STATUS.md`）。
3. ~~**lm_head 48% roofline → ESIMD, −3 ms**~~ **此项已作废**：lm_head 未分片 = 2.82 GB，已在内存墙 96%，
   ESIMD vs oneDNN 仅差 0.13 ms（见顶部 CORRECTION）。**唯一 roofline 支持的 lm_head 优化 = vocab-parallel 分片**
   （131072/rank + all_gather → 1.41 GB/rank → ~2.24 ms → 省 ~2.4 ms/step → ~37 ms TPOT），需新增 vocab-parallel 路径。
4. **FP8 GEMV (92% roofline) 与 attention 已到/接近极限 — 不要优化。** split-K G 已调好（§1）；
   FP8 GEMV 的 VL 也已验证 512 最优，改 256 反退 2–3%（§3.2）。

## Repro
- Launch (unitrace): `/home/intel/xiangyu/cc_workspace/gemma_splitk/launch_unitrace_current.sh`
- Launch (bench): `probe.sh` with `USE_GRAPH=0 SPLITK_G=64` + `bench_bsz1.py`
- BW probe: `bw_probe2.cpp`; kernel-level split-K microbench: `hd512_decode/bug_repro.cpp`
- Decode-window analysis: `/tmp/utrace_fam2.py <bigjson>` (isolates via prefill-only
  `_per_token_group_quant_8bit` last-cluster boundary).
