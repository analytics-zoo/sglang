# Gemma4-31B Decode Perf + Roofline (BMG TP=2, eager)

Config: **gemma-4-31B-it, TP=2, fp16 + ESIMD + intel_xpu + layered_fp8, split-K
`SGLANG_SPLITK_G=64` with in-kernel-chunk, XPU graph OFF (eager)**. Container
`txytest_sgl_bmg`, GPU 0,1, freq locked 2800 MHz. bsz=1. Measured 2026-07-01.

All decode-kernel numbers below come from a preserved unitrace capture (4k in /
64 out), analyzed over the **verified 64-step single-request decode window**
(gate_up GEMV = 3840/60 = 64.0 steps; allreduce = 7681/120 = 64.0; 0 intra-window
>0.5s gaps → one request). Trace kept at:
`/home/intel/xiangyu/cc_workspace/gemma_splitk/utrace_4k/python3.38621.json` (TP0,
690MB) + `python3.38622.json` (TP1). Analysis scripts: `/tmp/utrace_fam2.py`,
`/tmp/utrace_gemv2.py`.

---

## 1. TTFT / TPOT vs sequence length (eager, verified via `bench_bsz1.py`)

### TPOT vs context (64 output tokens) — split-K G sweep

The eager TPOT-grows-with-ctx problem was root-caused to the 10 global (hd512)
split-K attention layers running with too few KV splits. `chunk = ctx/G`; cost ∝
chunk, so a fixed G grows with ctx. Fix = compute chunk from real seqlen inside
the kernel + high G. TPOT (ms/token):

| ctx  | G=4 (old default) | G=16 | G=64 | **in-kernel-chunk G=64 (current)** |
|------|-------------------|------|------|-------------------------------------|
| 1024 | 42.4              | 42.1 | —    | **41.9** |
| 2048 | 42.6              | 42.6 | —    | — |
| 4096 | 46.1              | 42.7 | —    | **42.9** |
| 8192 | 54.8              | 54.6 | 43.4 | **41.8** |
| 16384| —                 | 46.0 | 43.3 | **42.6** |
| 32768| —                 | 54.6 | 44.0 | **43.9** |
| 65536| —                 | 71.6 | 50.1 | **50.1** |

- G=4 grows from 8k; G=16 flat to ~16k then grows (32k=54.6 ≈ 8k@G4, same
  chunk=2048); G=64 flat to 32k. **in-kernel-chunk + G=64 is flat 1k–32k (~42–44ms)
  and 50ms@64k** with no short-ctx regression. 64k still rises ~6ms because even
  1024 work-items (16 q-heads × 64 splits ≈ BMG's ~960 HW threads) can't fully hide
  a 64k/64 = 1024-token serial scan per split.

### TTFT vs input length (chunked prefill @ 1024)

TTFT is prefill, dominated by the chunked prefill (input/1024 chunks). Roughly
linear-plus in input length (independent of the split-K G knob):

| input | TTFT (ms) |
|-------|-----------|
| 1024  | ~623 |
| 4096  | ~2640 |
| 8192  | ~5700 |
| 16384 | ~12800 |
| 32768 | ~31100 |
| 65536 | ~84300 |

(64k prefill = 84s is the dominant wall cost of a 64k request; TPOT is the
per-token decode number after prefill. KV pool holds 181760 tokens so 64k fits.)

### Correctness (eager, unaffected by the G knob)
gsm8k chat eval (n=40, chat-templated, `#### N` extraction): G=4 = 0.950 (38/40),
G=16 = 0.975 (39/40), 0% invalid. bf16 baseline = 0.990. Coherent output at all G.

---

## 2. Decode kernel breakdown (4k ctx, verified 64-step window)

Device self-time = GPU-hardware-timestamped = **accurate for compute kernels**.
unitrace inflates wall/host-gaps and **collective sync-wait** — the oneCCL
allreduce self-time is UNRELIABLE (varied 36.7% → 9.8% between two captures of the
same config), so it is tagged and EXCLUDED from roofline / conclusions (per the
skill's known-issue note + observed variance).

Sanity: raw compute self-time (excl. allreduce) sums to 36.65 ms/step < real
TPOT 46 ms → raw compute times are physically consistent.

| kernel (disaggregated)              | ms/step | %busy | calls/step | µs/call |
|-------------------------------------|---------|-------|------------|---------|
| **FP8 GEMV** (4 modules, see §3)    | 25.44   | 62.4  | 240        | 106     |
| oneDNN GEMM lm_head                 |  4.78   | 11.7  | 1          | 4639    |
| oneCCL allreduce `[UNRELIABLE]`     | (3.98)  | (9.8) | 120        | —       |
| norm: FusedNorm (RMSNorm)           |  1.16   | 2.8   | 201        | 5.7     |
| attention: page_attn sliding (3-phase) | 1.49 | 4.0   | 150        | ~10     |
| attention: split-K global (2-phase) | 1.02    | 2.5   | 22         | ~50     |
| norm: RmsNormResidualScalar         |  0.78   | 1.9   | 60         | 12.9    |
| norm: FusedAddRmsNorm               |  0.57   | 1.4   | 60         | 9.6     |
| act: silu/gelu-mul                  |  0.50   | 1.2   | 60         | 8.3     |
| SYCL index/scatter                  |  0.49   | 1.2   | 135        | 3.6     |
| SYCL elem/copy/cast                 |  0.38   | 0.9   | 235        | 1.6     |
| ESIMD qkv_split_norm_rope           |  0.08   | 0.2   | 50         | 1.7     |
| RoPE                                |  0.03   | 0.1   | 10         | 3.3     |

Grouped: **FP8 GEMV 62%**, lm_head 12%, [allreduce ~unreliable], **norm TOTAL
(3 kernels) 2.51 ms/step (6.1%)**, **attention TOTAL 2.50 ms/step (6.5%)**, act/
index/elem/rope the rest. Device-busy ≈ 81% (≈8.6 ms/step host-gap that a working
XPU graph would reclaim — but graph currently deadlocks, see
`GEMMA4_XPU_GRAPH_STATUS.md`).

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
| o_proj + down_proj `<512>`            | 5376    | 120        | 5.01       | 8.87    | 565 GB/s | **90%** |
| qkv_proj sliding `<256>`              | 8192    | 50         | 2.20       | 3.84    | 573 GB/s | **91%** |
| qkv_proj global `<256>`               | 10240   | 10         | 0.55       | 0.96    | 576 GB/s | **91%** |
| **FP8 GEMV total**                    | —       | 240        | 14.70      | 25.44   | 578 GB/s | **92%** |

- `N=21504 = 2·I/TP` → gate_up (60/step = 60 layers). `N=8192` = sliding qkv out
  `(nH·hd+2·nkv·hd)/TP` (50 layers). `N=10240` = global qkv out (10 layers).
  `N=5376 = H` = o_proj (60) + down_proj (60) — same shape, can't split in trace;
  weight bytes computable: o_proj 1.54 GB + down_proj 3.47 GB per step.
- **All 4 modules sit at 90–94% of the measured read roofline** → uniformly
  memory-wall-bound; no single module is an outlier with headroom. FP8 GEMV cannot
  get materially faster on this DRAM system.

### Roofline of the other BW-relevant rows
- **lm_head** (bf16 weight `[H, V/TP]` = 1.41 GB read): 4.78 ms/step → **300 GB/s =
  48% of read roofline** → the ONE row with real bandwidth headroom. Levers: route
  M=1 lm_head through an fp16/fp8 GEMV (vLLM does this) → ~halve to ~2.2 ms/step.
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
| lm_head weight (bf16)     | 1.41    |
| KV cache (attention read) | 0.59    |
| activations (norm/act rw) | ~0.03   |
| **total**                 | **16.73** |

**Memory-roofline E2E floor** = 16.73 GB ÷ 630 GB/s (measured read ceiling) =
**26.6 ms/step**.

**Measured TPOT @4k (current: in-kernel-chunk, G=64, eager)** = **42.9 ms/step**.

→ **E2E efficiency = 26.6 / 42.9 = 62% of the memory roofline.**

Caveat: the 26.6 ms floor is a *pure memory-bound ideal* — it assumes zero TP
communication and perfect kernel/host overlap. Decode also carries an irreducible
**TP allreduce (communication-bound, NOT memory-bound)** cost and host-launch
overhead, so 62% is expected, not a defect. Gap decomposition (accurate compute
device self-times; TPOT 42.9 ms):

| bucket                         | ms/step | note |
|--------------------------------|---------|------|
| FP8 GEMV                       | 25.4    | 92% roofline — at the wall |
| lm_head                        |  4.8    | 48% roofline — recoverable ~2.5 ms |
| attention (page_attn+split-K)  |  2.5    | small; 37% roofline |
| norm ×3 + act + index + elem   |  ~4.5   | LAUNCH-bound (µs kernels), not BW — reclaim via fusion / XPU graph |
| TP allreduce (comm) + host-gap |  ~5.7   | comm-bound + idle; graph would reclaim the host-gap part |

So of the 42.9 − 26.6 = **16.3 ms/step above the memory floor**, roughly: ~2.1 ms is
FP8 GEMV's own 8% inefficiency (near-irreducible), ~2.5 ms lm_head, ~4.5 ms
launch-bound small kernels, ~1.6 ms attention, and ~5.7 ms TP-comm + host-gap. The
**realistically recoverable** portion is the launch-bound kernels + host-gap
(~8–9 ms, needs working XPU graph / kernel fusion) and lm_head (~2.5 ms) — i.e. a
well-optimized decode could approach ~32–34 ms/step (≈78–83% of roofline); the last
~26.6 ms (FP8 GEMV + weight/KV reads) is a hard memory-bandwidth wall on this DRAM.

## 5. Optimization levers, by ROI

1. **TP allreduce** — real cost ~4–7 ms/step (unitrace-inflated, not roofline'd
   here). Rides **PCIe not XeLink** (`oneccl_allreduce_pcie`). Algorithmic/
   interconnect lever: faster interconnect, or fuse allreduce+RMSNorm to cut count.
2. **Host-gap + launch-bound norms** (~8.6 ms host-gap + ~2.5 ms tiny-norm launches)
   — a working XPU graph reclaims most of this. Currently blocked by the graph
   decode deadlock (`GEMMA4_XPU_GRAPH_STATUS.md`).
3. **lm_head 48% roofline** → fp16/fp8 M=1 GEMV, ~2 ms/step.
4. **FP8 GEMV (62%, 92% roofline) and attention (6.5%) are at/near their limits — do
   NOT optimize.** split-K G is already tuned (§1).

## Repro
- Launch (unitrace): `/home/intel/xiangyu/cc_workspace/gemma_splitk/launch_unitrace_current.sh`
- Launch (bench): `probe.sh` with `USE_GRAPH=0 SPLITK_G=64` + `bench_bsz1.py`
- BW probe: `bw_probe2.cpp`; kernel-level split-K microbench: `hd512_decode/bug_repro.cpp`
- Decode-window analysis: `/tmp/utrace_fam2.py <bigjson>` (isolates via prefill-only
  `_per_token_group_quant_8bit` last-cluster boundary).
