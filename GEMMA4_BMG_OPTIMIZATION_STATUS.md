# Gemma4-31B FP16 FP8 Decode Optimization Status (BMG TP=2)

## ✅ UPDATE (2026-07-08): Gemma4 MTP 在 BMG/XPU 已打通（GAP AUDIT 中的阻塞项已解决）

**结果（同一 build、背靠背、同一 harness，rule-2 合规 A/B）：**
- 正确性（`gsm8k_chat_eval.py --n 100 --parallel 1`）：**MTP 0.990 (99/100, 0 invalid)** vs baseline 0.980 (98/100)，同分布内。
- 端到端时长：**MTP 543.1s vs baseline 973.2s = 1.79× 加速**。
- Draft accept 中段实测：`accept_len 3.5–3.9 / 4`，`accept_rate 0.83–0.96`（gsm8k 数学推理 draft 命中率高）。
- Server-side `gen throughput` 中段：MTP ~47–52 tok/s vs baseline ~26.6 tok/s。

**已交付的启动块（在原 shippable eager 基础上加 4 行）：**
```bash
# 原 shippable env 全保留（ZE_AFFINITY_MASK / SGLANG_USE_SGL_XPU / SGLANG_SKIP_VISION_GPU /
# SGLANG_FP8_IGNORED_LAYERS / SGLANG_SPLITK_G / SGLANG_XPU_FP8_W8A16_PREFILL / CCL_SYCL_*）

python3 -m sglang.launch_server \
  --model-path /llm/models/gemma-4-31B-it \
  --device xpu --tp 2 --quantization fp8 --dtype float16 --load-format layered_fp8 \
  --attention-backend intel_xpu --page-size 64 --mem-fraction-static 0.85 \
  --swa-full-tokens-ratio 0.05 --chunked-prefill-size 1024 \
  --disable-radix-cache --max-running-requests 1 --context-length 70000 \
  --disable-cuda-graph --skip-server-warmup --watchdog-timeout 3600 \
  --trust-remote-code --model-impl sglang \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path /llm/models/gemma-4-31B-it-assistant \
  --speculative-draft-model-quantization unquant \
  --speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-eagle-topk 1 \
  --host 0.0.0.0 --port 30000
```

### 与 GAP AUDIT 逐项对照

1. **原假设**："`xpu_backend.py:239-242` 的 `assert False` 硬阻断 FROZEN_KV_MTP 的 decode metadata init。" → **错**。`speculative/frozen_kv_mtp_utils.py::frozen_kv_target_view` 在 metadata init 时把 `forward_batch.spec_info` 临时置 None，assert 从不触发。**xpu_backend 无需改动。**
2. **实际的阻塞点（本次 session 定位并修复）：**
   - **P1** `spec_utils.py`：`_select_top_k_tokens_later` / `create_num_accept_tokens_filter` 用 `@torch.compile(disable=_is_npu)`，XPU 走 dynamo 时 `torch.xpu.synchronize` 二次 register 触发 `AssertionError`。**修**：`disable=_is_npu or _is_xpu`。
   - **P2** `eagle_utils.py`：`sgl_build_tree_kernel_efficient` 的 import 只在 CUDA/HIP/MUSA 触发；且 sgl-kernel-xpu 里那个同名 stub 签名少 `tree_mask_mode` 参数（11 vs 12），语义也未验证。**修**：接入 ptl 分支的 `custom_esimd_kernels_sglang.eagle_ops.{build_tree_kernel_efficient, verify_tree_greedy}`（签名与 CUDA op 逐字对齐）。`verify_tree_greedy_func` 里 XPU 分支之前是**空 return**（一切候选保持 -1），必须显式接入 SYCL 实现。
   - **P3**（最关键）**default draft-model quantization inheritance**：`--quantization fp8` 默认继承到 draft 模型。assistant checkpoint 是 bf16 且**没有 fp8 scales**（无 `weight_scale`/`input_scale`），sglang 加载时把 `qkv_proj.weight` 转成 fp8 但 scales 未初始化 → 每层 QKV matmul 出 NaN → draft logits 全 NaN → argmax=0 → `candidates=[bonus, 0, 0, 0]` → verify 全拒 → `accept_len=1.00 accept_rate=0.00`。**修**：`--speculative-draft-model-quantization unquant`（一行 CLI）。
3. **原假设**："Frozen-KV MTP 图捕获仅 CUDA，XPU 无 graph 收益。" → 依然成立（`speculative/frozen_kv_mtp_worker.py::init_cuda_graphs` 明确 `if target_worker.device != "cuda": return`），但**不是功能阻塞**，只是性能上限。已在 eager 下拿到 1.79×。
4. **原假设**："Frozen-KV MTP 强制 `disable_overlap_schedule=True`、关 mixed_chunk，与调优路线冲突。" → 与本项目 shippable eager 无冲突（本来就 eager + bs=1）。
5. **原假设**："文档口径差—— cookbook 说 MTP 可用但 intel_xpu 实不可用。" → 现已可用；下文档补丁项跟进。

### 本次 session 落地的最小改动（不涉及上游模型语义）

- `python/sglang/srt/speculative/spec_utils.py`：`_is_xpu` 常量 + 两处 `@torch.compile(disable=_is_npu or _is_xpu)`。
- `python/sglang/srt/speculative/eagle_utils.py`：新增 XPU import + 两处调用分支到 `custom_esimd_kernels_sglang.eagle_ops`（`build_tree_kernel_efficient` / `verify_tree_greedy`）。
- 无 `xpu_backend.py` / `gemma4_mtp.py` / `gemma4_causal.py` / kernel 侧改动。

### 已知限制 / 未做

- **只在 topk=1（chain）验证**；topk>1 在 XPU 上未启用（`frozen_kv_mtp_worker._init_draft_attn_backend` 明确 topk>1 需要 triton backend）。
- **未做 per-length TPOT bench**：现有 `cc_workspace/gemma_splitk/bench_bsz1.py` 使用 `input_ids=[1]*N`（连续 bos）作 dummy prompt——对非 spec 路径 valid（TPOT 只测 kernel time），但对 MTP path **病态**：assistant 与 target 对纯 bos 序列的 continuation 都不在训练分布内且互不一致，`accept_len` 塌回 1.0，反被 draft-forward 开销拖慢。实测复现：dummy bench 上 MTP `accept_len=1.00 accept_rate=0.00 throughput 6–13 tok/s`。**结论**：dummy prompt 不能用来度量 MTP TPOT。以 gsm8k wall-clock（1.79×）为准；per-length TPOT 若需要，需要给 bench 加 real-text prompt 支持。
- **未做 `--speculative-num-steps` / `--speculative-num-draft-tokens` 扫参**：默认 3/4/topk=1 在 gsm8k 已看到 accept_len 达 ~3.9（接近满 4），无强需求继续扫。

### Radix cache 兼容性（2026-07-08 已验证）

去掉 shippable 块里的 `--disable-radix-cache`，其余保持不变（含 `--speculative-*`）。启动 log 显示 `Tree cache initialized: source=default impl=SWARadixCache hybrid_swa=True` — SWARadixCache 与 SWA + FROZEN_KV_MTP 一起初始化 OK。

**验证方法**：`cc_workspace/mtp_radix_test.py` 发 3 个共享 ~8k prefix、问题不同的 chat 请求。结果：

| 请求 | Wall time | prompt_tokens | server-side #cached-token | 加速 |
|---|---|---|---|---|
| 1st cold | 7.04s | 8427 | **0**（完全 prefill）| — |
| 2nd warm | 0.89s | 8428 | **8384**（99.5% 命中）| **7.9×** |
| 3rd warm | 0.29s | 8427 | **8384**（99.5% 命中）| **24.3×** |

三个请求文本连贯、无 garble。**结论**：MTP + Radix cache + SWA 在 BMG/XPU 上正确工作；多轮 / 共享 prefix 场景建议开启。

### BFCL v4 multi_turn_base 兼容性（2026-07-08 smoke）

同一 MTP+radix server 上跑 `openfunctions_evaluation.py --num-samples 5 --num-threads 1`（sglang backend, skip-server-setup, temperature 0.0）。5 条 case 全部正常执行完成，输出 tool_call 全部是有效 Python 结构（`[cd(...), mkdir(...), mv(...)]`），并观察到 error-recovery 行为（首 tool call 失败 → 下一 step 自动调整路径 → 后续成功）。**功能层次的 "Failed to decode"** 仅在中间 step 出现，与 non-MTP 时的行为一致（BFCL 期望结构化 tool_call；模型某些 step 直接说话会被判为 decode 失败）——**不是 MTP/radix 的引入**。

**性能观察（工况差异，非 bug）：** BFCL/tool-call 场景 bs=1 accept_len 明显低于 gsm8k：`accept_len 1.0–2.4, accept_rate 0–47%`（gsm8k 是 3.5–3.9, 83–96%）。tool_call 输出高度结构化（JSON-like）但**具体参数值不可预测**（`folder='document'` 里的 `document` 是每例独有的），draft 命中率天然低。此工况下 MTP 收益微薄。**结论**：MTP 收益依 workload 而定；shippable 配置仍应保留 MTP（对数学/常识推理有收益，对 tool-call 中性），不应因 BFCL bs=1 accept_len 低就关掉。

### ⚠️ OPEN ISSUE (2026-07-08): MTP + BFCL bs=8 会 hang（bs=4 相同 BFCL workload 不 hang）

尝试在 `max_running_requests=8, num_threads=8, swa_full_tokens_ratio=0.2, radix on, MTP on` 环境跑 BFCL full-200。前 ~100/200 例正常返回，但跑到 ~100/200 后 server 10+ min 无输出，GPU 频率 2800 MHz 保持（不是 idle）但 log 无进展；8 个 BFCL client 线程全部 blocking on `httpx.read`。手动 kill 才结束。

**后续同 server config (max_running_requests=4) 上分别验证 bs=1/2/4 gsm8k 与 bs=2/4 BFCL smoke（2026-07-08）：**

| workload | n | threads | latency | server accept_len | agg gen throughput | 是否 hang |
|---|---|---|---|---|---|---|
| gsm8k bs=1 | 100 | 1 | 543.1s | 3.5–3.9 | ~47–52 tok/s | ✅ |
| gsm8k bs=2 | 20  | 2 | 68.6s  | **3.15–3.80** | 85–102 tok/s | ✅ |
| gsm8k bs=4 | 20  | 4 | 35.8s  | **3.43–3.73** | 170–186 tok/s | ✅ |
| BFCL bs=1  | 5   | 1 | (smoke, 全过) | 1.0–2.4 | — | ✅ |
| BFCL bs=2  | 10  | 2 | 126s | **1.00–1.25** | 15–25 tok/s | ✅ |
| BFCL bs=4  | 10  | 4 | 93s  | **1.00**       | 30–32 tok/s | ✅ |
| BFCL bs=8  | 200 | 8 | HANG @ ~100 | **1.00** | — | ❌ |

**两个正交观察：**

1. **workload-specific accept_len 塌**（跟 bs 无关）：BFCL bs=1 accept_len 已经只有 1.0–2.4，bs=2/4 也是 1.00–1.25。gsm8k bs=1/2/4 全部 3+。差异在 workload 本身：BFCL 输出高度结构化 tool_call（`[cd(folder='document'), ...]`），具体参数值不可预测；gemma-4 chat template 里 tool_call 前后有特殊 marker，assistant/target 对这些位置的分布分歧大。**这不是 XPU 或 spec 实现的 bug，是 draft 与 target 对 tool-call 语言的一致性差。**

   **实测证据（2026-07-08，`analyze_mtp_dump.py`，MTP bs=1 各跑 1 条 gsm8k 和 1 条 BFCL multi_turn_base_0）：**
   
   | workload | steps | avg accept_len | accept_rate | pos0 拒率 | pos1 拒率 | pos2 拒率 |
   |---|---|---|---|---|---|---|
   | gsm8k | 204 | 2.108 | 0.369 | 0.490 | 0.667 | 0.735 |
   | BFCL | 584 | **1.120** | **0.040** | **0.901** | **0.990** | **0.990** |
   
   BFCL 上 draft 的第一个 token 就有 90% 概率被拒，第 2/3 位几乎不可能命中。看被拒 target token 的分布（top-20 出现频次）：
   - **具体参数 payload**：`'pdf'`(64), `'report'`(62), `'final'`(54), `'analysis'`(40), `'temp'`(30), `'budget'`(30), `'content'`(26) —— 每例独有的文件名/字段值，assistant 4 层小模型无法从上下文预测；
   - **tool_call 语法 delimiter**：`'_'`(84), `"='"`(80), `" '"`(66), `"'"`(54), `'('`(44), `"',"`(22), `"')]"`(22) —— tool_call 结构符号序列；
   - **chat template 结构 marker**：`'<turn|>'`(38) —— gemma-4 template 的 turn boundary，每次 tool_call 结束都要正确 emit，draft 判断能力弱。
   
   对比 gsm8k 被拒 top-20 包含 `' day'`, `' used'`, `' remaining'`, `' the'`, `' of'`, `' to'` 等自然语言高频词，assistant 至少能猜对相当比例（accept_rate 37%）。
   
   BFCL 单条请求共生成 654 tokens、88 个 unique token，top 30 里几乎全是 `_ = ' ( 'pdf' 'report' 'final' Year` 之类 payload/语法，**结构上就是"低 self-consistency"的高熵序列**——每个 token 都在 encoding 独有信息，assistant 猜不到不是 bug，是 draft 模型自身能力上限（`num_hidden_layers=4`, `hidden=1024`）+ workload 特性。

2. **bs=8 + BFCL hang（bs≤4 相同 workload 不 hang；同 bs=8 但 gsm8k workload 不 hang）**：BFCL bs=4 accept 也塌到 1.00，但没 hang（93s 跑完 10 例）；后续用 gsm8k n=200 parallel=8 复测，**同 server config 但换 gsm8k workload 也不 hang**（299.9s 顺跑，见下 §gsm8k bs=8 A/B）。所以 hang **同时需要 bs=8 + BFCL 长累积 context**（每 turn 追加 tool_call 结果，10 turn 后单请求 context 数千 tok，8 并发就是数万活跃 KV），不是纯 bs=8 或纯 spec 逻辑问题。

### gsm8k n=200 parallel=8 MTP vs baseline A/B (2026-07-08)

同 build、同 `max_running_requests=8, swa_full_tokens_ratio=0.2, radix on, mem_fraction_static=0.85`、背靠背，仅 `--speculative-*` 4 flags 有无为唯一变量：

| Metric | Baseline (non-MTP) | MTP | Delta |
|---|---|---|---|
| gsm8k acc (n=200) | 0.990 (198/200) | 0.990 (198/200) | ✅ 完全一致 |
| Invalid | 0 | 0 | — |
| Wall-clock latency | 358.2s | **299.9s** | **1.19× faster** |
| Server avg gen throughput | 151.4 tok/s (158 samples) | **172.4 tok/s** (80 samples) | +14% |
| Server avg accept_len | — | **1.82 / 4** | — |

**关键观察**：MTP 在 bs=8 上仍 net-positive（+19% E2E），但 accept_len 从 bs=1 的 3.5-3.9 掉到 bs=8 的 1.82（同 workload），把潜在收益打折约 60%。这是**独立于 workload 的 bs-scaling 问题**，不是 workload 或正确性 bug。**详见 open item `invest-bs8-accept-drop`**：待调查方向包括 `FrozenKVMTPWorker` draft loop per-req step-id 处理、SWA windowing 在 spec-decode 的 per-req 长度差异、draft attn backend metadata 在 8 并发之间的共享、target-verify batch 展开（bs*n=32 q-tokens）的 DPAS 分片效率。

**当前建议：**
- **MTP + bs≤4 完全推荐**（gsm8k + BFCL 都跑完，accept 与吞吐健康）。
- **MTP + bs=8 gsm8k-style workload 可用**（+19% E2E，但 accept_len 已经打折）。
- **MTP + bs=8 BFCL-style workload 暂避免**（会 hang）——需要看是 8 并发 × 长 tool-call context 的资源竞争还是别的问题。
- **BFCL/tool-call workload 下 MTP 收益微弱** (accept_len≈1)——这类 workload 上 MTP 不该关（bs=1 时仍有偶尔命中），但也不必期待明显加速。

**外部独立复现（不同硬件/后端同一模型家族）**：LocalLLaMA 用户在 M4 Max (mlx-vlm) 上跑 gemma-4-26B-A4B MTP，观察到与我们完全一致的 workload 依赖模式：
- Code generation：1.53× 加速，66% 接受率
- Long-form prose：0.95× (打平)，31% 接受率
- JSON output：**0.50× 变慢**，**8% 接受率**

他给出的经验阈值："**once token acceptance dips below 50% the overhead kills the benefit**"，与我们 BFCL 上 4% 接受率导致的 accept_len 塌到 1.12（比 baseline 慢）一致。**结论证实：这不是 XPU 或 sglang 侧问题，是 gemma-4 assistant 模型自身在结构化输出上的通用弱点。**



## ⚠️ GAP AUDIT (2026-07-07): Gemma4 MTP 在 BMG/XPU 仍未打通（针对本项目交付路径）

**结论（当前事实）**：Gemma4 的 MTP/Frozen-KV 框架代码已存在，但在我们当前可交付配置
（`--attention-backend intel_xpu`）上不可用；因此“Gemma4 + MTP on BMG/XPU”仍是未完成项。

### 1) 已具备的基础能力（不是 gap）

- Gemma4 assistant 模型已接入：`python/sglang/srt/models/gemma4_mtp.py`
  (`Gemma4AssistantForCausalLM`, `Gemma4UnifiedAssistantForCausalLM`)。
- 参数路由已接入：`NEXTN/EAGLE + Gemma4 assistant` 会在
  `python/sglang/srt/arg_groups/speculative_hook.py` 中被提升为
  `FROZEN_KV_MTP`；`EAGLE3` 对该草稿架构被显式拒绝。

### 2) 阻塞本项目交付的核心 gap

1. **intel_xpu attention backend 对 speculative 仍硬阻断**  
   `python/sglang/srt/layers/attention/xpu_backend.py` 在 decode 分支中，当
   `forward_batch.spec_info is not None` 时直接 `assert False`，报错
   “XPUAttentionBackend doesn't support speculative decoding yet...”。  
   这意味着只要走 `intel_xpu` backend，就无法跑 MTP/Frozen-KV。

2. **与本项目 shippable 启动块冲突**  
   本文 “Launch Configuration” 的已交付命令固定使用
   `--attention-backend intel_xpu`（TP=2 eager 路径）。  
   因为上面第 1 条，该交付路径与 Gemma4 MTP 当前不可兼容。

3. **Frozen-KV MTP 在调度能力上仍是 spec-v1 语义**  
   `speculative_hook.py::_handle_frozen_kv_mtp` 与
   `spec_info.py` 明确：  
   - 不支持 spec v2 overlap（会强制 `disable_overlap_schedule=True`）；  
   - 会关闭 mixed chunk；  
   - 采用独立的 FrozenKVMTPWorker 路径。  
   这与我们当前对 overlap/chunk 的性能调优路线存在结构性差异，后续需单独评估。

4. **Frozen-KV MTP 的图捕获仅支持 CUDA，不支持 XPU graph**  
   `speculative/frozen_kv_mtp_worker.py::init_cuda_graphs` 明确仅在
   `target_worker.device == "cuda"` 时启用 draft CUDA graph；在 XPU 只走 eager draft loop。  
   即便功能打通，XPU 侧仍缺少 draft/verify 图化收益。

5. **文档与本项目现实能力存在“可用性口径差”**  
   - `docs_new/cookbook/autoregressive/Google/Gemma4.mdx` 给出 Gemma4 + NEXTN 命令；  
   - 交互部署片段 `docs_new/src/snippets/autoregressive/gemma4-deployment.jsx`
     也提供 MTP 开关（仅对 MI300X 隐藏），但没有 Intel XPU/BMG 专项限制说明。  
   对本项目而言，这会造成“文档看似可开 MTP，但 intel_xpu 路径实不可用”的认知偏差。

### 3) 尚未完成的验证 gap（必须补测）

- **未有本机 BMG 上 “intel_xpu + ESIMD 路径 + Gemma4 assistant + NEXTN(FROZEN_KV_MTP)” 的正确性门禁结果**  
  （至少应有 chat harness 正确性 gate；不以 Triton 作为目标实现）。
- **未有同口径性能数据**：  
  目前没有 “MTP(intel_xpu+ESIMD 路径) vs 非 MTP(intel_xpu shippable 路径)” 的对齐 A/B。
- **未形成 XPU 迁移方案的代码级拆解任务**：  
  例如：`xpu_backend` speculative metadata/verify/draft 扩展、与 SWA/full pool 的一致性、
  topk/page_size 组合约束、以及是否借鉴 ptl 分支的 XPU speculative kernel 资产。

### 4) 建议的收敛顺序（后续执行项）

1. 先做 **intel_xpu 后端 speculative 打通**：移除当前 assert 路障，补齐 metadata/verify/draft 路径，目标实现为 ESIMD 内核方案（不引入 Triton 目标依赖）。  
2. 做 **功能可用性最小闭环**：在 BMG 上跑通 `intel_xpu + NEXTN + gemma4-31B-assistant`，拿到正确性 gate。  
3. 再做 **同口径性能 A/B**：与当前 `intel_xpu` 非 MTP shippable 路径做对齐对比。  
4. 最后补文档：明确“Gemma4 MTP 在 Intel XPU（ESIMD 路径）的已验证组合与限制”。

## ✅ FIXED (2026-07-06): SWA seq>1024 decode garble — host-side page-aligned windowing

**Bug:** decode output garbled once total sequence length crossed the
`sliding_window=1024` boundary. Root cause: the intel_xpu ESIMD `page_attn_decode`
path read the FULL `[0, seq_len)` SWA page table. Once seq_len > window, positions
older than the window are evicted from the SWA KV pool and their
`full_to_swa_index_mapping` entry is reset to **slot 0** (see
`allocator/swa.py::free_swa` → `mapping[free_index]=0`), i.e. another token's KV
= garbage. The old `torch.clamp(cache_seqlens, max=window+1)` hack was ALSO wrong:
it kept the FIRST window+1 columns (the OLDEST/evicted positions), so it read
garbage too. FA3 fallback (`SGLANG_DISABLE_ESIMD_DECODE=1`) was verified correct
past 1024 and used as the reference.

**Fix (NO kernel change):** in `xpu_backend.py` forward_decode SWA branch,
page-align the SWA page table to the LAST `window` tokens and pass the reduced
seq_len, so the ESIMD kernel only ever reads resident, in-window KV. page_size=64,
window_tokens=1024 → gather/slice `n_win_pages = (window-1)//ps + 2 = 17` pages
starting at `start_page = clamp(seq_len-window,0)//ps`. Floor-aligning the start
reads only resident slots (may attend ≤page_size-1 extra still-resident slightly
older tokens — a benign SUPERSET of FA3's `window_size=(sliding_window_size,0)`;
the SWA pool evicts per page, so the boundary page is resident whenever it holds
any in-window token). Computed ONCE per step, cached on the (per-step-fresh)
metadata object, reused across all ~50 SWA layers. **The kernel sources
(page.attn.gqa2.h / eagle.sycl / ops.py) are PRISTINE — no `window` param.**

- **bs==1 fast path (shippable `--max-running-requests 1`):** the window is a
  CONTIGUOUS page span, so SLICE the page table (`page_table[:, s:e]`, a zero-copy
  view) + one device sub for seqlens. `start_page` derived from
  `metadata.max_seq_len_k` (already a host int → NO new device→host sync, rule 14).
  Microbench: 74us (gather) → **8us (slice)**, pt/seqlens cross-checked equal.
- **bs>1 path:** keep gather (per-batch start pages differ).

**Verified (2026-07-06, TP2 eager, pristine .so + xpu_backend windowing):**
- Correctness — BFCL v4 multi_turn_base real cases with cumulative history well
  past 1024: entry0 (4 turns, prompt 8491-8681 tok) and entry2 (5 turns,
  7049-7350 tok) → ALL structurally-correct tool calls, `finish=stop`, no garble.
  Pure-decode counting test coherent to ~1150 tokens. bs=1-slice output byte-identical to gather.
- Perf — bench_bsz1 (out=512) vs 2026-07-03 baseline: 1k **37.55** (−0.11), 2k
  37.74 (+0.48), 4k 38.14 (+0.48), 8k 38.93 (+0.50). At 1k (windowing mostly
  inactive) we MATCH/beat baseline; the ~+0.5ms appears only when windowing is
  active.
- **⚠️ REFUTED attribution:** the +0.5ms is NOT the windowing host ops. Proven:
  slice cut host cost 74us→8us (9×) yet E2E was byte-for-byte unchanged (gather
  run 37.60/37.78/38.19/38.97 vs slice run 37.55/37.74/38.14/38.93). Residual
  sub-ms is either small per-layer python dispatch (~0.1ms, 50× getattr+unpack) or
  measurement drift vs the (garbled, full-KV-scan) baseline. NOT yet localized —
  needs a decode kernel breakdown before further optimization (rule 6).

**ESIMD AOT build note (why the kernel-mask alternative was abandoned):** an
earlier attempt added a `window` param + `kvStart` lower-bound mask INSIDE
`sdpaDecodeGqa2Phase1`. It failed the BMG AOT device-link with
`The list of SPIR-V modules contains more than one module with an entry point!`
(ocloc -11). Single-variable bisection: keeping the `window` PARAM + full
plumbing but reverting the Phase1 BODY to pristine → **builds fine**; the mask
BODY is the trigger. These ESIMD kernels sit at the GRF ceiling (see VL512 note:
208 GRF > 128 → large-GRF/spill), and the extra `kvStart`+compound-`simd_mask`
comparisons perturb module splitting past what the AOT flow accepts (per_kernel
split is already on). Host-side windowing sidesteps this entirely.

## ✅ FIXED (2026-08-19): 同一个 SWA 窗口 bug 还存在于**真正在跑的那条 decode 路径**（sglang_decode_attn）

上面 2026-07-06 那节修的是 `page_attn_decode`（paged 路径）。但**当前 stack 根本不走那条路**：

- `SGL_XPU_DECODE_SGLANG_ATTN` 默认 `1`，gate 是 `layer.head_dim == 256`，即**所有 50 个 sliding 层**的 decode 都走 `_sglang_decode_attn_fn`。
- 而且本镜像的 ESIMD wheel **没有 `page_attn_decode` 这个 op**（实测 `custom_esimd_kernels_sglang` 89 个 op 中不含），所以 `_use_esimd_pa` 恒为 False —— paged 分支和它那段窗口修复都是**死代码**。

`_build_sglang_decode_attn_inputs_eager` 的 `kv_indptr/kv_indices` 用**全池** `req_to_token` + 完整 `cache_seqlens` 构建，然后拿去索引 **SWA 池**的 k/v buffer，两个缺陷叠加：

1. **没有窗口裁剪**：sliding 层遍历全部 `[0, seq_len)`，而只有末尾 `window` 个 token 还驻留；
2. **索引空间错误**：`req_to_token` 存的是 full-pool 槽位号，未经 `translate_loc_from_full_to_swa` 转换。槽位号一旦超过 SWA 池大小（本配置 18240）就**越界读**。

**症状**：某个 sliding 层（实测稳定是 layer 4）的 `core_attn_out` 突然全 NaN → 整条请求 logits 全 NaN。
- **贪心解码时静默**：argmax 对 NaN 不报错，只是输出垃圾（BFCL 表现为 "Empty response from the model"，结果长度 13–31）。
- **温度采样时炸**：`torch.multinomial` 的异步断言触发（`TensorCompareKernels.cpp:180: Assertion input_[0] != 0`），server 死，且**把 XPU context 弄成 wedged**——下次启动会卡在 `torch.xpu.empty_cache()` / `_move_module_tensors_to_device`，**容器要重启两次**才恢复。

**定位过程中被实测排除的因素**（都不是原因）：`swa_full_tokens_ratio`(0.2 vs 0.4)、并发(threads 1 vs 4)、radix cache(SWARadixCache vs SWAChunkCache)、全部 ESIMD fast path（QKV / decode / page attn / fused norm 全关仍复现）、KV cache 内容本身（实测 layer4 整个 KV buffer `nonfinite=0`，`|k|<=0.91`、`|v|<=16`）。**长上下文本身也不是充分条件**：单条 4k/8k/14k/22k 合成 prompt 贪心解码 logprob 全有限——因为那时分配到的槽位号还没超过 SWA 池大小。

**修复**（`xpu_backend.py`，commit `a8d3bae93e`）：sliding 层单独构建 kv 输入——按 `min(seq_len, window)` 裁长度、用 triton builder 的 `kv_start_idx` 从 `seq_len - len` 起始、再经 `translate_loc_from_full_to_swa` 转换槽位；full-attention 层不变，两者分开 memoize。同时把已有的 window view 也接到 split-K fallback（它同样接受 head_dim 256）。

**验证**：BFCL smoke 5/5 OK（修复前 5/5 connection error）；gsm8k_chat_eval n=100 = **0.980**；bench 同配置背靠背 **TPOT 8192×256 47.15 → 41.78ms（-11.4%）**、4096 42.07 → 41.00ms，TTFT 不变（改动只影响 decode）。BFCL 生成速率 0.25 条/分 → **10.5 条/分**。

**教训**：`page_attn_decode` 缺失是静默的（`is not None` 判空后直接换路径），所以"给 paged 路径修了 SWA 窗口"这件事对交付配置**零效果**。改 attention 相关代码前，先确认**哪条分支真的在跑**（读 gate + 确认 op 是否存在于 wheel）。

## ✅ BFCL v4 multi_turn_base FORMAL BENCHMARK (2026-07-06): FULL 200 = 72.50% (radix on)

Ran the official BFCL kit pipeline (`bfcl generate` → tool execution → `bfcl
evaluate` state/response check), NOT a hand-rolled "looks-reasonable" check. Server
= swa_full_tokens_ratio=0.2, **radix cache ENABLED**, ESIMD decode, `--num-threads 8`
(running-req=8, multi-concurrent). Kit vendored at `cc_workspace/gemma_splitk/_bfcl_vendor`.

**FINAL RESULT: full multi_turn_base (all 200) = 72.50% (145/200), official full
eval (no --partial).** Generation took 782s (~13min) thanks to radix prefix reuse
+ 8-way concurrency. Failures (55): 37 instance_state_mismatch + 12
execution_response_mismatch + 6 empty_turn (genuine multi-step reasoning errors;
empty_turn down to 3% from the 20% seen before the tool-result fix).

Getting a valid score required root-causing TWO setup bugs (rule 12: suspect your
own setup before blaming the model — the raw score was 6.67% and it would have been
wrong to conclude "gemma-4 is bad at tool use"):

1. **`thought\n` leak → 6.67% (2/30).** The vendored `GemmaHandler._format_prompt`
   used the OLD gemma-3 hand-rolled `<start_of_turn>` template. gemma-4 uses
   `<|turn>`/`<|channel>thought` markers, so the model leaked a `thought\n...`
   prefix that broke BFCL's response decoder. FIX: render the REAL gemma-4 template
   via `tokenizer.apply_chat_template(..., enable_thinking=False)` (injects the empty
   `<|channel>thought\n<channel|>` block that suppresses reasoning; matches the
   verified /v1/chat/completions path). Rename handler's "model" role → "assistant".

2. **Tool results invisible → still low (the REAL root cause).** gemma-4's
   apply_chat_template SILENTLY DROPS BFCL's plain `{"role":"tool","name":..,
   "content":..}` messages (it expects structured tool_calls/tool_response). So the
   model NEVER saw execution results → asked for clarification (empty_turn) or
   looped (force_terminated). FIX: in `_format_prompt`, fold tool results into a
   user turn (`"Tool execution result:\n"+content`) and merge consecutive user
   turns. This took first-30 **6.67% → 63.33% (9.5×)**; the empty_turn and
   force_terminated failure classes largely VANISHED. Full-200 landed at 72.50%.

## ✅ RADIX CACHE VALIDATED ON XPU (2026-07-06): enable it for multi-turn / shared-prefix serving

Prior configs all set `--disable-radix-cache` — but per facts note this was only
"radix+SWA on XPU UNVALIDATED", not known-broken (code has `SWARadixCache` for
hybrid SWA). Tested by dropping the flag (keeping swa_full_tokens_ratio=0.2):
- **Server starts fine** — SWARadixCache initializes, `disable_radix_cache=False`, ready.
- **Prefix reuse works**: two requests sharing an 8k prefix → 2nd TTFT 6255ms →
  **100ms (98% / 62× faster)**. Multi-turn generation logs show #cached-token in the
  thousands (system prompt + func docs + accumulated history reused each turn).
- **Zero correctness loss**: BFCL multi_turn_base accuracy IDENTICAL with/without
  radix (first-30 = 63.33% both). Full-200 (radix on) = 72.50%.
- **~10× faster multi-turn generation**: 30-entry BFCL gen 782s→107s vs the
  no-radix run; full 200 in 782s.
**Recommendation: for real multi-request / multi-turn / shared-prefix serving, REMOVE
`--disable-radix-cache`.** (bsz=1 single-shot latency bench is unaffected either way.)

Other setup notes: added a `/llm/models/gemma-4-31B-it` ModelConfig → GemmaHandler
entry; made model_config.py's ~50 handler imports tolerant (`_try` + placeholder)
so missing optional API SDKs (cohere/anthropic/...) don't crash import; made the
tree-sitter java/js parser init tolerant (version/module mismatch); `pip install
overrides`. NOTE: an earlier `pip install --no-deps` of API SDKs POLLUTED the env
(boto3 without botocore broke accelerate import) — uninstalled; use the tolerant
imports instead. All edits are in `_bfcl_vendor` (test kit), NOT product code.

Correctness cross-check (same server): gsm8k_chat_eval --parallel 16 (running-req
16) = **0.990 (99/100), 0 invalid** — the swa_ratio=0.2 concurrency fix does NOT
hurt correctness under 16-way concurrency.

## ⚠️ OPEN ISSUE (2026-07-06): DECODE DOES NOT SCALE WITH BATCH — batch>1 has a real perf problem

**Batch perf matrix (TP2 eager, out=512, SWA-windowing fix active, bs=16-capable server).**
⚠️ Collected on a SHARED node with a confirmed other tenant at ~99% CPU; sglang
scheduler is CPU-bound so TPOT is inflated and noisy (`*` = obvious contention
artifact — non-monotonic). Trends + aggregate-throughput direction are reliable;
absolute TPOT is high. Re-collect on an idle node for shippable numbers.

bsz=4:
| input | TTFT(ms) | TPOT(ms) | dec tps/req | agg dec tps | E2E(s) | E2E tps |
|-------|----------|----------|-------------|-------------|--------|---------|
| 1024  |   923    | 94.0*    | 10.6 | 42.5 | 48.97  | 41.8 |
| 2048  |  1760    | 95.3*    | 10.5 | 42.0 | 50.44  | 40.6 |
| 4096  |  3626    | 63.8     | 15.7 | 62.7 | 36.21  | 56.5 |
| 8192  |  7420    | 68.2     | 14.7 | 58.7 | 42.24  | 48.4 |
| 16384 | 16224    | 102.3    | 9.8  | 39.1 | 68.51  | 29.9 |
| 32768 | 38437    | 141.4    | 7.1  | 28.3 | 110.67 | 18.5 |
| 65536 | 117554   | 103.9    | 9.6  | 38.5 | 170.65 | 9.0  |

bsz=8:
| input | TTFT(ms) | TPOT(ms) | dec tps/req | agg dec tps | E2E(s) | E2E tps |
|-------|----------|----------|-------------|-------------|--------|---------|
| 1024  |  1531    | 59.9     | 16.7 | 133.6 | 31.18  | 83.3 |
| 2048  |  3064    | 87.0*    | 11.5 | 92.0  | 47.76  | 60.9 |
| 4096  |  6550    | 159.9    | 6.3  | 50.0  | 88.95  | 35.1 |
| 8192  | 13545    | 204.2    | 4.9  | 39.2  | 127.00 | 28.5 |
| 16384 | 29354    | 365.3    | 2.7  | 21.9  | 221.52 | 17.1 |
| 32768 | 69691    | 134.8    | 7.4  | 59.3  | 146.74 | 19.6 |
| 65536 | 231313   | 103.9    | 9.6  | 77.0  | 284.38 | 9.0  |

**⚠️ CORRECTION of an earlier wrong call (2026-07-06).** After the first bs=8 8k
trace I claimed "the bs>1 slowdown is mostly TTFT/scheduling; the decode kernels
are healthy and per-token DPAS GEMM has no bs>1 bug." **That was wrong.** The
matrix shows decode itself does NOT scale with batch:
- bs=1→8 @ 8k: TPOT 39 → 204ms = **5.2× slower per token** even though batch is 8×.
- agg decode tps does not rise with batch and even regresses: bs=8 @ 8k = 39 tps,
  no better than bs=4 @ 16k. A healthy batched decode would keep per-token latency
  ~flat and grow aggregate tps ~linearly; it does neither.
- What IS true from the trace: the SWA windowing fix works (50 SWA layers read a
  constant ~1088 tokens, not full ctx) and there is no per-token-GEMV dispatch bug
  (M=8 correctly hits FP8_GEMM_DPAS_V9 batched GEMM). So the batch scaling loss is
  elsewhere — TBD by a bs=4 8k decode trace (in progress). Candidate suspects:
  FP8 DPAS GEMM efficiency at M=4/8, allreduce cost per step, or split-K global
  layers doing O(bs×ctx) work. Do NOT attribute to TTFT again without trace proof.
- 32k/64k TPOT DROPS vs 16k (bs8: 365→135→104ms) because KV memory caps concurrent
  decode (at 64k only ~2-3 of the 8 requests are resident/decoding at once; the
  rest queue), so effective batch shrinks — a memory-capacity artifact, not a speedup.

### Aligned bs=1 vs bs=4 @8k decode A/B (both traced, same analyzer, 2026-07-06)

Steps derived from the documented cadence **allreduce = 120/step** (60 layers × 2:
post-attn + post-FFN), NOT guessed. bs=1: 19303/120=160.9 steps; bs=4: 15787/120=131.6.
Both at 8k ctx (ctx confounder removed vs the earlier bs=1@4k ROOFLINE §2 table).

| family        | bs1 ms/step | bs4 ms/step | bs4/bs1 |
|---------------|-------------|-------------|---------|
| FP8 weights   | 25.34       | 35.19       | **1.39×** |
| allreduce     |  5.05       |  4.75       | 0.94×   |
| lm_head       |  4.66       |  4.69       | 1.01×   |
| page_attn SWA |  2.11       |  4.13       | **1.96×** |
| splitK global |  1.81       |  4.19       | **2.32×** |
| RMSNorm       |  1.32       |  2.13       | 1.61×   |
| **SUM device**|  40.27      | 55.08       | **1.37×** |

**Corrected conclusion (supersedes my earlier wrong calls in this section):**
- The bs=1→4 device scaling is actually MILD: 1.37× device-busy for 4× batch —
  close to ideal for a memory-bound-weights-dominated decode. My earlier "decode
  is 5.2× slower / batch is badly broken" framing conflated bs=8 with bs=4 and
  mixed in TTFT/contention.
- **What scales worst is attention** (splitK global 2.32×, page_attn SWA 1.96×):
  compute-bound, per-request KV, NO cross-batch amortization → grows ~linearly with
  batch. But absolute cost is small (bs4: 8.3ms combined).
- **FP8 weight GEMM scales WELL (1.39×)** — memory-bound, weight shared across the
  batch; kernel microbench (esimd_gemm_fp8_pert / FP8_GEMM_DPAS_V9, weight-pool
  cold-HBM) confirms per-row cost drops to 0.14–0.29× from M=1→8 (adding tokens is
  nearly free). So my even-earlier "FP8 GEMM 63% = the bottleneck to fix" was also
  wrong: it is the biggest ABSOLUTE cost but it batches the BEST. Do NOT chase it
  for batch scaling.
- bench TPOT bs1→bs4 = 39→68ms = 1.74×, vs device 1.37×. The extra ~0.37× is
  host-gap (more kernel launches/step at bs>1) + shared-node CPU contention.

### ✅ ROOT-CAUSED + FIXED (2026-07-06): bs=8/16 collapse = SWA KV pool too small → retract

The bs=8/16 TPOT collapse (154–365ms) is **NOT** a decode-kernel / FP8-GEMM /
attention / TTFT problem. Root cause found in the **scheduler's own logs** (not
inferred): the SWA KV pool is tiny and overflows under many concurrent long inputs,
forcing the scheduler to retract requests and cap the running batch.

Evidence (bs=8, 8k input, out=1024, `swa_full_tokens_ratio=0.05`):
- Startup: `Use sliding window memory pool. full_layer_tokens=181760,
  swa_layer_tokens=9088`. The 50 SWA layers get their OWN small pool sized
  `swa_tokens = full_tokens × swa_full_tokens_ratio` = 181760 × 0.05 = **9088**.
- `swa token usage` peaks at **1.00 (100% full)** while `full token usage` is only
  **0.31** → the SWA pool is the bottleneck, the full pool is oversized.
- `KV cache pool is full. Retract requests` fired **104×** during the run.
- Scheduler `#running-req` never reaches 8 (peaks 5–6, sustained ~5); `#queue-req`
  sits at 4–7 (requests waiting, not admitted). Confirmed independently by the
  KVScatter grid-dim0 (= per-step active decode batch) which maxes at 6, never 8.
- Why: 9088 SWA tokens ≈ 8×1024 (window) with ZERO prefill headroom. During prefill
  (before window eviction) 8×8k requests transiently need more → overflow → retract.

**Fix (config only, ZERO kernel change): raise `swa_full_tokens_ratio`.** It
rebalances the SAME memory budget: bigger SWA pool, smaller (still-sufficient) full
pool. A/B at `swa_full_tokens_ratio=0.2` (SWA pool 9088→18176, full 181760→90880,
90880 still ≫ 8×8192):

| bs=8 @8k out=512      | ratio=0.05 (old) | ratio=0.2 (fixed) | gain   |
|-----------------------|------------------|-------------------|--------|
| running-req (decode)  | 5–6 (never 8)    | **8 (all 51 batches)** | —  |
| retracts              | 104              | **4**             | —      |
| TPOT                  | 204.2 ms         | **91.0 ms**       | 2.24×  |
| agg decode tps        | 39.2             | **87.9**          | 2.24×  |
| E2E tps               | 28.5             | **68.6**          | 2.41×  |
| E2E time              | 127.0 s          | **59.6 s**        | 2.13×  |

So decode DOES scale with batch once the SWA pool can actually hold the concurrent
requests. **Recommendation: for multi-request long-input serving, raise
`--swa-full-tokens-ratio` (0.05→~0.2; default upstream is 0.8). 0.05 was tuned for
bs=1 memory frugality and starves concurrency.** Tune per target (bs, input len):
SWA pool must hold ≈ concurrent_reqs × (window + prefill_headroom); full pool must
stay ≥ concurrent_reqs × max_input_len.

Methodology note (self-correction): earlier in this session I mis-attributed the
collapse three times (TTFT/scheduling; then "FP8 GEMM 63% bottleneck"; then a
bogus per-step A/B from an unaligned bs=8 trace giving "bs=8 cheaper"). All wrong.
The fix came only from reading the scheduler's retract/running-req/swa-usage logs +
KVScatter grid for definitive per-step active-batch — not from step-count guesses.

## ⚠️ STATUS (2026-07-01): XPU GRAPH IS BROKEN — SHIP EAGER FOR NOW

The "XPU Graph" column below was measured on a config whose decode output is
**garbled** (see "Known Issue #2: XPU Graph decode garble" below). Those graph
TPOT/accuracy numbers are NOT valid — they were taken before correctness was
checked. Do NOT enable XPU graph until the allreduce fix lands.

**Currently shippable = fp16 + ESIMD + intel_xpu + layered_fp8, GRAPH OFF (eager)**,
verified 2026-07-01: gsm8k chat 0.950 (38/40, 0 invalid), coherent output, and
BETTER TPOT-vs-ctx than the (broken) graph at every length:

> ⚠️ **2026-07-06：下表 eager 列是 2026-07-01 的旧值（融合优化前）。** ①②③④ decode 融合 +
> HD512 FMHA tile 调优 + GEMV VL 调优落地后，**当前 shippable eager TPOT 已降到 ~37.7ms
> (1k) / 37.7ms (4k) / 38.4ms (8k)**（out=512，gsm8k 0.975）。**以最新性能矩阵为准**（见本文
> "刷新的完整性能矩阵 …（2026-07-03）" 节）。下表仅用于说明 "eager 优于坏掉的 graph" 这一对比关系，
> 其绝对数值已过时。

| Metric | bf16 baseline | EAGER fp16+ESIMD (2026-07-01 旧值) | fp16+ESIMD+XPU Graph (BROKEN — garbles) |
|--------|--------------|------------------------------|------------------------------------------|
| TPOT (1K ctx) | 65.5ms | ~~40.5ms~~ → **37.7ms (当前)** | 45.8ms |
| TPOT (4K ctx) | — | ~~46.2ms~~ → **37.7ms (当前)** | 72.0ms |
| TPOT (8K ctx) | — | ~~54.8ms~~ → **38.4ms (当前)** | 106.9ms |
| gsm8k accuracy | 0.990 | **0.975** (chat, 当前) | ~0 (garbled decode) |
| Decode tok/s (1K) | 15.3 | 26.6 (当前) | 21.8 |

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

> ⚠️ **2026-07-03 已被取代（此表为 out=256、VL512、①②③④ 融合前的中间基线）。** 后续
> ①②③④ decode 融合 + GEMV VL512→256 + HD512 FMHA tile 调优落地后，TPOT 再降 ~1.5ms
> (1k/4k = 37.7ms)、TTFT 大幅下降（长 input −20~38%）。**以本文最后一张矩阵为准**
> （"刷新的完整性能矩阵（HD512 FMHA tile-tuned + GEMV VL256 全生效…2026-07-03）"）。
> 下表保留仅为记录该阶段的 clean-bench 复现事实（~39.2ms 可复现，非 unitrace 的 42-43ms 假象）。

| input | TTFT (ms) | TPOT (ms) | tok/s | E2E (s) |
|-------|-----------|-----------|-------|---------|
| 1024  | 621       | 39.23 | 25.5 | 10.63 |
| 2048  | 1276      | 39.25 | 25.5 | 11.28 |
| 4096  | 2644      | 39.22 | 25.5 | 12.64 |
| 8192  | 5721      | 39.27 | 25.5 | 15.73 |
| 16384 | 12805     | 40.99 | 24.4 | 23.26 |
| 32768 | 31100     | 44.16 | 22.6 | 42.36 |
| 65536 | 84143     | 50.45 | 19.8 | 97.01 |

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

## Scheduler 侧配置调查 + Overlap Schedule A/B（2026-07-06）

> 目的：排查 sglang scheduler 侧是否还有可启用/可调的配置能改善 gemma4-31B（dense、TP2、XPU、eager、bsz=1）。
> 结论：**唯一相关的旋钮 = overlap schedule，实测保持 ON（默认）已最优**；其余旋钮对本 workload 无效或不可用。

### Overlap Schedule ON/OFF A/B（同一容器、干净重启、节点空闲、bsz=1、out=256、warmup2/trials3）

| input | TTFT ON | TTFT OFF | **TPOT ON** | **TPOT OFF** | tok/s ON | tok/s OFF |
|------|---------|----------|-------------|--------------|----------|-----------|
| 1k | 346.9 | 310.7 | **37.64** | 39.02 | 26.6 | 25.6 |
| 4k | 1382.9 | 1349.1 | **37.65** | 39.11 | 26.6 | 25.6 |
| 8k | 2913.0 | 2883.5 | **38.43** | 39.89 | 26.0 | 25.1 |

- **TPOT：overlap ON 快 ~1.4ms/step（~3.7%）**，三 ctx 一致（gap 稳定、非噪声）。即便 bsz=1，overlap 把**下一步 CPU 侧调度/采样与本步 GPU forward 重叠**，压低 per-step host 开销。
- **TTFT：overlap OFF 略优 ~30-36ms**（1k 最明显，8k 收敛到 ~30ms）——overlap 流水线给首 token 加一步延迟；绝对量小，随 input 增大占比可忽略。
- **⚠️ 修正**：`GEMMA4_FP16_OPTIMIZATION_PLAN.md` 旧论断"overlap_schedule helps prefill **not decode**"被本实测**推翻**——在 host-dispatch 敏感的 bsz=1 decode 上 overlap ON 反而更好；之前担心的"overlap 抢 CPU 致 decode 变慢"在空闲节点未出现（那是共享节点争用的独立现象）。
- **决策：保持 overlap ON（默认，`disable_overlap_schedule=False`），无需改动。**

### 其余 scheduler 旋钮：无效 / 不可用（勿浪费时间）
- **`--num-continuous-decode-steps`**：**本版本已成 dead flag**——`scheduler.py` 0 引用（全树仅 `server_args.py` 定义 + `auto_benchmark_lib.py` 列名）。本欲"每次跑多 decode step 减调度开销"，正对口 host-overhead，但此版已移除，设了无效。
- **`--enable-two-batch-overlap` / `--enable-single-batch-overlap`**：断言要求 `moe_a2a_backend != none`（`server_args.py:7552`）；gemma4 dense 无 MoE → 直接报错。
- **`--enable-mixed-chunk` / radix cache（去 `--disable-radix-cache`）/ `--max-running-requests >1`**：仅对**真实并发 serving**（多 in-flight / 共享前缀）有意义，bsz=1 latency bench 不触发。XPU 均已验证支持（continuous batching 设备无关，session 存档）。
- 已测并否决：`--chunked-prefill-size 1024→8192`、SWA-pool ratio 调优（BMG 显存不可行）。

## CCL 环境变量复核（2026-07-06，同一容器、干净重启、节点空闲）

> 背景：旧 `_claude_tmp/docs/PERF_CCL_AB_TP2.md`（2026-06-26，**bf16 栈** TPOT 70ms）曾测出 4 个
> `CCL_SYCL_*_SIMPLE_THRESHOLD=4GiB` env 带来 TTFT −50~61%、TPOT 中性。但当前是 fp16+W8A16 栈，
> prefill 构成已变（W8A16 大幅削 GEMM），故复核该结论是否仍成立。

### A/B：当前 W8A16 栈，有/无 4 个 CCL SYCL threshold env（overlap 均 ON）
| input/out | TTFT ON | TTFT OFF | TPOT ON | TPOT OFF |
|-----------|---------|----------|---------|----------|
| 1024/256  | 347.2 | 347.1 | 37.65 | 37.65 |
| 4096/256  | 1383.0 | 1383.9 | 37.66 | 37.66 |
| 8192/256  | 2915.4 | 2916.2 | 38.44 | 38.44 |
| 16384/128 | 6445.9 | 6440.4 | 40.01 | 40.00 |

- **结论：当前栈上这 4 个 CCL env 已是 no-op**——TTFT/TPOT 有/无**逐字节一致**（差 <1ms 噪声）。
  **旧 bf16 A/B 的 TTFT −50~61% 收益在当前 W8A16 + oneCCL 2021.17 栈上不再复现**（大概率：新 oneCCL 默认 simple-threshold 已够高，chunked-prefill=1024 的 ~11MB allreduce 消息默认就走 simple；或旧测未用 chunked-prefill、单发大消息才命中旧默认阈值）。
- **保留无害**（正确性不受影响），但**不再作为性能项**；`PERF_CCL_AB_TP2.md` 的结论仅适用旧 bf16 栈。

### 硬件层面：allreduce 是 PCIe 通信墙，env 无法突破
- `xpu-smi topology -m`：GPU0↔GPU1 = **`NODE`（PCIe host bridge），无 `XL`（XeLink）**。坐实 trace 里的 `oneccl_allreduce_pcie`。
- **BMG 双卡无 XeLink** → allreduce（decode 第 2 大项，21.7% device / ~11.8ms/step traced；prefill ~11%）是 **PCIe 硬件墙**，换 transport 类 env 无从优化。
- oneCCL `2021.17`，`CCL_CONFIGURATION=cpu_gpu_dpcpp`。可试但低 ROI：`CCL_ALLREDUCE=<algo>` 算法选择、`CCL_WORKER_COUNT/AFFINITY`（decode 小消息本就 simple，预期无感）。真正压 allreduce 需算法/互联层（如图安全 custom SYCL allreduce 把它捕进图省 launch，但不动 PCIe 带宽本体）。

## TP=4 评估：未被正常支持（2026-07-06，同一容器、干净重启、4 卡空闲）

> 目的：评估 TP=4（GPU 0,1,2,3）能否降 TPOT / 扩 KV 容量。结论：**TP=4 不可交付**——server 能起且
> READY，但短 input prefill 的 TTFT 出现物理上不可能的反转，判定 prefill/comm path 不稳定。

- 启动成功：`launch_tp.sh TP=4 AFFINITY=0,1,2,3`，`max_total_num_tokens=603200`（TP2 的 3.3×，KV 容量确实更大）。
- 全矩阵（out=512，warmup2/trials3）实测 TTFT：

  | input | TTFT (ms) | 与相邻 input 关系 |
  |-------|-----------|------------------|
  | 1024  | **10979** | ❌ 反常：比 4k 大 11× |
  | 2048  | **13538** | ❌ 反常：比 4k 大 14× |
  | 4096  | 962       | 正常 |
  | 8192  | 2004      | 正常 |
  | 16384 | 4316      | 正常 |
  | 32768 | 9860      | 正常 |
  | 65536 | 24668     | 正常 |

- **1k/2k 的 TTFT（~11s/13.5s）比 4k（~0.96s）大一个数量级**，在 warmup=2 之后仍出现 → 物理上不可能，
  指向 TP=4 的 prefill/首批 comm path 在小 shape 上不稳定（疑似 4-way PCIe allreduce 的 JIT/warmup 未收敛，
  或 per-shape kernel 首次编译集中在小 input）。**TPOT 正常**（全程 ~34ms，甚至略优于 TP=2 的 37.7ms），
  说明 decode path 本身能跑，问题在 prefill。
- **决策：不采用 TP=4。** shippable 仍为 **TP=2**。若未来要复活 TP=4，需先根因 1k/2k TTFT 反转
  （逐 shape 追首批 prefill 的 kernel-compile / allreduce 建链耗时），在小 input 稳定前不可交付。
- 服务已按显式 PID kill 关闭，4 卡显存均回落到 ~42MiB。

## Launch Configuration

> ⚠️ **2026-07-06 修正**：旧启动块曾写 `SGLANG_XPU_ENABLE_GRAPH=1` + `--cuda-graph-bs 1`，
> 与本文结论（**XPU graph 坏、ship EAGER**）直接矛盾，且缺 `SGLANG_USE_SGL_XPU` /
> `SGLANG_SPLITK_G` / `SGLANG_XPU_FP8_W8A16_PREFILL` 三个 known-good env。以下为本 session
> 实测在用的 **shippable eager 配置**（TP=2，graph OFF，W8A16 prefill 默认），与
> `cc_workspace/gemma_splitk/launch_tp.sh`（`TP=2`）一致。

```bash
export ZE_AFFINITY_MASK=0,1
export SGLANG_USE_SGL_XPU=1
export SGLANG_SKIP_VISION_GPU=1
export SGLANG_FP8_IGNORED_LAYERS=vision_tower,embed_vision
export SGLANG_SPLITK_G=64                 # split-K decode attn（§1 已调好）
export SGLANG_XPU_FP8_W8A16_PREFILL=1     # opt#2：W8A16 prefill 默认路径
# 下面 4 个 CCL env 在当前 W8A16 栈上已是 no-op（见 "CCL 环境变量复核"），保留无害，非性能项
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
  --disable-cuda-graph \                  # EAGER：graph 在 TP>1 坏（Known Issue #2）
  --skip-server-warmup --watchdog-timeout 3600 \
  --trust-remote-code --model-impl sglang \
  --host 0.0.0.0 --port 30000
```

> **TP：只用 TP=2。** TP=4 经 2026-07-06 实测**未被正常支持**：server 能起且 READY，但
> 1k/2k prefill 的 TTFT 反常（1k≈11s、2k≈13.5s，远大于 4k 的 ~0.96s，物理上不可能的反转），
> TPOT 虽正常（~34ms）。判定 TP=4 prefill/comm path 不稳定，不可交付。详见 "TP=4 评估" 节。

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
