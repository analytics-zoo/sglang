# Qwen3.8-27B GGUF / XPU 开发验证日志

本文件记录 Qwen3.8-27B GGUF 原生 XPU 支持开发过程中实际执行的命令、结果、问题、显存和性能。测试要求和阶段门禁见 [实施与验证计划](./QWEN3_8_GGUF_XPU_SUPPORT_PLAN.md)。

## 1. 记录规则

- 每次测试分配唯一 Run ID：`YYYYMMDD-HHMM-阶段-序号`。
- 所有时间使用 UTC。
- 每条记录必须包含代码版本、dirty diff、容器、Python 路径、kernel `.so` 路径及哈希。
- `未执行`、`失败`、`通过`必须明确区分。没有数据时填 `N/A（原因）`，不能留出容易被误认为通过的空白。
- 性能保留每次原始值并计算中位数，不能只登记最好结果。
- HTTP 200 与输出正确性分别登记。
- 遇到问题先新增 Issue ID，再在 Run 中引用；修复后补充根因和回归结果，不删除失败记录。
- 大日志保存在稳定路径时，在本文件登记绝对路径和 SHA256；关键错误摘要直接写入本文件。

状态取值：`未开始`、`进行中`、`阻塞`、`通过`、`失败`。

## 2. 阶段看板

| 阶段 | 状态 | 最近 Run | 正确性 | 显存 | 性能 | 隔离性 | 备注 |
|---|---|---|---|---|---|---|---|
| 0 基线 | 通过 | `20260910-0451-S0-04` | 30000 已由外部停止 | XPU 6/7 约 229 MiB | N/A | 无待停止服务 | 目标设备为 6/7 |
| 1 Q4_K col_perm | 通过 | `20260910-0502-S1-03` | 固定请求集通过 | 峰后 31.76/31.01 GiB | 已记录 | 仅使用 6/7 | 含 large-FP16 transpose OOM 修复 |
| 2 IQ4_NL/XS native | 通过 | `20260910-0612-S2-04` | reference/kernel/native/fallback E2E 通过 | 权重少约 9.60 GB/rank | A/B 已记录 | 仅使用 6/7 | M=1 decode 提升；prefill/并发待优化 |
| 3 Q3_K native | 通过 | `20260910-0640-S3-01` | canonical/kernel/native/fallback E2E 通过 | 权重少约 0.57 GB/rank | A/B 已记录 | 仅使用 6/7 | decode 基本持平；获得显存/KV 收益 |
| 4 IQ3_S native | 通过 | `20260910-S4-E2E` | canonical/kernel/native/fallback E2E通过 | 权重少约0.26 GB/rank | decode基本持平 | 仅使用6/7；30000未操作 | 新增四类型0 fallback |
| 5 全量收口 | 通过（保留算术例外） | `20260910-S5-delivery-close` | 314请求持续门禁及完整回归通过 | 新增四类型0 fallback | 最终矩阵完成；Qwen3.6恢复35.01 tok/s | 30001已停，30000未操作 | 精确显存/产物/清理完成 |
| 6 可选性能优化 | 未开始（后续可选） | — | N/A | N/A | I-009回归修复已完成 | N/A | 不阻塞本次原生支持交付 |

## 3. 已确认事实

### 3.1 环境与代码

| 项目 | 当前值 |
|---|---|
| SGLang 宿主机仓库 | `/home/intel/shaojun/sglang/sglang` |
| SGLang origin | `https://github.com/analytics-zoo/sglang` |
| SGLang branch / commit | `feature/qwen3.8-gguf-xpu` / `998a68870`（运行逻辑51a0d849b，后续仅测试脚本；另有最终文档提交） |
| SGLang upstream base | `origin/dev-bmg` / `66861ee2e0c485c4d34d1de56787ddfdf3fd2895` |
| llm-scaler 宿主机仓库 | `/home/intel/shaojun/sglang/llm-scaler` |
| llm-scaler origin | `https://github.com/intel/llm-scaler.git` |
| llm-scaler branch / commit | `feature/qwen3.8-gguf-xpu` / `0a08e20`（native IQ3_S 及 GDN 有序状态更新） |
| llm-scaler upstream base | `origin/main` / `5e2fea9596146af6e90038365ebd462ef59f5d23` |
| 容器 | `sglang-dev-gguf` |
| 宿主机 `gguf.py` | `/home/intel/shaojun/sglang/sglang/python/sglang/srt/layers/quantization/gguf.py` |
| 容器 SGLang build source | `/llm-scaler/sglang/sglang` |
| 容器默认 SGLang runtime | `/usr/local/lib/python3.12/dist-packages/sglang` |
| host/build-source/runtime `gguf.py` 初始 SHA256 | `a0d552d5f780a4382313907a586ec5aad19fd51f2b3b75cd8cd738a19397f929` |
| ESIMD kernel 宿主机源码 | `/home/intel/shaojun/sglang/llm-scaler/sglang/custom-esimd-kernels` |
| ESIMD kernel 容器源码 | `/llm-scaler/sglang/custom-esimd-kernels` |
| ESIMD 默认 runtime package | `/usr/local/lib/python3.12/dist-packages/custom_esimd_kernels_sglang` |
| ESIMD core `.so` 初始 SHA256 | `26a26d3083c6d234eb4f2fc04565574d1a782f3d6c70b7dbe56493a2e6d4258b` |
| sgl-kernel-xpu 容器 build source | `/llm-scaler/sglang/sgl-kernel-xpu` |
| sgl-kernel 默认 runtime package | `/usr/local/lib/python3.12/dist-packages/sgl_kernel` |
| 测试设备 | XPU 6、7 |
| TP | 2 |
| 测试端口 | 30001 |
| 既有服务 | 30000 在阶段 1 开始前已由外部停止，后续各 Run 前后均确认不存在；因此没有可恢复实例 |
| 非任务设备 | XPU 4、5，不得被本任务使用 |

### 3.2 模型

| 项目 | Qwen3.6-27B | Qwen3.8-27B |
|---|---:|---:|
| 主层数 | 64 | 64 |
| GDN / full attention | 48 / 16 | 48 / 16 |
| hidden size | 5120 | 5120 |
| GGUF tensor 数 | 851 | 866 |
| GGUF 文件大小 | 16,817,244,384 bytes | 16,464,440,224 bytes |
| 额外 tensor | — | `blk.64` 的 15 个 MTP tensor |

- 851 个共有 tensor 的名称和 shape 一致。
- 382 个共有 tensor 的量化类型不同。
- Qwen3.8 类型分布：F32 360、IQ3_S 4、IQ4_NL 7、IQ4_XS 117、Q3_K 7、Q4_K 104、Q5_K 131、Q6_K 30、Q8_0 106。
- 普通非 speculative 启动不设置 `SPEC_DRAFT_PATH`。

### 3.3 当前技术状态

- Qwen3.6 GGUF 已验证可以启动和推理。
- `scripts/start_qwen3_6_service.sh` 的 GGUF 分支已有 `--skip-server-warmup`。
- Qwen3.8 首次启动在 `_xpu_prepare_shard()` 失败：`AssertionError: q4_k GDN out_proj col-perm unsupported`。
- 触发的 Q4_K `ssm_out`：`blk.14/22/29/38/45.ssm_out.weight`。
- TP2 预期 `col_perm=(3,8,128)`。
- IQ4_XS、IQ4_NL、IQ3_S、Q3_K 均已有原生压缩态常驻；保留逐类型 FP16 fallback 用于正确性与性能对照。
- Q4_K col-perm 与 large-FP16 transpose cache 修复已提交为 `3d8d99ecf`；IQ4 canonical 已提交为 `f0958175d`；阶段 2B native kernel、dispatch 和 E2E 已完成。

### 3.4 任务开始时的代码 provenance 与 dirty baseline

- 三个私有/公开 Git remote 均已通过 `git ls-remote` 读取，不需要在容器内 `gh login`。
- SGLang 当前任务 branch 比 `origin/dev-bmg` 多一个基线 commit：`b4599c0ef xpu: disable GGUF kernel probes by default`。
- llm-scaler 当前任务 branch 比 `origin/main` 多两个基线 commit：`3ef4b18 SGL: refresh BMG downstream patches`、`1ee060e sglang: enable GGUF fusions for GGUF services`。
- llm-scaler 尚有一处任务开始前已存在的未提交修改：`sglang/scripts/start_qwen3_6_service.sh` 增加 `--skip-server-warmup`。
- SGLang 工作区中的两份 Qwen3.8 文档是当前会话新增的未跟踪文件；功能源码尚未修改。
- `custom-esimd-kernels` 是 llm-scaler repo 内的 tracked 目录，不是独立 Git repo/submodule。
- 当前 llm-scaler Dockerfile 实际用 `sgl-project` 固定版本/commit 加 patch 构建 SGLang 和 sgl-kernel-xpu，不直接从 analytics-zoo fork 构建。
- 容器默认 `PYTHONPATH` 为空；默认导入 site-packages。后续 30001 必须使用显式 source/wheel overlay，并记录模块 `__file__`。

## 4. 初始观测记录

### Run `20260910-0436-S0-01`

目的：创建文档前进行只读环境快照。没有同步文件、停止服务或启动 30001。

#### 版本与文件

| 项目 | 结果 |
|---|---|
| UTC 采样时间 | 2026-09-10 04:36 左右 |
| repo commit | `b4599c0ef327` |
| 原有 tracked/untracked 修改 | `git status --short` 无输出（创建本文档之前） |
| GGUF 文件大小 | 16,464,440,224 bytes |
| 宿主机 `gguf.py` SHA256 | `a0d552d5f780a4382313907a586ec5aad19fd51f2b3b75cd8cd738a19397f929` |
| 容器 `gguf.py` readlink | `/llm-scaler/sglang/sglang/python/sglang/srt/layers/quantization/gguf.py` |
| 容器 `gguf.py` SHA256 | `a0d552d5f780a4382313907a586ec5aad19fd51f2b3b75cd8cd738a19397f929` |
| 容器状态 | running，PID 807459，started `2026-09-10T01:39:11.061023738Z` |

#### 端口与健康

| 检查 | 结果 | 判断 |
|---|---|---|
| 30000 listen | `0.0.0.0:30000` 正在监听 | 仅证明监听存在 |
| `GET 30000/health` | 5 秒超时，HTTP 000 | 未通过，需在开始开发前调查/重测 |
| 30001 listen | 无 | 符合尚未启动测试服务的状态 |
| `GET 30001/health` | connection refused，HTTP 000 | 符合尚未启动 |

注意：30000 健康超时可能是服务忙、请求路径行为或真实异常；当前证据不足，不能判断服务健康，也不能把阶段 0 标记为通过。下一次应记录其服务 PID/命令行并延长超时重测。

#### XPU 瞬时显存

`xpu-smi dump` 连续采样中的首个样本：

| Device | Memory Used (MiB) | Memory Utilization |
|---:|---:|---:|
| 0 | 229.13 | 0.70% |
| 1 | 229.11 | 0.70% |
| 2 | 864.40 | 2.64% |
| 3 | 987.20 | 3.02% |
| 4 | 987.21 | 3.02% |
| 5 | 987.28 | 3.02% |
| 6 | 32648.77 | 99.97% |
| 7 | 32639.61 | 99.94% |

该数据是单次只读快照，不是正式 E2E 的五时点显存结果。卡 6、7 高显存与现有 30000 服务相符。本次采样没有操作该服务；后续设备策略在 Run `20260910-0448-S0-03` 中修订为“受控停止、测试后恢复”。

#### 结论

- 文档化开始时宿主机与容器 `gguf.py` 一致。
- 30001 未运行。
- 30000 有 listener，但健康请求超时，因此阶段 0 仍为“进行中”。
- 本次没有产生性能数据，也没有验证模型输出。

### Run `20260910-0446-S0-02`

目的：核对代码来源、容器实际 import 位置，并建立任务分支。没有修改功能源码、安装 wheel、同步容器或启动服务。

#### Git remote 与分支

| Repo | Remote/base | 任务分支起点 | 本次动作 | 当前未提交内容 |
|---|---|---|---|---|
| SGLang | `analytics-zoo/sglang:dev-bmg` `66861ee2e` | `b4599c0ef` | 创建并切换到 `feature/qwen3.8-gguf-xpu` | 两份本任务文档 |
| llm-scaler | `intel/llm-scaler:main` `5e2fea959` | `1ee060ea0` | 创建并切换到 `feature/qwen3.8-gguf-xpu` | 既有 `--skip-server-warmup` 一行 |
| sgl-kernel-xpu | `analytics-zoo/sgl-kernel-xpu:dev-bmg` `fd89d0fde` | N/A | 只读 `ls-remote`，未 clone/建 branch | N/A |

分支切换没有修改文件内容。当前分支是“已验证基线上的任务分支”，不是裸 upstream；开始 Q4_K 功能代码前仍需把文档和既有启动脚本变更分开保存，使 working tree clean。

#### 容器路径解析

| 检查 | 结果 |
|---|---|
| 容器 `PYTHONPATH` | 空 |
| `sglang` 默认 import | `/usr/local/lib/python3.12/dist-packages/sglang/__init__.py` |
| `gguf.py` 默认 runtime | `/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/gguf.py` |
| SGLang build source | `/llm-scaler/sglang/sglang/python/sglang/srt/layers/quantization/gguf.py` |
| runtime/build-source `gguf.py` SHA256 | 均为 `a0d552d5f780a4382313907a586ec5aad19fd51f2b3b75cd8cd738a19397f929` |
| ESIMD 默认 import | `/usr/local/lib/python3.12/dist-packages/custom_esimd_kernels_sglang/__init__.py` |
| ESIMD core `.so` | `/usr/local/lib/python3.12/dist-packages/custom_esimd_kernels_sglang/custom_esimd_kernels.cpython-312-x86_64-linux-gnu.so` |
| ESIMD core `.so` SHA256 | `26a26d3083c6d234eb4f2fc04565574d1a782f3d6c70b7dbe56493a2e6d4258b` |

关键结论：只复制 build-source `gguf.py` 不足以保证默认服务加载它。后续使用 `/llm-scaler/sglang/sglang/python` 加独立 ESIMD target overlay，并在启动前/服务进程内验证实际模块路径。

#### 结论

- Git 读取权限已经足够，无需向容器注入 GitHub 凭据。
- 两个任务分支已经创建，但 clean-baseline gate 尚未通过。
- 本次无模型正确性、显存或性能结果。

### Run `20260910-0448-S0-03`

目的：记录用户对测试设备的最终选择。本次仅更新计划和日志，没有停止 30000、启动 30001 或运行模型。

#### 决策

- 测试设备从 XPU 4、5 改为 XPU 6、7，`TP_SIZE=2` 不变。
- 30001 仍为测试端口。
- 30000 当前占用同一组 XPU 6、7，不能与 30001 并行运行。
- 每次 E2E 前必须完整保存 30000 现场并精确停止；测试完成后按原命令和环境恢复，并复测健康与输出。
- XPU 4、5 不用于本任务，测试前后检查其显存没有来自本任务的变化。

#### 当前状态

- 30000 尚未停止。
- 30001 尚未启动。
- 未产生正确性、显存或性能新数据。
- 阶段 0 仍为“进行中”：必须先解决/解释 30000 `/health` 超时，并记录可复现的停止与恢复方法。

### Run `20260910-0451-S0-04`

目的：开始阶段 1 前复查端口和 XPU 状态。

- 30000、30001 均无 listener。
- `sglang-dev-gguf` 内没有 SGLang 模型服务进程。
- 30000 在本任务采取停止动作前已经被外部停止，因此本轮没有可记录或可恢复的服务进程。
- XPU 6/7 分别约 228.96/228.82 MiB，可用于本任务。
- SGLang 文档基线已提交为 `5034fd1b6`。
- llm-scaler 的既有 `--skip-server-warmup` 已独立提交为 `4f47c57`。

结论：阶段 0 通过。本轮 30000 停止/恢复项明确记为 N/A。

### Run `20260910-0452-S1-01`

目的：验证 Q4_K GDN `ssm_out` 压缩态 `col_perm`。

#### 实现

- `_xpu_repack_q4_k(qweight, col_perm=None)` 在 nibble element order 中重排。
- scale/min 以 `head_v_dim // 32` 粒度重排。
- chunked wrapper 传递 `col_perm`。
- `_xpu_prepare_shard()` 的 Q4_K 分支传递 `col_perm`，删除旧 assert。
- 对 `head_v_dim % 32` 和 `ratio * nk * head_v_dim == K` 增加显式检查。

#### Synthetic 测试

命令：

```bash
PYTHONPATH=/llm-scaler/sglang/sglang/python \
python3 -m pytest -q \
  test/registered/unit/layers/quantization/test_gguf_xpu_q4_k_repack.py
```

结果：首次 Q4-only 测试 `5 passed`；加入 large-FP16 cache 回归后为 `7 passed`。覆盖：

- 默认参数与显式 `col_perm=None` 一致；
- packed nibble、scale、min 与 element-order reference 一致；
- repack/dequant 与 dense permutation bitwise 一致；
- chunked 与 non-chunked 一致；
- 非 32 对齐和 K 不匹配会明确失败；
- 大型 FP16 fallback 不进入 transpose cache，小型 FP16 shard 仍缓存。

#### 五个实际 tensor 全量验证

命令：

```bash
PYTHONPATH=/llm-scaler/sglang/sglang/python \
python3 test/manual/quant/validate_qwen3_8_q4_k_col_perm.py \
  /models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --all-rows --row-chunk 256
```

每个 tensor 均为 `[N=5120,K=6144]`，TP2 local K=3072，`col_perm=(3,8,128)`。以下误差同时覆盖 base repack 与 rank0/rank1 组合路径；mean abs 均约 `8e-6`：

| Tensor | 全路径最坏 max abs |
|---|---:|
| `blk.14.ssm_out.weight` | 0.000371933 |
| `blk.22.ssm_out.weight` | 0.000403404 |
| `blk.29.ssm_out.weight` | 0.000268936 |
| `blk.38.ssm_out.weight` | 0.000431061 |
| `blk.45.ssm_out.weight` | 0.000270844 |

总体最坏 `max_abs=0.000431061 < 0.0005`，五个 tensor 全部 5120 行通过。

### Run `20260910-0458-S1-02`

结论：失败，但已定位并修复。

- PID：11630。
- 使用 `PYTHONPATH=/llm-scaler/sglang/sglang/python`，确认实际加载新 `gguf.py`。
- 模型加载成功，Q4_K assert 已消失。
- 权重加载耗时 TP0/TP1 为 89.14/87.79 秒；每 rank 报告权重占用 22.08 GB、剩余 9.81 GB。
- cache 分配后剩余 6.32 GB；XPU 6/7 观测显存约 27,518.98/27,154.44 MiB。
- 首次 `/health` 触发 forward 后，在 `_xpu_shard_matmul()` 的 `w.t().contiguous()` 报 `UR_RESULT_ERROR_OUT_OF_RESOURCES`，服务退出。

根因：IQ4/Q3/IQ3 当前已作为大矩阵 dense FP16 常驻，原 `_fp16_wt_cache` 又逐层永久保存 contiguous transpose，形成第二份大权重并在首次 forward 中累积。这个 cache 原本只为小型 GDN a/b FP16 shard 设计。

修复：新增默认 16 MiB 的 `SGLANG_GGUF_XPU_FP16_WT_CACHE_MAX_BYTES` 上限；大矩阵直接把 transpose view 交给 `torch.mm`，不进入永久 cache，小矩阵保持原优化。

### Run `20260910-0502-S1-03`

结论：阶段 1 通过。

#### 环境与产物

| 项目 | 值 |
|---|---|
| 服务 PID | 12937 |
| 日志 | `/tmp/qwen3_8_q4_stage1_20260910_0503.log`（容器内） |
| 日志 SHA256 | `b42c5d21ebcbeceea0aacb92e174cd941a1e4c8496012842869f3a6bfd44ccb0` |
| `gguf.py` SHA256 | `bb08e06578a041b9e7bc293679cc3a4a5d5a167b36a2bea72adb8c41acd7b0aa` |
| SGLang import | `/llm-scaler/sglang/sglang/python/sglang/.../gguf.py` |
| ESIMD import | system `custom_esimd_kernels_sglang`，Q4_K symbol 可用 |
| 设备/TP/端口 | XPU 6、7 / TP2 / 30001 |

#### 启动和正确性

- 权重加载耗时 TP0/TP1：88.01/86.91 秒。
- 权重阶段每 rank：22.08 GB，剩余 9.81 GB。
- memory pool 后每 rank 剩余 6.32 GB。
- `/health`：HTTP 200，首次 5.015 秒，warm 后 1.002 秒。
- `/v1/models`：HTTP 200，2.7 ms，model id 为 `/models/Qwen3.8-27B`。
- thinking 模式 `max_tokens=128`：`1+1 -> 2`，`17*23-19 -> 372`。
- non-thinking 模式：严格 JSON 可解析且为 `{"answer":7,"ok":true}`；完整素数函数逻辑正确；标准大气压水的冰点回答 `0℃`。
- 长上下文：实际 `prompt_tokens=2952`，正确提取 `BLUE-7391`。
- non-thinking 256-token 连续 decode：顺序数字输出正常，无乱码/异常重复，因长度限制结束。
- thinking 模式 `max_tokens=32` 时 reasoning 用尽 token、最终 content 为空；这是请求预算问题，不是模型数值错误，正确性测试后续固定 `max_tokens>=128` 或关闭 thinking。

#### 显存

| 时点 | XPU 6 | XPU 7 |
|---|---:|---:|
| 启动前 | 228.96 MiB | 227.01 MiB |
| 加载完成、首次 forward 前 | 27,518.98 MiB | 27,154.44 MiB |
| 首次 forward 后 | 31,704.81 MiB | 30,953.31 MiB |
| 长 prefill/并发后 | 31,762.81 MiB | 31,009.40 MiB |
| 停止 30001 后 | 228.96 MiB | 227.01 MiB |

显存余量非常小，但多轮请求后没有继续线性增长。阶段 2 的 native IQ4 必须明显降低该值。

#### 基础性能

性能固定 `temperature=0`、`enable_thinking=false`，以下为当前大量 FP16 fallback 下的阶段基线：

| 场景 | Run 1 | Run 2 | Run 3 | 中位数 |
|---|---:|---:|---:|---:|
| short TTFT | 0.9738 s | 0.9621 s | 0.9621 s | 0.9621 s |
| decode 256 e2e | 21.831 tok/s | 21.952 tok/s | 22.060 tok/s | 21.952 tok/s |
| 1462-token prefill | 935.643 tok/s | 1041.434 tok/s | 1041.966 tok/s | 1041.434 tok/s |

| 并发 | completion tokens | wall time | 聚合吞吐 | HTTP 成功 |
|---:|---:|---:|---:|---:|
| 2 | 256 | 8.3015 s | 30.838 tok/s | 2/2 |
| 4 | 512 | 8.2563 s | 62.013 tok/s | 4/4 |

#### 清理与隔离性

- 仅对记录的 PID 12937 发送 SIGTERM，服务 graceful exit。
- 30001 已释放；XPU 6/7 回落到 228.96/227.01 MiB。
- 本轮开始前 30000 已由外部停止，因此未执行停止或恢复操作。
- 服务显式设置 `ZE_AFFINITY_MASK=6,7`；没有把本任务进程放到卡 4/5。

#### 已知限制与下一步

- IQ4_NL、IQ4_XS、Q3_K、IQ3_S 仍为 dense FP16 fallback。
- 峰后显存达到 97.26%/94.95%，仅适合作为过渡正确性路径。
- 日志中的部分 fusion 因混合 quant type 不适用，这是阶段 2 后需要重新评估的性能现象。
- 下一步进入阶段 2：IQ4_NL/IQ4_XS canonical repack、native kernel 和完整 E2E A/B。

### Run `20260910-0519-S2-01`

目的：完成 IQ4_NL/IQ4_XS canonical repack、共享 ABI、reference dequant 和 IQ4_XS GDN TP2 `col_perm` 验证；本 Run 尚未接入 native kernel/dispatch。

关联阶段：阶段 2A。

结论：局部验证通过，E2E fallback 回归另记后续 Run。

#### 实现与 ABI

- IQ4_NL：解析 32 元素/18 字节 raw block，输出相邻元素 packed LUT index `[N,K/2] uint8` 与 `[N,K/32] fp16` scale。
- IQ4_XS：解析 256 元素/136 字节 raw super-block，将 high/low bits 合成 signed 6-bit subscale，并预计算同一 `[N,K/32] fp16` final scale。
- 两类 resident rep 统一使用 `weight = final_scale * IQ4_LUT[index]`；LUT 为 `[-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113]`。
- `col_perm=(3,8,128)` 同步排列 element index 和每 head 4 个 scale group；`head_v_dim` 必须被 32 整除。
- 当前代码仅定义 canonical repack/dequant，不进入 `_xpu_prepare_shard()`，所以服务仍使用原 FP16 fallback。

#### 代码和产物

| 项目 | 值 |
|---|---|
| UTC 完成时间 | 2026-09-10 05:19 UTC |
| SGLang commit | `3d8d99ecf` + 未提交阶段 2A diff |
| llm-scaler commit | `4f47c5783f19`，clean |
| host/build-source `gguf.py` SHA256 | `5d7a6148e615afaff5e9d4867aed5a1190be8941c83857833dfb6a911890eda3` |
| runtime import | `/llm-scaler/sglang/sglang/python/sglang/srt/layers/quantization/gguf.py`，已确认新 symbol 存在 |
| unit test | `test/registered/unit/layers/quantization/test_gguf_xpu_iq4_repack.py` |
| actual-weight validator | `test/manual/quant/validate_qwen3_8_iq4_repack.py` |
| validator 日志 | 宿主机 `/tmp/qwen3_8_iq4_canonical_20260910.log` |
| validator 日志 SHA256 | `ddbba89c26cff30b0f2d8132144934d0a98e7e73d28448dd4c07f8c2c3023e64` |

#### 测试结果

- 第一次宿主机 pytest 因未设置 `PYTHONPATH` 无法 import `sglang`；设置后又因宿主机环境缺 `orjson` 无法收集。因此正式单测在容器开发源码环境执行。
- 容器第一次单测失败 10 项，根因仅为 synthetic fixture 对 0 维 FP16 tensor 执行 byte reinterpret；改为一元素 tensor 后修复。
- 修复后单测：`12 passed`，覆盖 reference、共享 ABI、row chunk、两类 `col_perm`、非法 shape。
- 实际权重：确认并验证 `IQ4_NL=7`、`IQ4_XS=117`；所有 tensor 验证首/中/末行。
- 五个 IQ4_XS `ssm_out` 全部 5120 行均验证 base、TP rank0、TP rank1，共覆盖各 tensor base 31,457,280 元素、每 rank 15,728,640 元素。
- IQ4_NL 与 GGUF reference bit-exact：`max_abs=0`。
- IQ4_XS 总体最坏 `max_abs=0.0002422333 < 0.0005`，mean abs `1.4001e-6`；五个 `ssm_out` 各路径最坏值不超过该值。
- 本 Run 未启动服务、未使用 XPU，30000/30001 均无 listener，卡 4/5 未使用。

### Run `20260910-0521-S2-02`

目的：阶段 2A 结束时执行一次完整 fallback 服务回归，确保尚未接 dispatch 的 canonical 辅助代码不改变既有推理路径，并为后续 native IQ4 A/B 保留同机基线。

关联阶段：阶段 2A E2E 门禁。

结论：通过；存在一条关闭 thinking 的额外算术请求答错，按阶段 1 同条件开启 thinking 后正确，已保留原始现象。

#### 环境

| 项目 | 值 |
|---|---|
| UTC 开始/结束 | 2026-09-10 05:21 / 05:25 |
| 服务 PID | 15516 |
| 设备/TP/端口 | `ZE_AFFINITY_MASK=6,7` / TP2 / 30001 |
| source overlay | `PYTHONPATH=/llm-scaler/sglang/sglang/python` |
| 模型/GGUF config | `/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf` / `/models/Qwen3.8-27B` |
| 日志 | 容器 `/tmp/qwen3_8_iq4_stage2a_fallback_20260910.log` |
| 日志 SHA256 | `2b97f6feef6abe2f8a0719318523b90efac798dce605097fdc7822454e5ee29e` |

#### 启动与正确性

- TP0/TP1 load weight：87.98/90.31 秒；各 rank 权重仍为 22.08 GB，证明 IQ4 仍走 dense FP16 fallback。
- `/health` 首次 forward：HTTP 200，5.0138 秒；`/v1/models`：HTTP 200，model id 正确。
- non-thinking：`1+1 -> 2`；严格 JSON 得到 `{"answer":7,"ok":true}`；1428-token marker 三次均得到 `BLUE-7391`。
- 额外 non-thinking 请求 `17*23-19` 得到错误的 `272`；同题按阶段 1 条件启用 thinking、`max_tokens=128` 后得到正确 `372`。
- 第二条 thinking 逐步计算请求的 reasoning 正确，但 128 token 用尽导致最终正文截断；这是已知 token budget 行为，后续 thinking 正确性用例至少给 256 token。
- canonical helper 尚未进入 `_xpu_prepare_shard()`，因此上述波动不能由新 IQ4 repack 数值导致。

#### 显存

| 时点 | XPU 6 | XPU 7 |
|---|---:|---:|
| 启动前 | 45.61 MiB | 45.33 MiB |
| 加载后、首次 forward 前 | 27,335.65 | 26,970.91 |
| 首次 forward 后 | 28,581.37 | 27,831.72 |
| decode/prefill 后 | 31,564.62 | 30,812.85 |
| 停止后 | 45.61 | 45.33 |

#### 简单性能回归

固定 `temperature=0`、`enable_thinking=false`：

| 场景 | Run 1 | Run 2 | Run 3 | 中位数 |
|---|---:|---:|---:|---:|
| streaming 首个 content TTFT | 0.0982 s | 0.1307 s | 0.1212 s | 0.1212 s |
| decode 256 e2e | 20.822 tok/s | 20.820 tok/s | 21.222 tok/s | 20.822 tok/s |
| 1428-token 请求端 prompt/e2e | 725.70 tok/s | 1145.81 tok/s | 1152.87 tok/s | 1145.81 tok/s |

前两次后缀相同的 prefill 命中 radix prefix cache，因此该数字只作为后续使用完全相同脚本的 A/B 基线，不能当作纯 uncached prefill kernel 吞吐。

#### 清理与隔离性

- 仅向记录的 PID 15516 发送 SIGTERM，服务 graceful exit。
- 30000 在启动前已不存在，未停止也无需恢复；30001 已释放。
- 卡 4/5 未被本任务使用。

### Run `20260910-0526-S2-03`

目的：完成 IQ4_NL/IQ4_XS 原生 ESIMD kernel、SGLang dispatch 集成和 native 端到端验证。

关联阶段：阶段 2B native 路径。

结论：局部数值、服务加载、串行正确性、长上下文和资源回收通过；并发 2 发现一路数字序列退化，保留到 forced-fallback A/B 判断。

#### 代码、产物与局部验证

| 项目 | 值 |
|---|---|
| UTC 测试区间 | 2026-09-10 05:26～06:03 |
| SGLang branch/base | `feature/qwen3.8-gguf-xpu` / `f0958175d`，集成 diff 尚未提交 |
| llm-scaler branch/最终 kernel commit | `feature/qwen3.8-gguf-xpu` / `03ccfa3` |
| `gguf.py` SHA256 | `2075a19d3ecf55aaa244b80ca177df5bd39ef2f211e87f855cc1d1dfc00fad52` |
| wheel | `/llm-scaler/sglang/custom-esimd-kernels/dist/custom_esimd_kernels_sglang-0.1.0-cp312-cp312-linux_x86_64.whl` |
| wheel SHA256 | `3cc2fc247645c29491fc6be54b94c2a3c40087190e33d8edac02a78ab5fc7890` |
| overlay | `/llm-scaler/overlays/qwen3_8` |
| runtime core `.so` SHA256 | `6ac73a06bb82000ca2dfcd2acc95d5d2c6d73e3ea3998a89930c2e353da413c5` |
| runtime `PYTHONPATH` | `/llm-scaler/overlays/qwen3_8:/llm-scaler/sglang/sglang/python` |
| service PID / 设备 / 端口 | 23568 / XPU 6、7 TP2 / 30001 |
| 服务日志 / SHA256 | `/tmp/qwen3_8_iq4_native_20260910.log` / `85db9b13553b20fec4b9d960c4ddc0fa077735daa3c4fc10978fa15d4475ea56` |
| benchmark JSONL / SHA256 | `/tmp/qwen3_8_iq4_native_bench_20260910.jsonl` / `ceea8806c5a7374a4144e6df919b8c01cd2d5ac94dd159c788a46de0a36f870f` |

实现覆盖：

- `esimd_gemv_iq4` 和 `esimd_gemv_iq4_m` 共用 canonical packed LUT-index + final-scale ABI。
- 接入 prepare、M=1/M<=16 matmul、大 M dense reconstruction、row permutation、same-kind merge、mixed-kind group 和非连续 output slice。
- `SGLANG_GGUF_XPU_NO_IQ4=1` 仅回退 IQ4；IQ4 kernel symbol 单独 optional import，不影响旧 kernel。
- kernel synthetic：`20 passed`，覆盖 M=1/2/4/8/16，另覆盖 M=3/17、K=96 fallback、K=256/512/3072 和非连续 output slice。
- SGLang repack/dispatch：`16 passed`，覆盖两种 IQ4、类型专属 fallback、row permutation、same-kind merge 和 mixed-kind group。
- 使用不含 IQ4 symbol 的系统旧 `.so` 导入新 `gguf.py`：`esimd_gemv_iq4[_m] is None`，同时 Q4_K/Q5_K/Q6_K symbol 均仍可用，证明 optional import 不会连带关闭旧 kernel。
- 真实 GGUF kernel validator：23 cases，9 种代表 tensor shape；五个 IQ4_XS `ssm_out` 全 5120 行、TP rank0/rank1；最坏 `max_abs=0.001953125 < 0.01`，全部 finite。
- validator 日志：`/tmp/qwen3_8_iq4_native_kernel_validation_20260910.log`，SHA256 `2165db819afa29184f52ccc2ff746b19e106e52af163c192d66c7d309ac62a46`。

#### 加载、正确性和显存

- TP0/TP1 权重加载：43.97/43.70 秒；每 rank 权重 12.48 GB、剩余 19.41 GB。
- KV capacity 为 350,016 tokens；对照 dense IQ4 fallback 为 35,136 tokens。
- `/health` HTTP 200；`1+1 -> 2`；thinking 256-token 的 `17*23-19 -> 372`；严格 JSON 为 `{"answer":7,"ok":true}`。
- 固定 2836-token marker 三次均返回 `BLUE-7391`；18,935-token 长上下文返回 `ORANGE-86420`。
- 256-token 连续 decode 三次均为 HTTP 200、`finish_reason=length`，数字序列连续无乱码。

| 时点 | XPU 6 | XPU 7 |
|---|---:|---:|
| 启动前 | 约 45.61 MiB | 约 45.33 MiB |
| 加载后、首次 forward 前 | 27,322.68 | 26,957.96 |
| 首次 forward 后 | 28,450.53 | 27,700.54 |
| 256-token decode 后 | 30,756.36 | 30,006.50 |
| 2836-token marker 后 | 31,137.39 | 30,385.64 |
| 并发后 | 31,151.10 | 30,399.34 |
| 18,935-token 长上下文后 | 32,655.89 | 32,272.40 |
| 停止后 | 45.85 | 47.75 |

固定 `mem-fraction-static=0.8` 会把 native IQ4 释放的权重空间继续分配给 KV cache，所以压力后的 `xpu-smi` 总占用与 fallback 接近；真实收益体现在权重从 22.08 降到 12.48 GB/rank，以及 KV capacity 约 9.96 倍。

#### 性能和并发现象

| 场景 | Run 1 | Run 2 | Run 3 | 中位数 |
|---|---:|---:|---:|---:|
| streaming content TTFT | 1.3876 s | 1.3878 s | 1.3871 s | 1.3876 s |
| 256-token decode | 23.1018 tok/s | 23.1701 tok/s | 23.1237 tok/s | 23.1237 tok/s |
| 2836-token marker prompt/e2e | 1011.67 tok/s | 1747.49 tok/s | 1747.79 tok/s | 1747.49 tok/s |

- concurrency 2：256 completion tokens / 8.0873 秒，31.6545 tok/s。
- concurrency 4：512 completion tokens / 9.0866 秒，56.3469 tok/s。
- 18,935-token 长上下文端到端 10.7735 秒。
- 并发 2 首轮有一路出现重复数字；追加三轮时，编号 0 请求均出现不同程度粘连/重复或提前结束，而相同编号 0/1 请求串行均正确。该现象在下一 Run 的 forced fallback 中复现。

#### 清理与隔离性

- 仅向 PID 23568 发送 SIGTERM，约 5 秒退出；30001 释放，XPU 6/7 回落。
- 30000 在本轮开始前不存在，无需恢复。
- 服务环境固定 `ZE_AFFINITY_MASK=6,7`。卡 4/5 上有其他既存占用，但本任务没有绑定或启动进程到卡 4/5。

### Run `20260910-0603-S2-04`

目的：通过 `SGLANG_GGUF_XPU_NO_IQ4=1` 做同代码、同 wheel、同启动参数的 forced-fallback A/B，并判断 native 并发异常是否为 IQ4 专属。

关联阶段：阶段 2B fallback 门禁；Issue I-005、I-006。

结论：通过。fallback 正确性与长上下文通过；并发退化同样可复现，排除 IQ4 native 为唯一原因；阶段 2 可以结束，但大 M/prefill 与并发性能留待 profiling。

#### 启动过程和环境

- 第一次手写启动遗漏 `SGLANG_GGUF_HF_CONFIG_DIR`，在解析 GGUF `qwen35` config 时退出，尚未加载权重。
- 第二次补 config 但仍遗漏 `SGLANG_MAMBA_CONV_DTYPE=float16` 和 `SGLANG_MAMBA_SSM_DTYPE=float16`；权重加载完成后默认 FP32 SSM state 使 memory-pool 可用字节为负，报 `Not enough memory`。日志 `/tmp/qwen3_8_iq4_forced_fallback_20260910.log`，SHA256 `93e41eb0f5ef04bd3eed9884047da1117927f5074000488698e6cbc20ba6bdeb`。
- 最终直接使用 `scripts/start_qwen3_6_service.sh` 复现完整 GGUF 环境，只额外设置 `SGLANG_GGUF_XPU_NO_IQ4=1`。PID 27908，日志 `/tmp/qwen3_8_iq4_forced_fallback_20260910_run2.log`，SHA256 `2c82874458b9f9598aff5922a06bbdecd0a7e2d6d8764bb7704ff8dc7213038e`。
- benchmark JSONL `/tmp/qwen3_8_iq4_forced_fallback_bench_20260910.jsonl`，SHA256 `e3183127a7ec828b4c9664a4ca23da61fcb5ab6c8f3346363dea532cccd2c5bf`。
- TP0/TP1 权重加载：86.21/87.17 秒；每 rank 22.08 GB；KV capacity 35,136 tokens；首次 `/health` HTTP 200、5.011 秒。

#### 正确性、显存和 A/B

- `1+1 -> 2`，严格 JSON 为 `{"answer":7,"ok":true}`。
- 2836-token marker 三次均返回 `BLUE-7391`；18,935-token 长上下文返回 `ORANGE-86420`。
- 三次 256-token decode 均连续、无乱码并以长度限制结束。
- concurrency 2 首轮一路数字粘连；追加三轮也分别出现一路粘连/重复或提前结束，另一条正常。concurrency 4 首轮四条均正常。这与 native 的并发现象同类，因此登记为共同的模型/批调度问题，不归因于 IQ4 kernel。

| 时点 | XPU 6 | XPU 7 |
|---|---:|---:|
| 启动前 | 45.85 MiB | 47.75 MiB |
| 首次 health 后 / benchmark 前 | 28,579.43 | 27,831.58 |
| 256-token decode 后 | 31,511.73 | 30,763.65 |
| 2836-token marker 后 | 31,552.95 | 30,803.29 |
| 并发后 | 31,588.65 | 30,838.99 |
| 18,935-token 长上下文后 | 32,654.97 | 32,207.12 |
| 停止后 | 43.43 | 45.33 |

| 场景 | native 中位数/结果 | forced fallback 中位数/结果 | native 相对变化 |
|---|---:|---:|---:|
| 权重加载 | 43.84 s | 86.69 s | -49.4% |
| resident weight/rank | 12.48 GB | 22.08 GB | -9.60 GB |
| streaming content TTFT | 1.3876 s | 0.9613 s | +44.3% 延迟 |
| 256-token decode | 23.1237 tok/s | 21.7367 tok/s | +6.4% |
| 2836-token marker prompt/e2e | 1747.49 tok/s | 2308.74 tok/s | -24.3% |
| concurrency 2 aggregate | 31.6545 tok/s | 32.9790 tok/s | -4.0% |
| concurrency 4 aggregate | 56.3469 tok/s | 66.0749 tok/s | -14.7% |
| 18,935-token marker e2e | 10.7735 s | 9.3620 s | +15.1% 延迟 |

prefill 重复轮次会命中 radix prefix cache，且 TTFT/e2e 同时包含少量 decode，因此这些数据只适合同脚本 A/B。现阶段结论是 native IQ4 明显改善加载时间、权重显存和可用 KV capacity，并改善 M=1 长 decode；大 M/prefill 和并发没有性能收益，进入阶段 6 前不做 fusion 猜测。

#### 清理与阶段结论

- 仅向 PID 27908 发送 SIGTERM，约 3 秒退出；30001 已释放，XPU 6/7 回落至 43.43/45.33 MiB。
- 30000 在本轮前后均不存在，恢复项为 N/A。
- 卡 4/5 没有被测试进程使用；结束采样为 30,176.69/1,189.45 MiB，是其他既存负载。
- 阶段 2 的 reference、kernel、dispatch、native/fallback E2E、显存和性能门禁均完成，允许进入阶段 3 Q3_K。

### 阶段 2 总结

- 状态：通过。
- commits：llm-scaler kernel `03ccfa3`；SGLang canonical `f0958175d`；SGLang native 集成 `812e55a9a`。
- Reference 数值：124 个实际 IQ4 tensor canonical 验证通过，最坏 `max_abs=0.0002422333`。
- Kernel 数值：23 个真实 case，最坏 `max_abs=0.001953125 < 0.01`，无 NaN/Inf。
- E2E：native/fallback 均加载并通过固定算术、JSON、marker 和 18,935-token 长上下文。
- 显存：native resident weight 少约 9.60 GB/rank，KV capacity 从 35,136 增至 350,016。
- 性能：M=1 decode 中位数 +6.4%；prefill/并发当前退化，留待阶段 6 profiling。
- 阶段 2 结束时的已知限制：Q3_K 与 IQ3_S 仍为 dense FP16 fallback；并发数字序列退化在 native/fallback 均存在。
- 资源：30001 已停止；30000 原本不存在；仅使用 XPU 6、7。

### Run `20260910-0640-S3-01`

目的：完成 Q3_K canonical ABI、原生 ESIMD kernel、SGLang dispatch，以及 native/单类型 fallback 的完整 A/B。

关联阶段：阶段 3 Q3_K 原生纵向切片。

结论：通过。Q3_K 数值、服务加载、串行输出、长上下文、显存回收通过；native 相对 fallback 主要收益是约 0.57 GB/rank 权重显存和 18,624 tokens/rank KV capacity，单路 decode 基本持平。

#### 代码、产物与局部验证

| 项目 | 值 |
|---|---|
| UTC 测试区间 | 2026-09-10 约 06:40～07:09 |
| SGLang branch/commits | `feature/qwen3.8-gguf-xpu`；canonical `be56dd188`；integration `bb76c0bcc` |
| llm-scaler branch/kernel commit | `feature/qwen3.8-gguf-xpu` / `9449de8` |
| `gguf.py` SHA256 | `a362e5f5ab4d8c493c7ed701378246830341ded5540c71925689ba10643e770e` |
| wheel / SHA256 | `/llm-scaler/sglang/custom-esimd-kernels/dist/custom_esimd_kernels_sglang-0.1.0-cp312-cp312-linux_x86_64.whl` / `41ecb47ab904829f07d988eded87a2ed229b6849a96ea22320f95640453c912b` |
| runtime core `.so` SHA256 | `288f7b306e0494e3c34ebe0afcaacec445cb7b01235a32e03cedfcc6f70c4fc5` |
| runtime `PYTHONPATH` | `/llm-scaler/overlays/qwen3_8:/llm-scaler/sglang/sglang/python` |
| native PID / fallback PID | 37862 / 39269 |
| native service log / SHA256 | `/tmp/qwen3_8_q3_native_20260910.log` / `01bab460951a5b8a6caa201ce5b0e6e7785e3f7c607a38e2b8412224c24d3ca6` |
| fallback service log / SHA256 | `/tmp/qwen3_8_q3_fallback_20260910.log` / `914a7ec2884ac63d7dccc3cf682548c013e13bcbf03d39aac707f2b5b1751b05` |
| native benchmark / SHA256 | `/tmp/qwen3_8_q3_native_bench_20260910.jsonl` / `1222fdf847dfd238d6830d1e9329bd3f7c6784dfb4254c22cb9fb5e0673746ed` |
| fallback benchmark / SHA256 | `/tmp/qwen3_8_q3_fallback_bench_20260910.jsonl` / `d2e4a252bd0f75681c5f47cd2cb8ca1355f9b5fa132b99f865ed075aaf8c8326` |

实现与验证：

- canonical resident ABI 为 `ql[N,K/4] uint8`、`qh[N,K/8] uint8`、`scale[N,K/16] fp16`，其中 `weight[k] = scale[k/16] * (low2[k] - 4*subtract[k])`。
- 七个真实 Q3_K tensor 的 109,568 行全部与 `gguf.dequantize` 对比；最坏 `max_abs=6.103515625e-05 < 1e-4`，CPU/XPU repack bit-exact。
- kernel synthetic 18/18，通过 K=256/512/5120、M=1/2/3/4/8/16/17 和 non-contiguous output slice。
- 真实 kernel 覆盖 `[17408,5120]`、`[5120,17408]` 两种方向，首/中/末三个区段共 192 行、M=1/2/4/8/16；10/10 通过，最坏 `max_abs=0.00048828125 < 0.01`。
- SGLang Q3+IQ4 回归 26/26；覆盖 native/fallback、dense reconstruction、row permutation、same-kind merge、mixed-kind group 和 output slice。
- `SGLANG_GGUF_XPU_NO_Q3K=1` 只回退 Q3_K；Q3 symbol 独立 optional import，不影响 IQ4/Q4/Q5/Q6/Q8。

#### E2E、显存与正确性

| 项目 | Q3 native | Q3 forced fallback |
|---|---:|---:|
| TP0/TP1 load weight | 56.96 / 57.33 s | 45.37 / 45.57 s |
| resident weight/rank | 11.91 GB | 12.48 GB |
| KV capacity/rank | 368,640 tokens | 350,016 tokens |
| memory-pool 后余量 | 6.32 GB | 6.32 GB |
| benchmark 后 XPU 6/7 | 32,655.05 / 32,333.44 MiB | 32,655.00 / 32,282.46 MiB |
| 停止后 XPU 6/7 | 43.43 / 45.33 MiB | 43.43 / 45.34 MiB |

- 两条路径 `/health`、`/v1/models` 均 HTTP 200，model id 为 `/models/Qwen3.8-27B`。
- 两条路径均有 `1+1 -> 2`、严格 JSON 正确、素数函数合理、`巴黎` 正确、2,829-token `BLUE-7391` 与 18,831-token `ORANGE-86420` 提取正确。
- 禁用 thinking 的 `17*23-19` 在 native/fallback 均稳定回答 `362`（正确值 372），因此不是 Q3 kernel 专属数值问题；该请求不能单独作为 kernel 判错依据，严格数值由 tensor/kernel validator 保证。
- native/fallback 的 concurrency 2 都出现一路数字粘连或提前停止，concurrency 4 都正常，继续归入 Issue I-006，不归因于 Q3_K。
- XPU 4/5 在测试前后保持约 985～992 MiB 的既存占用；启动环境只设置 `ZE_AFFINITY_MASK=6,7`。30000 原本不存在，恢复为 N/A。

#### 性能 A/B

| 场景 | Q3 native | Q3 fallback | native 相对变化 |
|---|---:|---:|---:|
| short TTFT 中位数 | 1.4196 s | 1.3889 s | +2.2% 延迟 |
| 256-token decode 中位数 | 23.1910 tok/s | 23.1078 tok/s | +0.36% |
| 2,829-token cold prefill | 979.23 tok/s | 1007.16 tok/s | -2.8% |
| 2,829-token cached prefill 中位数 | 7666.30 tok/s | 7729.51 tok/s | -0.8% |
| concurrency 4 aggregate | 58.73 tok/s | 58.97 tok/s | -0.4% |
| 18,831-token marker E2E | 11.2636 s | 10.6528 s | +5.7% 延迟 |

concurrency 2 因 fallback 一路只生成 11 tokens，吞吐不可比。阶段 3 的结论是 Q3_K native 显著降低常驻权重并增加 KV capacity，但只有七个 tensor，当前端到端性能基本持平且 prefill 略慢；优化前应 profile，不引入未经证据支持的 fusion。

#### 清理与阶段结论

- 仅分别向记录 PID 37862、39269 发送 SIGTERM；两次均正常退出，30001 释放，XPU 6/7 回落。
- 30000 在阶段前后都不存在；没有恢复动作。
- 阶段 3 所有门禁完成，下一步进入阶段 4 IQ3_S。

### Run `20260910-S4-canonical`

- 两个 feature 工作区初始干净；30000/30001 均无 listener。本轮不操作 30000。
- IQ3_S ABI：qs[N,K/4]、qh[N,K/32]、signs[N,K/8] uint8；scale[N,K/32] FP16。120 bytes/256 elements；col_perm 按压缩 group 重排，hvd 必须整除 32。
- Synthetic canonical 14/14 通过。首次全行测试四个 tensor 共 32,768 行，CPU/XPU repack 全部 bit-exact、all finite；max_abs 分别 5.7220459e-5 / 5.0067902e-5 / 1.0728836e-4 / 5.7220459e-5，mean_abs 约 1.51～1.56e-6。
- 首次沿用 Q3_K 的 1e-4 门限失败，未进入服务测试。独立诊断确认 blk.15 row=3456 k=12258：FP32 scale=0.016304492950439453、FP16 scale=0.0163116455078125；magnitude=15，使 scale 舍入放大为 0.00010728836059570312。该 tensor 所有行与 reference 按 FP16 scale 重算后逐元素一致，无 index/sign 映射误差。
- 依据该舍入证据，将 IQ3_S canonical 固定阈值设为 1.2e-4；validator 增加全行 scale 舍入解释检查后重跑。Kernel 阈值继续固定为 0.01。
- 重跑通过：四个 tensor 全部 32,768 行，所有误差均由 FP16 scale 舍入解释，cosine >= 0.9999999792；CPU/XPU bit-exact。通过日志 `/tmp/qwen3_8_iq3_s_canonical_pass_20260910.log`。
- 初测日志 `/tmp/qwen3_8_iq3_s_canonical_20260910.log`；诊断 `/tmp/qwen3_8_iq3_s_rounding_20260910.log`；单测 `/tmp/qwen3_8_iq3_s_unit_canonical_20260910.log`（容器内）。

### Run `20260910-S4-kernel-integration`

- canonical commit `08de9966c`；llm-scaler kernel `8a1a725`；SGLang integration `a74c8a047`。
- IQ3_S 常驻 9-bit grid index + sign + final scale，512x4 LUT 以 kernel 固定常量 lookup，不展开为 [N,K] 常驻 magnitude。
- `CXX=icpx MAX_JOBS=8 python3 -m build --wheel --no-isolation` 构建通过；只安装到 `/llm-scaler/overlays/qwen3_8`。
- wheel SHA256 `fe61a59fa84d7c6033576f48971c899d86af3b6d43c607f226e20dcdff80b667`；core .so SHA256 `62f4c6051f595104ea7187a271ef40609ea6eb0aa081e9df342969a807d6143d`。
- runtime `gguf.py` SHA256 `8b23158b796e99af653435d1e841224537f4f23f4cc783c55b6e63c900fd9bf3`，import 已确认指向 source + overlay。
- kernel synthetic 23/23，覆盖 M=1/2/3/4/8/16/17、K=256/512/5120、非连续 output slice 和 ABI 参数拒绝。
- 真实 kernel 20/20：两种完整矩阵方向、首/中/末共192行、down 的 TP0/TP1 local K=8704、M=1/2/4/8/16；worst max_abs=0.00048828125 < 0.01，全部 finite。
- SGLang IQ3/Q3/IQ4/Q4 联合单测49/49；原 IQ4/Q3 kernel 回归38/38。
- 新 gguf.py 使用 `/llm-scaler/overlays/qwen3_8_pre_iq3` 旧产物导入：仅 IQ3_S 两个 symbol 为 None，Q4/Q5/Q6/Q8/IQ4/Q3 均保留。
- 覆盖审计（每个源 tensor 一行实际 `_xpu_prepare_shard` probe，再外推逻辑字节数）：main 851 tensors、MTP blk.64另15；四种新增类型均0 fallback。IQ3_S四个 tensor 全量逻辑字节为167,116,800，相比 FP16 713,031,680 少545,914,880 bytes（未计TP、merge副本、allocator）。
- 局部日志：`/tmp/qwen3_8_iq3_s_{build,kernel_synthetic,kernel_tp,sglang_regression,older_kernel_regression,old_wheel}_20260910.log`。覆盖 JSON：`/tmp/qwen3_8_iq3_s_coverage_{before,native}_20260910.json`。
- 首轮 native 服务 PID43714，memory sampler PID43715；端口30001，ZE_AFFINITY_MASK=6,7，TP2，不设置SPEC_DRAFT_PATH。30000仍无listener。

### Run `20260910-S4-E2E`

结论：阶段4通过。native/fallback均加载、完成串行正确性与长上下文，IQ3_S压缩常驻收益成立；既有Issue I-006仍然存在。

| 指标 | IQ3_S native | IQ3_S fallback |
|---|---:|---:|
| PID | 43714 | 51816 |
| load weight TP0/TP1 | 47.67 / 49.72 s | 43.97 / 44.17 s |
| resident weight/rank | 11.65 GB | 11.91 GB |
| KV capacity/rank | 376,960 | 368,640 |
| short TTFT median | 1.4563 s | 1.4230 s |
| decode 256 median | 23.1303 tok/s | 23.2023 tok/s |
| 2,829-token cold marker prompt/e2e | 971.95 tok/s | 997.81 tok/s |
| 2,829-token cached marker median（仅后两轮） | 7778.14 tok/s | 7713.16 tok/s |
| 18,831-token marker E2E | 11.0080 s | 10.8145 s |
| sampled peak XPU6/7 | 32655.95 / 32537.14 MiB | 32655.39 / 32340.72 MiB |
| stop后 XPU6/7 | 43.43 / 45.33 MiB | 43.43 / 45.33 MiB |

- 同一source/wheel/TP2/启动模板，fallback仅增加`SGLANG_GGUF_XPU_NO_IQ3S=1`。具体启动参数同交接模板，未设置SPEC_DRAFT_PATH。
- 健康与models均200；两边1+1=2，严格JSON正确，素数函数正确，巴黎正确，BLUE-7391与ORANGE-86420均提取正确。
- native/fallback的三轮256-token连续decode均顺序正常；concurrency2均有一路数字粘连/重复，concurrency4均四路正常。不能把此共同问题归因于IQ3_S。
- 禁用thinking的17*23-19均回答362；单独开启thinking后两边均回答372，并保存完整reasoning与content。
- native节省约0.26 GB/rank，KV多8,320 tokens；decode -0.31%，TTFT +2.34%，cold marker -2.59%，性能基本持平或略慢。仅4个tensor，当前不进行未经profile的fusion。
- memory sampler每轮查询4/5/6/7后等待5秒（含查询实际约6～7秒），记录加载、forward、decode、prefill和long-context阶段。4/5其他负载从约992 MiB变化到约26 GB后回落；本任务进程始终只绑定6/7。
- 只向记录PID43714、51816发送SIGTERM，服务均正常退出，30001释放；30000在全部检查中无listener，未停止或恢复任何30000实例。
- 原始服务日志`/tmp/qwen_iq3_{native,fallback}_20260910.log`；bench`/tmp/qwen_iq3_{native,fallback}_bench_20260910.jsonl`已记录完整request/response，重复prompt不再把cached速率冒充cold prefill。
- native service SHA256 `adb25152f5f1a31b9ae4a87b6305829ccd89244dd2b8e7de897b5e7d0bd868af`，fallback service `96d64c12523f7d30307439722462a06b41c33a0e2ff38a2dd8319c777940ae5b`。
- native bench SHA256 `c26759aba1978639f3b5acf23694b5cad93418db074a69a36d1814d9961f4ce4`，fallback bench `9bb5f05dc8f2326c4d191a8eaa21dd5bf601c0ce1fe07b6841b0b062600ae968`。
- 允许进入阶段5覆盖收口、Qwen3.6与持续稳定性回归。

## 5. 后续验证与 Run 记录模板

复制本节建立新 Run，不要覆盖旧 Run。

### Run `20260910-S5-stability-first`

- 第二次 native 冷启动成功：`iq3_stability`，server PID `58492`，sampler `58493`，TP PID `59032/59033`，detokenizer `59034`；仅 XPU 6/7、TP=2、30001。
- 计划连续 1800 秒，实际没有完成：前置 0.1 秒客户端 timeout、SSE cancel、health/JSON recovery 均通过；完成请求 42 后，请求 43 超时 300 秒。随后生成式 `/health` 也超时，故本轮稳定性失败。
- 请求 34（concurrency=2、4K filler、256 decode tokens、`ignore_eos=True`）返回 HTTP 200，但 content 为空，reasoning_content 出现重复文本；同批请求 33 正常。完整请求/响应保留，不将其归为运输超时，也不据此单独归因 IQ3_S。
- TP 栈采样处于 `SchedulerRequestReceiver` 的空请求 CPU Gloo broadcast，`batch=None`，未执行模型 kernel；TP0/TP1 分别在 `common.py:1423/1438`，是匹配的 size broadcast。此证据定位采样等待点，尚未证明 collective 本身是根因。默认 `/health` 会生成 token，超时不能单独证明 HTTP event loop 挂起。
- 显存平台约 XPU6 `32655.72 MiB`、XPU7 `32286.20 MiB`，未见持续上升。停止准确 client PID `61663` 和 server PID `58492`（SIGTERM），稍后确认父子进程均退出且 30001 释放；30000 始终无 listener。
- 原始容器日志：`/tmp/qwen_iq3_stability_20260910.log`、`/tmp/qwen_iq3_stability_stress_20260910.jsonl`、`/tmp/qwen_iq3_stability_memory_20260910.jsonl`；TP 栈 `/tmp/qwen_iq3_stability_tp{0,1}_stack_20260910.txt`。异常并发请求另存 `/tmp/qwen_concurrency_33_34_repro.json`。客户端因人工终止没有末尾 summary，不得将预设 duration 当成已完成时长。
- 新登记 I-007；后续先补逐类型 fallback 与旧模型回归，并做有限时长的相同压力客户端对照。固定 TP=2，不采用改变 TP 的绕过来宣称完成。

### Run `20260910-S5-q36-regression`

- Qwen3.6 `/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf` + `/models/Qwen3.6-27B`，仍通过同一脚本、overlay、新 wheel 和 IQ3 integration 源码启动。server PID `80529`，sampler `80530`；health 200，模型枚举正确。
- load TP0/TP1 `47.20/46.56 s`；resident weight `14.10 GB/rank`，KV `296832 tokens`；decode256 三次 `35.0600/35.2965/35.3117 tok/s`，中位数 `35.2965`；TTFT 中位数 `1.2567 s`。
- 1+1=`2`，17*23-19=`372`，严格 JSON、素数代码、巴黎均正常；marker 三次和 18,831-token 长上下文 marker 正确。冷 marker `1178.41 tok/s`（prompt/e2e），cached 两次 `10467.71/12106.23` 单列；长上下文 `9.8258 s`。
- concurrency2 数字序列 `[true,false]`，concurrency4 全通过；此基线也复现 I-006，因此不将并发异常称作已解决。指定的旧模型启动/固定正确性/基础性能回归完成。
- 采样峰值 XPU6/7 `32373.00/31623.36 MiB`。准确 PID SIGTERM 后 30001 释放；30000 始终无 listener。日志 `/tmp/qwen_q36_regression_20260910.log`、`/tmp/qwen_q36_regression_bench_20260910.jsonl`、`/tmp/qwen_q36_regression_memory_20260910.jsonl`。

### Run `20260910-S5-coverage`

- 源文件 866 tensors：主模型 851，另有未启用 MTP `blk.64` 的 15 个（7 F32、6 Q6_K、2 Q8_0）。下表为主模型原始 tensor 计数和 canonical 逻辑 bytes；metadata probe 按单行 repack 推算完整 tensor，不包含 TP、合并复制和 allocator 开销。

| 原始类型 | tensors | native | FP16 fallback | 未量化 | canonical native bytes |
|---|---:|---:|---:|---:|---:|
| F32 | 353 | 0 | 0 | 353 | 0 |
| IQ3_S | 4 | 4 | 0 | 0 | 167,116,800 |
| IQ4_NL | 7 | 7 | 0 | 0 | 330,301,440 |
| IQ4_XS | 117 | 117 | 0 | 0 | 5,040,046,080 |
| Q3_K | 7 | 7 | 0 | 0 | 311,951,360 |
| Q4_K | 104 | 104 | 0 | 0 | 4,671,078,400 |
| Q5_K | 131 | 131 | 0 | 0 | 5,273,026,560 |
| Q6_K | 24 | 24 | 0 | 0 | 1,571,225,600 |
| Q8_0 | 104 | 104 | 0 | 0 | 69,632,000 |

- F32 原始逻辑 bytes `10,582,016`，不计入量化 fallback。四个新增类型的主模型 fallback 均 0，probe failures 0。
- 运行期源码复核：IQ4/Q3/IQ3 的大M shard matmul在调用内临时重建dense权重，没有写入常驻缓存；decode读取compressed rep。两目标模型HF配置均 `tie_word_embeddings=false`，不走tied lm-head的resident dense缓存分支。设备allocator保留的空闲显存与仍被模型引用的FP16权重分开理解。
- 配套启动脚本同步后，另核对59个SGLang生产文件、ESIMD kernel源文件及启动脚本，全部host/container SHA256相同；逐文件记录 `/tmp/qwen_full_source_sync_audit_20260910.json`。
- `4adbc1272` 新增 opt-in `SGLANG_GGUF_XPU_LOG_RESIDENCY=1`，默认关闭；同步前逐文件 readlink 后 docker cp。当前容器 gguf.py SHA256 `bf9ced9528f18e7a2e7e8c380df8583e03eaeb3cf86ec2e58b64a11f5cfa7efe`，wheel/overlay 未变。新增日志与 IQ3/Q3/IQ4 回归 `46 passed`（`/tmp/qwen_residency_unit_20260910.log`）。
- native_final 真实加载每 rank 306 层记录、594 个 prepared shard，全为 native。该计数包含融合层拆分 shard，不能当作 GGUF 唯一 tensor 数；NL/XS 同样在最终 canonical kind `iq4` 中汇总。日志中没有 layer.prefix 的层为 `?`，按每条记录处理，storage 只按 `(device,data_ptr,nbytes)` 去重。

| canonical kind | 每 rank 去重 physical storage bytes |
|---|---:|
| iq3_s | 83,558,400 |
| iq4 | 3,858,923,520 |
| q3_k | 200,540,160 |
| q4_k | 3,246,489,600 |
| q5_k | 3,544,842,240 |
| q6_k | 794,787,840 |
| q8_0 | 52,920,320 |
| 合计 | 11,782,062,080 |

- 这个加载期统计只涵盖 dense linear/embedding 的 final reps、merged reps 和 groups；不是进程总显存，不含 KV、其他模型缓冲区和后续运行期缓存，不声称覆盖 MoE。按源码来源映射的 bytes 与实际 storage 分开保存。
- 实际混合 topology 中包括 IQ3+IQ4、IQ4+Q3、IQ4+Q4、IQ4+Q5、IQ4+Q5+Q8、IQ4+Q6+Q8；完整 36 种带 multiplicity 的 topology 及各层 reps/groups/merged 组合在汇总 JSON。此为加载覆盖证据，数值集成回归由前述 49 项测试和 E2E 提供，未声称逐层完整数值比较。
- 原始记录容器 `/tmp/qwen_native_final_20260910.log`；离线汇总宿主 `/tmp/qwen_native_final_residency_20260910.json`，脚本 `test/manual/quant/summarize_gguf_xpu_residency.py`，含零记录/错误 JSON 失败处理。测试构建与请求日志已归档宿主 `/home/intel/shaojun/sglang/artifacts/qwen3_8_iq3_20260910`，完成后追加最终日志和 SHA256 manifest。

### Run `20260910-S5-native-final-stress`

- 第三次 native 冷启动 server `87535`、sampler `87536`。加载 `40.68/40.63 s`，resident `11.65 GB/rank`，KV `376960`，固定 benchmark decode 中位数 `23.2396 tok/s`、TTFT `1.4539 s`，固定集与阶段4一致（无 thinking 的多步算术仍错为362）。
- benchmark 后持续客户端计划1800秒，`--skip-probes --request-timeout-seconds 60 --stop-on-transport-error --fail-on-errors`。实际 `227.733 s` 后退出1：42个请求，41个 HTTP200，一个 timeout；末尾 health+marker recovery 失败，`completed_requested_duration=false`。
- 第18、38个请求在 concurrency2 JSON marker 输出中重复数字；第34个请求 content 为空且 reasoning 重复。第42个请求超时。最后模型日志完成一次decode时 `#queue-req:1`，之后没有新batch，不能仅凭 TP idle broadcast 栈将根因归为 Gloo。
- 该轮没有前置 timeout/cancel，故这些探测不是复现 I-007 的必要条件。首轮失败记录保留。
- 原始容器 `/tmp/qwen_native_final_stress_20260910.jsonl`、`/tmp/qwen_native_final_stress_summary_20260910.json`。宿主现场 `/tmp/qwen_native_final_tp{0,1}_stall_20260910.txt` 与 `http_stall`。客户端正常写出失败 summary；准确server PID SIGTERM后30001释放。
- 下一步分别诊断共享 GDN 小 batch 输出路径与 scheduler/Mamba 排队停滞，并继续最终 fallback 矩阵。保持 TP2，不以改变 TP 或掩盖输出检查作为稳定性通过。

### Run `20260910-S5-gdn-race-repro`

- I-006 定位取得局部复现证据：实际 TP2 local heads `H=8, HV=24, K=V=128`；`gdn_conv_fused_seq.h:630` 用 `N*HV<=WG_SIZE`（64）决定 inline shift，因此 N2 为内联，N3/4 为独立 shift。
- 每个 HV 是独立 workgroup；workgroup barrier 无法保证所有 head 的 conv-state 读取完成，hv0 提前写 conv state 会与其他 head 读取竞争。总 workgroup 数小不能提供跨 workgroup 同步保证。
- 使用当前未修改 wheel，对相同两个输入和缓存槽 `[3,1]` 分别运行 N2 与 N3（额外独立有效槽4），再运行第二份 N3 control。4随机种子×8连续步骤，共32步；N2/N3 每一步均不同，且只发生在第二行。输出 max_abs `0.00803375244140625`，SSM `0.21857452392578125`；第一行 bit-exact，N3/control 全 bit-exact；全部 conv 更新与独立 shift oracle bit-exact，无非有限值。
- 原始容器 `/tmp/qwen_gdn_shift_before_20260910.jsonl`；临时复现宿主 `/tmp/validate_gdn_inline_shift_20260910.py`，仅在30001停止、XPU6空闲后运行。该实验强力支持 inline 分支污染，仍需修复后同测及 E2E 确认 I-006 是否全部解决。
- 决策：独立于 IQ3_S 的小修复，统一在 recurrence kernel 后运行已有 conv-state shift kernel；不增加 fusion，也不改变 TP。I-007 调度停滞仍单独诊断，不能假定同时修复。

### Run `20260910-S5-quant-matrix-before-gdn-fix`

以下使用同一个 IQ3_S wheel `fe61a59f…`，公共 GDN 仍有已定位的内联状态竞争。它是量化接入后的完整逐类型对照，不能作为修复后稳定性结果。所有行均 TP2/XPU6,7/30001、同一 benchmark；cold 与 cached prefill 严格分开。

| 配置 | resident GB/rank（日志） | KV tokens | decode256 median tok/s | TTFT median s | cold marker prompt/e2e tok/s |
|---|---:|---:|---:|---:|---:|
| native_final | 11.65 | 376960 | 23.2396 | 1.4539 | 975.66 |
| iq3_fallback | 11.91 | 368640 | 23.2023 | 1.4230 | 997.81 |
| q3_final_fallback | 12.22 | 358528 | 22.9265 | 1.4246 | 992.34 |
| iq4_final_fallback | 20.43 | 89344 | 21.4802 | 1.0422 | 1193.74 |
| q36_regression | 14.10 | 296832 | 35.2965 | 1.2567 | 1178.41 |

- `q3_final_fallback` 只禁 Q3_K，IQ3_S/IQ4 保持 native；server `104736`、sampler `104737`。`iq4_final_fallback` 只禁 IQ4_NL/XS，Q3_K/IQ3_S 保持 native；server `111215`、sampler `111216`。两轮固定 API 集及长 marker 完成，C2 异常仍存在；准确 PID SIGTERM 后30001均释放。
- 容器原始 `/tmp/qwen_{q3_final_fallback,iq4_final_fallback}_20260910.log`、对应 `_bench_20260910.jsonl` 和 `_memory_20260910.jsonl`。统计宿主 `/tmp/qwen_<label>_report_20260910.json`。数据为独立单轮三次测量，不能将小幅差异解释为统计显著。
- 修复前后使用隔离 overlay 备份 `/llm-scaler/overlays/qwen3_8_pre_gdn_shift`；旧wheel另存 `/tmp/qwen_pre_gdn_shift_20260910.whl`，便于复现，不修改全局site-packages。

### Run `20260910-S5-gdn-port-kernel`

- 历史溯源：llm-scaler `2ac6a6f` 已在2026-09-07修复 vLLM 副本的相同竞争；SGLang独立副本没有同步。新提交 `0a08e20` 只将当前 SGLang seq dispatch 改为统一独立 shift，保留本分支 native/transposed 布局与负索引保护；不是改写或替换既有 IQ kernels。
- 新回归在旧wheel两种布局均失败（`/tmp/qwen_gdn_shift_pytest_before_20260910.log`），新wheel GDN+IQ3/Q3/IQ4 synthetic `63 passed`（`/tmp/qwen_gdn_quant_regression_20260910.log`）；新wheel IQ3真实shape/TP-local共20 cases通过，扩大到384采样行后worst max_abs `0.0009765625 < 0.01`，全部finite（`/tmp/qwen_iq3_kernel_after_gdn_20260910.log`）。与较早192行sample的最大值不直接比较。
- 构建发现：当前自定义 Ninja `sycl_compile` rule 没有 header depfile。第一次仅改header后build虽成功，实际只重新device-link，`esimd_kernel_lgrf.o`仍旧；该中间wheel没有安装。删除准确的生成物 `build/temp.linux-x86_64-cpython-312/csrc/xpu/esimd_kernel_lgrf.o` 后重新build，日志确认实际编译 `esimd_kernel_lgrf.sycl`。日志 `/tmp/qwen_gdn_build_20260910.log`（无效重链轮）、`/tmp/qwen_gdn_build_force_20260910.log`（有效重编轮）。后续只改kernel header时必须显式让对应object重新编译。
- 构建仍使用 `source /opt/intel/oneapi/setvars.sh` + `CXX=icpx MAX_JOBS=8 python3 -m build --wheel --no-isolation`，仅安装overlay。最终wheel SHA256 `dc8cd7c9ec463e43d5a18e2be8636d0a5799b71b2c649379314993667fb14d70`；lgrf SO `3d5c04e121958d7877783231cb033343510086062e4926ccd4e1a719486508a8`；core SO `1032f160680452c5024d6919c8860e4356a8c262d810aa19655c1a79b10da304`。旧wheel/overlay备份保持不动。
- 两次kernel submit使用相同 PyTorch XPU stream，其native queue为in_order（已按安装的torch commit `70d99e998b4955e0049d13a98d77ae1b14db1f45`核对），因此所有head读取完成后才会shift；不需要新增host同步。普通decode小batch多一次既有kernel launch，性能待E2E记录。
- 接下来仅带临时 `QWEN_SCHEDULER_DIAG=1` 重放原调度代码，验证GDN修复后内容与剩余I-007。临时文件宿主 `/tmp/qwen_scheduler_diag/{scheduler,request_receiver}.py`，逐文件readlink后docker cp；仓库canonical文件不含该heartbeat。最终bench前必须恢复canonical。

### Run `20260910-S5-gdn-fixed-scheduler-diag`

- 新 GDN wheel，原调度代码加 opt-in heartbeat；server PID `122449`，sampler `122450`，TP2/6,7/30001。连续客户端计划300秒，45秒请求timeout，不含前置probe。
- 前44个请求均HTTP200且内容检查全部通过，包括原来稳定出错的C2 marker与第二路diary；GDN局部修复后暂未复现I-006。第45/46个请求在停止服务之前超时，随后诊断流程结束并向准确server PID发SIGTERM；末尾47/48及recovery与清理时间可能重叠，不作为额外独立失败证据。客户端总记录48，退出1，实际265.85秒，未完成300秒。
- 决定性证据：08:45:52→08:45:57，TP0 recv_iteration `1,049,110→1,116,442`，TP1同步持续增长。两rank持续显示 `waiting_queue_len=2, running_batch_len=0, batch_is_full=True, admission_status=blocked_batch_is_full`。因此先前的空请求broadcast栈只是高频正常轮询采样，**Gloo未停滞**。
- I-007根因：`alloc_group_begin()`把匹配前预留槽从free_slots移走；随后的`get_num_allocatable_reqs()`只读free_slots，误判没有容量，设置full并在消耗预留槽前break。group_end归还槽后，空running batch没有后续decode更新来清full，形成持续拒绝准入。实际CPU复现同一方法显示可用量 `1→0→1`。
- 修复范围收敛为独立的remaining-reserved计数，仅用于scheduler的已有准入上限；不改变available_size的free-only定义，也不重写prefix/COW/HiMamba/preemption/session/multimodal路径。上层`_resolve_max_num_reqs()`始终按Mamba pool/ratio限制request pool，普通64槽/no-overlap配置上限16；小池回归会核对此生产约束。
- 原始容器 `/tmp/qwen_gdn_fixed_diag_20260910.log`、`/tmp/qwen_gdn_fixed_diag_stress_20260910.jsonl`、`/tmp/qwen_gdn_fixed_diag_stress_summary_20260910.json`。CPU复现宿主 `/tmp/repro_mamba_group_admission_20260910.py`；历史kernel审计宿主 `/tmp/qwen_kernel_history_audit_20260910.md`。
- 30001确认释放、30000无listener；临时diagnostic源码将在下一轮正式修复验证前恢复。

### Run `20260910-S5-mamba-reservation-fix`

- SGLang commit `7ae8901f0`：Mamba allocator 单独暴露尚未消耗的 group reservation，scheduler 的原有容量上限包含这些槽；available_size/bulk allocation 保持原语义。
- CPU regression 调用真实 `_get_new_batch_prefill_raw()`，确认最后一个槽被 group 暂存时仍能到达 request init，并消耗该槽；另覆盖归还、clear、失败/重复 group、bulk 分配，以及真实生产 pool4→request-cap1。
- 用 Git HEAD 修复前 `get_num_allocatable_reqs()` 替换测试进程中的方法：6 tests 中上述准入两项失败；新版本 `6 passed`。日志 `/tmp/qwen_mamba_reservation_{red,unit}_20260910.log`。不修改运行中服务代码对象。
- 独立审查无 blocker；调度/COW/HiMamba/priority/session/multimodal 原流程保留。临时 heartbeat 的 request_receiver 和 scheduler 已恢复仓库源码；新服务没有 diagnostic 环境变量。
- 原生量化单测汇总 `53 passed`（IQ3/Q3/IQ4/Q4及驻留统计），日志 `/tmp/qwen_final_quant_unit_20260910.log`。

### Run `20260910-S5-gdn-snapshot-tracking`

- 全面审查此前快速路径发现 I-008：模型层 ESIMD decode 更新 working conv/SSM 后直接 return，跳过标准 backend 的 `_track_mamba_state_decode()`。extra_buffer/interval64 的调度仍标记 snapshot 长度并将 tracking slot 放入 prefix cache，可能将旧状态关联到已生成的 token 前缀。
- 最小修复：写回真实 pool 后、任何 norm/out-projection 返回之前调用既有 tracking helper。传入 native `pool_conv`/`pool_ssm`；不新增 CPU mask.any 同步。
- CPU test 对实际 AST 提取 wrapper 执行状态更新及复制，旧代码 4 个 tracked 子场景失败，新代码 3 tests/8 子场景通过；覆盖两种 norm 返回、legacy conv 写回顺序和 false/None masks。日志 `/tmp/qwen_gdn_snapshot_tracking_{red,green}_20260910.log`。
- 实际启用的 extend 路径仍返回标准 backend 的 conv/SSM tracking；另一个模型层 prefill shortcut 环境开关未启用，未发现当前 reachable prefill 的对应遗漏。
- 修复前服务 `tracking_before` server PID136233、sampler136234，GDN race 已修、admission 已修，model module 在 snapshot hook 同步前已加载。generated-prefix probe 命中256 token，大于初始64 token；cached/cold 8 个生成 token 相同，首 token logprob 差第一次 -0.03019、重复一次 -0.18070。它们是数值观测，不构成错误文本复现；局部状态测试提供直接根因证据。
- 首次 probe 将 HTTP200 text/plain flush 响应误判为 JSON 错误，脚本已修复且保留原始记录；重跑 flush confirmed、cold cached_tokens=0、prefill control hit、generated-prefix hit 均成立，transport failures=0。
- server136233 已准确 SIGTERM，两个测试端口均无 listener。实际 XPU snapshot equality 与 snapshot/source 复用后下一步 output/conv/SSM bit-exact 已通过（`/tmp/qwen_gdn_snapshot_xpu_20260910.log`）；修复提交 `70e30c29d`。后续运行修复后相同 API probe，以及最终持续回归。

### Run `20260910-S5-final-fixed-native`

- 最终代码 SGLang `70e30c29d`、llm-scaler `0a08e20`，wheel `dc8cd7c9…`；六个关键 host/container source SHA256 相同，已恢复无 heartbeat 的仓库版本。完整 provenance `/tmp/qwen_final_provenance_20260910.json`，wheel/module hashes `/tmp/qwen_final_wheel_hashes_20260910.json`。
- label `native_closed`，server146380、sampler146381，启动09:10:54 UTC；resident11.65 GB/rank，KV376960，加载39.74/40.14 s。
- benchmark：decode中位21.8241 tok/s，cold marker995.0285有效prompt tok/s；C2/C4所有序列检查通过，marker与18831-token长上下文通过；thinking-off算术仍362，保留此前 native/fallback 均出现的错误，不标记此题正确；未以独立 HF/FP16 后端定位其来源。
- generated-prefix 修复后 API probe：缓存256>初始64，cold cached_tokens=0，全部flush确认；generated cached/cold 输出8token相同，首token logprob差 -0.002326，普通prefill control差+0.002292。此对照仅作数值观察，不要求prefill/decode路径bit-exact。
- 最终驻留日志再次确认每rank306层记录/594准备shards均native；去重物理storage11,782,062,080bytes/rank，与修复前统计相同。原始完整log和 `/tmp/qwen_native_closed_residency_20260910.json` 一起归档。
- 旧符号兼容性补充：除真实 pre-IQ3 wheel 测试外，使用当前源码实际 `_imp_kernels` 与 import assignments，模拟分别缺 IQ4/Q3/IQ3 以及同时缺全部六个新符号的模块，4/4通过，其他20或16个kernel符号保持原对象。此为CPU模拟模块检查，不声称额外安装过三个历史wheel。宿主 `/tmp/qwen_optional_symbol_matrix_20260910.{py,log}`。
- 30分钟stress **通过**，09:14:31.272→09:45:18.121 UTC，实际1846.84896秒（包括探测及完整收尾），请求窗口完成，退出0。308/308 HTTP200、308/308内容检查通过、0 transport errors。前置主动timeout与取消流被观察到，初始与最终health/marker恢复均通过。
- 客户端并发1/2/4分别44/88/176个请求；1K/4K × 128/256全部覆盖。154个decode请求实际输出均等于目标128/256 token；另154个marker_json请求严格检查marker。检查不等同于任意生成内容的全面事实准确性。
- 各phase最长延迟51.8858秒（4K/256），小于60秒request timeout。服务批处理大小可随缓存状态与准入变化，客户端并发数不能当作每步实际device batch size。
- 压测期间显存平台约XPU6 32648.41 MiB、XPU7 32536.91 MiB，无持续上升。停止准确server PID146380后端口30001释放，XPU6/7回落50.6484/52.5742 MiB；下一轮启动前实测同值。原始交接空闲43.43/45.33 MiB不作为本轮实测值替代。30000始终无listener。
- 原始 `/tmp/qwen_native_closed_stress_20260910.jsonl`、summary JSON、service/benchmark/memory logs；下一轮开始逐类型最终矩阵。

### Run `20260910-S5-external-device-observation`

- 09:55:53→09:56:00 UTC，非任务卡4/5的外部显存占用从约1511/999 MiB升至19875/19682 MiB，之后约30494/30128 MiB。此时本任务在newtypes_closed_fallback加载阶段，任务server环境仍明确 `ZE_AFFINITY_MASK=6,7`。
- 本任务对4/5只使用xpu-smi只读采样，没有启动计算或停止任何外部进程；核对时30000仍无listener，同容器当前实际launch_server为本轮记录的30001 PID。未追踪或改变其他容器服务。
- 最后两组（新增类型全部fallback、Qwen3.6）与此前测试的外部负载状态不同，因此性能表作为本机实测描述，不能将所有小幅差异作严格受控因果比较。权重常驻字节数/原生覆盖及数值验证结论不依赖此性能假设。

### Run `20260910-S5-fixed-matrix-before-empty-tracking-guard`

以下六组均为 SGLang `70e30c29d` / llm-scaler `0a08e20` / wheel `dc8cd7c9…`。这是公共正确性修复后的完整矩阵，和较早未修 GDN 的矩阵分开。每组 decode、TTFT 各三次；cold marker 是 prompt tokens / 完整请求耗时，包含输出开销，不称纯 prefill kernel 吞吐；cached 两次另存原始JSON。

| 配置 | resident GB/rank（日志） | KV tokens | decode median tok/s | TTFT median s | cold marker prompt/e2e tok/s |
|---|---:|---:|---:|---:|---:|
| native_closed | 11.65 | 376960 | 21.8241 | 1.4549 | 995.03 |
| q3_closed_fallback | 12.22 | 358528 | 21.4789 | 1.4243 | 969.89 |
| iq4_closed_fallback | 20.43 | 89344 | 18.6300 | 1.0388 | 1180.91 |
| iq3_closed_fallback | 11.91 | 368640 | 21.8454 | 1.4187 | 1022.02 |
| newtypes_closed_fallback | 22.08 | 35136 | 18.5962 | 0.9621 | 1210.76 |
| q36_closed_regression | 14.10 | 296832 | 21.5226 | 1.2493 | 1114.94 |

- 六组 C2/C4 所有数字前缀检查通过，18,831-token 长 marker 均正确，HTTP 均200。Qwen3.8 各配置 thinking-off 多步算术均362，保留此错误；Qwen3.6 则372，其他固定集通过。
- native 相比新增类型全部 fallback，日志 resident 从22.08降至11.65 GB/rank，节省10.43 GB/rank，KV从35,136增至376,960。不能把“新增类型全部 fallback”称作全模型FP16；Q4/Q5/Q6/Q8始终原生。
- 最终加载记录各rank306条、594个prepared shards。native/fallback分别为594/0；Q3回退587/7；IQ4回退456/138；IQ3回退590/4；新增类型全部回退445/149。这里是包含融合拆分的shard数，不替代源GGUF唯一tensor计数。
- 各rank最终去重weight storage bytes：native 11,782,062,080；Q3回退12,383,682,560；IQ4回退21,197,127,680；IQ3回退12,055,019,520；新增类型全部回退22,962,995,200。融合/合并副本随配置变化，单类型差值不要求严格相加。
- 对应准确server PID：146380、211400、217802、225578、231973、240557，均已逐一SIGTERM并确认30001释放；30000均无listener。完整启动/采样PID账本 `/tmp/qwen_iq3_this_session_pids.json`。
- 汇总 `/tmp/qwen_final_matrix_20260910.json`，包含原始三次值、固定题文本及并发/长上下文判断；驻留汇总 `/tmp/qwen_final_residency_matrix_20260910.json`。原始 `/tmp/qwen_<label>_20260910.log`、`_bench_20260910.jsonl`、`_memory_20260910.jsonl`。

### Run `20260910-S5-q36-performance-investigation`

- Qwen3.6 公共修复前decode中位35.2965 tok/s，修复后21.5226。外部4/5负载消退后的独立冷启动 `q36_performance_recheck`（server246776）三次21.1756/21.9929/21.8711，中位21.8711，故不能仅归因外部负载。固定内容检查仍通过。
- py-spy 200Hz轮明显落后，采样时间失真，只保留诊断原始文件，不用于定量性能结论。20Hz×10秒复测199 samples、0 errors、无behind警告，17 samples包含 `_track_mamba_state_decode`，54包含ESIMD wrapper，30包含GGUF shard matmul；仅说明host路径有开销，不足以归因全部性能回退。被profiler扰动的decode请求不并入正式benchmark。
- 源码确认 eager `_forward_metadata` 已一次性计算明确的 `has_mamba_track_mask`。图捕获/回放构造器不赋值，因此不能使用旧的默认False直接跳过。试验方案将默认改为None代表未知，只在明确False时跳过每层masked tracking调用，True/None仍复制，不增加逐层GPU同步。
- 原始 `/tmp/qwen_q36_performance_recheck_decode_20260910.jsonl`、`/tmp/qwen_q36_tracking_profile{,_lowrate}_20260910.txt`。server246776已准确SIGTERM，30001释放。

### Run `20260910-S5-empty-tracking-guard`

- SGLang `51a0d849b`：`ForwardMetadata.has_mamba_track_mask` 改为Optional bool，默认None表示producer未知，仅明确False跳过ESIMD快照helper。eager每次产生明确bool；graph capture/replay省略字段时仍提交tracking。其他metadata消费者的truthiness行为不变，无新增逐层同步。
- CPU容器测试6项/14子场景全部通过，覆盖真实producer、dataclass未知默认、False零调用、True/None/缺字段保留复制、working pool写回和两条norm路径；实际XPU未知标志snapshot及下一步reuse仍bit-exact。日志 `/tmp/qwen_gdn_empty_tracking_{unit,xpu}_20260910.log`。宿主最初两次pytest因缺少sglang依赖（最终为orjson）未收集；正式通过来自完整容器依赖环境，未修改宿主依赖。
- Qwen3.6 server269269，新gate+同一正确性wheel；decode三次24.3678/24.6638/24.9409 tok/s，中位24.6638，较无gate空闲复测21.8711提升约12.8%，仍低于旧35.2965。固定集、C2/C4数字前缀、长marker通过，TTFT中位1.25016 s。
- Qwen3.6 generated-prefix全部flush/cache检查通过：命中256>初始64，cached/cold8token相同，首tokenlogprob差-0.00128844；普通prefill control差-0.00015945。该API数值仅观察，不声称两个计算路径bit-exact。
- server269269准确SIGTERM后30001释放；新版本七个关键host/container源码hash一致，provenance `/tmp/qwen_guard_provenance_20260910.json`。旧30分钟结果严格归属于70e30c29d，此后补充新gate版本回归，不覆盖或改写旧结果。
- 独立GDN旧/新overlay计时只作诊断：N1吞吐式每调用8.113→11.165微秒，逐调用同步延迟44.978→43.010微秒；N2 10.305→12.712微秒；N4旧/新本来均独立shift，21.882→16.663微秒。局部新增约3微秒×48层不足以解释服务剩余降速，不能把全部成本归因独立shift。原始 `/tmp/qwen_gdn_shift_cost_{old,new}_20260910.json`，不把已知有race旧wheel当正确性通过版本。

### Run `20260910-S5-q36-wheel-and-hook-isolation`

- 同一当前Python `51a0d849b` 只将服务PYTHONPATH指向旧 `qwen3_8_pre_gdn_shift` overlay；不重新安装、不覆盖当前正确性wheel。诊断server277096，decode三次23.7453/24.5680/24.8752，中位24.5680，与当前wheel24.6638接近。已知旧wheel有GDN race，绝不作为交付版本或正确性通过证据。
- 第二次仅容器 `qwen3_5.py` 临时使用Git `7ae8901f0` 的完整原文件（没有新增snapshot调用），当前正确性wheel保持不变；诊断server282320。decode三次23.5925/24.4124/24.5482，中位24.4124，同样没有恢复35.2965。文件逐次readlink后docker cp，诊断后恢复canonical源码。
- 两个诊断server均已准确SIGTERM，30001释放。逐一对照表明当前条件下新wheel/快照hook均不足以解释旧35.30到当前24.6的差异；不能将全部历史差异归因某个公共正确性修复。原始 `/tmp/qwen_q36_{wheel_cost_old,no_snapshot_hook}_decode_20260910.jsonl`，临时前置源码 `/tmp/qwen_q36_no_snapshot_hook_20260910.py`。
- 这些是各一轮三次测量，未控制整台宿主机历史负载/频率，仍保留历史性能下降的观察。后续收口必须报告当前最终版本实测，而非选择较快旧数字。

### Run `20260910-S5-launcher-provenance-audit`

- 对容器当前已修改的49个SGLang tracked文件逐个SHA256核对，全部与宿主canonical一致。保存的scheduler/request_receiver diagnostic文件相对7ae父提交只多显式heartbeat hunk，没有丢弃额外容器优化。加载期residency日志没有改变decode热路径。
- 解析旧/新Qwen3.6完整ServerArgs，除random_seed外相同：模型、TP2、dtype、page64、track64、graph/overlap/cache、请求容量16都一致。实际decode均running1、queue0、mamba_num2，准入修复没有扩大实际单请求batch。
- 发现启动脚本漂移：当前容器脚本与早先宿主 `/tmp/start_qwen3_6_service.container.sh` 完全相同（SHA256 `97a694…`），但比宿主llm-scaler canonical少 `SGL_XPU_GGUF_RESADD_NORM_KQ`、`SGL_XPU_GGUF_MLP_SILU`、`SGL_XPU_GGUF_NORM_OUT_Q5K` 三个export；其余内容相同。这三个开关早已在用户提交 `1ee060e` 添加，SGLang默认关闭。
- 没有证据表明该旧容器脚本在两次历史性能测量之间变化，因此不把它直接认定为35→24差异根因。需要修复最终交付的配置一致性：现有native_guard为旧脚本基线，结束后同步宿主已提交脚本并单独验证既有fusion路径，保留两套配置的结果。
- 后续找到配套提交证据：SGLang `b4599c0ef` 把这三个环境开关默认从1改成0，llm-scaler `1ee060e` 配套在GGUF启动时显式设1。系统site-packages旧模型文件仍默认1，且旧Qwen3.6服务日志有大量 `gguf_resadd_norm_kq` / `gguf_mlp_silu` ACTIVE。模型文件同步到宿主版本而启动脚本未同步，造成既有fusion被关闭；这比“kernel修复导致降速”有更直接的执行路径证据。待配套脚本同步后的同模型测量确认性能恢复。
- 双rank ACTIVE条数提供直接执行证据：旧Qwen3.6为KQ128 / MLP128 / Q5K96，修复后旧脚本两轮均0/0/0；旧Qwen3.8 native_final为2/10/66，native_closed与native_guard均0/0/0。这也解释为什么Qwen3.6受影响大于Qwen3.8：后者大量IQ4/Q3/IQ3混合层本来就不满足旧Q4/Q6 fusion类型guard。计数是日志ACTIVE条数，不是唯一GGUF tensor数。

### Run `20260910-S5-canonical-launcher-q36`

- 配套部署修复：readlink后docker cp宿主llm-scaler `sglang/scripts/start_qwen3_6_service.sh`，脚本内容来自已有 `1ee060e` + `4f47c57`，没有新增fusion kernel。SGLang仍51a0d849b，wheel仍dc8cd7c9…。
- server306766，TP2/6,7/30001；读取其实际 `/proc/PID/environ` 确认三个GGUF fusion开关均1，SPEC_DRAFT_PATH不存在。四个运行关键文件（包括launcher）host/container hash一致，记录 `/tmp/qwen_q36_canonical_launcher_provenance_20260910.json`。
- Qwen3.6 decode三次34.8545/35.0673/35.0056 tok/s，中位35.0056，接近旧35.2965（约-0.82%，单轮波动范围内，不声称性能提升）。相较旧脚本guard24.6638恢复约42%。TTFT中位1.25176 s，C2/C4数字前缀全部通过，固定集与18,831-token长marker通过。
- generated-prefix全部flush/cache门禁通过，transport failures0。准确server306766 SIGTERM后30001释放，30000无listener。
- I-009闭环：历史ACTIVE→关闭→配套脚本恢复ACTIVE与性能恢复互相印证，原因是默认开关修改与配套启动脚本漏同步，而不是Q3/IQ4/IQ3 kernel数值或GDN正确性修复。后续最终矩阵/稳定性使用配套完整配置；此前70e/51a旧脚本结果保留为独立基线。

### Run `20260910-S5-existing-fusions-numeric`

- 为既有三类fusion补直接数值证据，脚本 `test/manual/quant/validate_qwen_existing_gguf_fusions.py`。模型不加载、服务停止后仅XPU6/7执行；使用production Q4/Q5/Q6 canonical repack/dequant，FP32 dense reference保留FP16 residual/norm/GEMV/gated中间舍入。
- Synthetic权重使用有限GGUF block scales（d=2^-15、dmin=2^-16），随机量化payload/subscale；这些范围在第一次执行前选定，避免不合理的大权重使FP16 SiLU乘积溢出。门限从第一次执行固定max_abs0.01，没有放宽。实际K5120、GDN K3072/V128，小输出行64/128用于局部运算检查；不把它称作完整真实tensor验证。
- KQ M1/2/4/8/16共5场景：混合Q4+Q6共享输出buffer的偏移/gap、fp16 BA、输入residual保持不变、独立new-residual和normed输出；worst max_abs0.001953125。
- Dense MLP Q4 SiLU M1/2/4共3场景，worst0.0001220703125。Q5 gated norm M1一个场景，max_abs0。均finite，9/9场景通过；MLP默认M<=4、Q5默认M1，未伪称启用了不支持的M。
- 原始 `/tmp/qwen_existing_fusions_numeric_20260910.json` 与stderr log（空），实际安装的是当前dc8cd7c9…wheel。已有IQ3/Q3/IQ4数值证据仍按此前各自reference/kernel门禁保留，不重复编造新的量化验证轮次。

### Run `20260910-S5-final-canonical-native`

- 最终完整运行配置：SGLang逻辑51a0d849b（另776268441/998a68870仅测试脚本）、llm-scaler0a08e20、dc8cd7c9…wheel、已同步1ee/4f启动脚本。实际进程的三个既有GGUF fusion开关均1，TP2/6,7/30001，SPEC_DRAFT_PATH不存在；provenance `/tmp/qwen_native_canonical_provenance_20260910.json`。
- label native_canonical，server314138、sampler314139；加载44.38/44.60 s，日志resident11.65 GB/rank，KV376960。decode三次22.4960/22.8788/22.8746，中位22.8746 tok/s；TTFT中位1.45490 s；首次marker989.9489 prompt/e2e tok/s，cached两次7646.44/7719.29单列。
- C2/C4数字前缀检查全部通过，固定API集、marker及18,831-token长marker完成；多步算术仍按已知错误单独记录。ACTIVE日志双rank计数恢复KQ2/MLP10/Q5K66；每rank306加载记录、594prepared shards全部native，去重weight storage11,782,062,080 bytes。
- generated-prefix门禁通过：flush均确认、命中已生成前缀、cold cached_tokens0，缓存/冷跑输出8token相同。首tokenlogprob差-0.0151707，普通prefill control差+0.0125694；仅记录路径舍入观察，不将cached/prefill数值要求为bit-exact。实际snapshot等于working以及下一步reuse的bit-exact证据仍来自单独XPU测试。
- **最终30分钟稳定性通过**：10:46:09.724→11:16:33.548 UTC，elapsed1823.8241 s，完成1800 s窗口，退出0。314/314 HTTP200和内容检查通过，0传输错误；前置主动timeout与cancel被观察到，初始/最终health+marker恢复通过。
- 客户端C1/C2/C4各46/92/176请求；157个decode的实际输出长度均满足128/256目标，另157个marker_json严格检查marker。最长请求49.6619 s，小于60 s timeout。内容检查的范围是长度/非空和marker等约定规则，不等同任意文本全面事实准确性。
- 实际输入长度：stress的1K阶段1051–1060 token，4K阶段4052–4060 token（含chat模板/marker差异）；固定benchmark marker为2829 token。最终独立显存探测另用严格1000/4000 raw input_ids并核对服务返回计数，不混用这三个口径。
- 显存平台XPU6约32641.19 MiB、XPU7约32540.91 MiB，无持续增长。11:16:37准确SIGTERM server314138后30001释放，XPU6/7回到50.6484/52.5742 MiB；下一轮启动前同值。30000无listener且未操作。
- 原始 `/tmp/qwen_native_canonical_{bench,stress,memory,prefix}_20260910.jsonl`，stress summary JSON、服务log及每rankresidency JSON。后续逐类型最终对照固定这套完整配置。

### Run `20260910-S5-delivery-close`

指定开发容器与隔离 overlay 的原生支持任务完成。最终依据为配套 canonical launcher 下的六组对照、314 请求持续回归、精确输入长度显存探测及清理记录；较早名字含 final/closed 的报告保留为历史，不替代本节。此结论不包含另建 Docker 发布镜像或未要求的后续性能优化。

#### 最终性能对照

所有行固定 SGLang 51a0d849b 运行逻辑、llm-scaler 0a08e20、同一 dc8cd7c9… wheel，以及宿主已有 1ee060e/4f 配套 launcher。后续 776268441/998a68870 仅增加及格式化测试脚本。TP2、XPU6/7、30001，三个既有 GGUF fusion 开关均为1，SPEC_DRAFT_PATH 未设置。

| 配置 | load 两rank (s) | resident 日志 GB/rank | KV tokens | decode 三次 (tok/s) | decode 中位 | TTFT 中位(s) | cold marker prompt/e2e tok/s | cached marker 两次 |
|---|---|---:|---:|---|---:|---:|---:|---|
| native | 44.38/44.60 | 11.65 | 376,960 | 22.4960 / 22.8788 / 22.8746 | 22.8746 | 1.45490 | 989.95 | 7646.44 / 7719.29 |
| Q3 fallback | 42.39/42.55 | 12.22 | 358,528 | 22.7211 / 22.8710 / 22.8341 | 22.8341 | 1.41745 | 964.95 | 7682.21 / 7723.76 |
| IQ4 fallback | 89.90/90.33 | 20.43 | 89,344 | 21.1757 / 21.1473 / 21.2873 | 21.1757 | 1.03933 | 1198.89 | 7860.44 / 8078.32 |
| IQ3 fallback | 42.86/42.89 | 11.91 | 368,640 | 22.8790 / 23.0171 / 23.0561 | 23.0171 | 1.42481 | 988.84 | 7743.22 / 7681.05 |
| 新增类型全部 fallback | 95.26/95.58 | 22.08 | 35,136 | 21.4806 / 21.7103 / 21.7683 | 21.7103 | 0.96200 | 1251.30 | 8070.99 / 8144.54 |
| Qwen3.6 native | 38.50/38.63 | 14.10 | 296,832 | 34.8545 / 35.0673 / 35.0056 | 35.0056 | 1.25176 | 1160.62 | 10410.19 / 12010.15 |

Fallback 只将对应类型恢复为常驻 FP16，其他类型继续原生；“新增类型全部 fallback”关闭 IQ4_NL/XS、Q3_K、IQ3_S，Q4/Q5/Q6/Q8 保持原生。Native 的大 M prefill 可临时 dense reconstruction，不代表权重永久展开。

Native 相比全部新增类型 fallback 的主要收益是权重/KV 容量，不能宣称所有场景更快。IQ3 单独回退的 decode 略高，按本轮观察保留；不以三次短测做显著性结论。Cold marker 是实际2829-token输入除以完整请求耗时，含生成开销；后两次命中 prefix cache，分列且不称纯 prefill 吞吐。Qwen3.6 decode35.0056 tok/s，与旧35.2965相差约-0.82%，此前大幅回退已恢复。

六组 C2/C4 数字前缀检查、marker 与18,831-token长上下文检查通过。固定1+1、JSON、代码结构、常识正常。**已知正确性例外保留**：Qwen3.8 thinking-off 的17*23-19仍答362（应为372），native和各fallback一致；未使用独立HF/FP16后端，不能据此确定是模型本身问题。Qwen3.6回答372。HTTP200不等于此算术内容通过。

#### 最终真实驻留

以下是每rank的prepared shard分类及按storage地址去重的权重字节数；不含KV、临时激活/解量化或allocator缓存。主模型唯一源tensor仍为498量化+353F32，额外15个MTP未启用。不能将prepared shard数当作GGUF唯一tensor数。

| 配置 | rank | prepared 分类 | physical weight storage bytes |
|---|---:|---|---:|
| native_canonical | 0 | {"native": 594} | 11,782,062,080 |
| native_canonical | 1 | {"native": 594} | 11,782,062,080 |
| q3_canonical_fallback | 0 | {"fallback": 7, "native": 587} | 12,383,682,560 |
| q3_canonical_fallback | 1 | {"fallback": 7, "native": 587} | 12,383,682,560 |
| iq4_canonical_fallback | 0 | {"fallback": 138, "native": 456} | 21,197,127,680 |
| iq4_canonical_fallback | 1 | {"fallback": 138, "native": 456} | 21,197,127,680 |
| iq3_canonical_fallback | 0 | {"fallback": 4, "native": 590} | 12,055,019,520 |
| iq3_canonical_fallback | 1 | {"fallback": 4, "native": 590} | 12,055,019,520 |
| newtypes_canonical_fallback | 0 | {"fallback": 149, "native": 445} | 22,962,995,200 |
| newtypes_canonical_fallback | 1 | {"fallback": 149, "native": 445} | 22,962,995,200 |
| q36_canonical_launcher | 0 | {"native": 498, "unquantized": 96} | 14,315,028,480 |
| q36_canonical_launcher | 1 | {"native": 498, "unquantized": 96} | 14,315,028,480 |

IQ4_NL、IQ4_XS、Q3_K、IQ3_S 在最终 native 下 fallback 均0。完整原始类型计数/逻辑bytes见阶段5覆盖表，实际storage按kind明细见 qwen_canonical_residency_matrix_20260910.json。旧pre-IQ3 wheel实测仅IQ3回退；缺少IQ4/Q3/IQ3/全部新增符号的optional import模拟矩阵4/4通过，后者不是四个实际安装wheel。

#### 精确输入长度显存探测

新启动独立native_memory_final，加载后记录idle；每次flush确认成功，分别送入严格1000/4000 raw input_ids、生成1token，并断言服务prompt_tokens吻合、cached_tokens=0；随后实际生成256token。设备显存单位MiB。

| 时点 | XPU4（只读） | XPU5（只读） | XPU6 | XPU7 |
|---|---:|---:|---:|---:|
| 启动前 | 992.54 | 992.59 | 50.65 | 52.57 |
| 加载后idle | 996.45 | 996.50 | 28538.60 | 27788.88 |
| 1000-token prefill采样峰值 | 999.30 | 999.37 | 30849.12 | 30099.40 |
| 4000-token prefill采样峰值 | 999.31 | 999.37 | 31219.33 | 30469.73 |
| 256-token decode采样峰值 | 999.30 | 999.37 | 31560.47 | 30810.94 |
| 256-token decode采样中位 | 999.30 | 999.37 | 31560.47 | 30810.93 |
| 停止后 | 992.53 | 992.59 | 50.65 | 52.57 |

Sampler每轮sleep0.1 s，加上telemetry耗时后的实际间隔min/median/max为1.489/1.499/1.512 s。表中是设备采样峰值，不是精确分配峰值；时间戳在查询结束后生成，短请求窗口存在边界误差。总显存包含预分配KV与allocator缓存，不能据此反推常驻权重为FP16。完整采样/请求窗口见 qwen_exact_memory_report_20260910.json。

#### 交付和清理

- IQ3 canonical/kernel/dispatch 已分提交；公共路径 I-006 GDN有序shift、I-007准入预留槽、I-008生成前缀快照，以及I-009配套launcher同步均完成回归。明确空快照跳过保留未知/graph metadata的保守复制。
- 量化SGLang单测53通过，GDN+新增量化kernel回归63通过，既有三类fusion直接数值9/9通过；真实IQ3四tensor全行canonical与两shape/TP局部kernel门禁见阶段4记录。
- 最终原生稳定性1823.8241秒、314/314请求内容门禁通过，0传输错误、timeout/cancel后恢复正常，无显存持续增长；不将固定算术错误隐藏在稳定性pass中。
- 所有本轮服务均以记录的准确PID发送SIGTERM。最终30001释放、没有遗留本轮launch_server，XPU6/7回落；30000始终只读检查，不停止、不恢复、不操作外部服务。最终观测时间和PID表见 delivery provenance。
- 归档目录：[验证产物](../artifacts/qwen3_8_iq3_20260910/README.md)。入口为 qwen_canonical_matrix_20260910.json、qwen_canonical_residency_matrix_20260910.json、qwen_exact_memory_report_20260910.json、qwen_delivery_provenance_20260910.json；manifest.json记录每个文件bytes和SHA256，保留原始请求/响应与历史失败日志。
- 最终wheel为 custom_esimd_kernels_sglang-0.1.0-cp312-cp312-linux_x86_64.whl，306,270,361 bytes，SHA256 `dc8cd7c9ec463e43d5a18e2be8636d0a5799b71b2c649379314993667fb14d70`。只安装到 `/llm-scaler/overlays/qwen3_8`，并在宿主归档一份；未覆盖全局site-packages。

### Run `YYYYMMDD-HHMM-Sx-NN`

目的：

关联阶段：

关联 Issue：

结论：`通过 / 失败 / 阻塞 / 仅诊断`

#### 代码和环境

| 项目 | 值 |
|---|---|
| UTC 开始/结束 |  |
| SGLang branch/commit/merge-base |  |
| llm-scaler branch/commit/merge-base |  |
| 两个 repo 的 `git status --short` |  |
| 相关 diff/commit |  |
| host `gguf.py` SHA256 |  |
| container build-source path/SHA256 |  |
| runtime imported `gguf.py` path/SHA256 |  |
| kernel commit |  |
| wheel/`.so` 路径及 SHA256 |  |
| Python executable/venv |  |
| `PYTHONPATH` |  |
| SGLang/kernel import 实际路径 |  |
| 容器 ID/启动时间 |  |
| 启动环境变量 |  |
| 完整启动命令 |  |
| server PID |  |
| 完整日志路径及 SHA256 |  |

#### 启动前隔离性

| 项目 | 结果 |
|---|---|
| 30000 所属容器/进程、PID、工作目录 |  |
| 30000 完整命令行/关键环境 |  |
| 30000 日志路径 |  |
| 30000 `/health` |  |
| 30000 smoke 输出摘要 |  |
| 30000 恢复命令已核对 |  |
| 30001 初始状态 |  |
| XPU 4/5 显存 |  |
| XPU 6/7 显存 |  |
| 30000 停止后端口/显存 |  |

#### 局部数值验证

| Test ID | 类型/tensor | shape/rank/M | reference | max abs | mean abs | max rel | cosine | NaN/Inf | 结果 |
|---|---|---|---|---:|---:|---:|---:|---|---|
|  |  |  |  |  |  |  |  |  |  |

补充检查：

- [ ] no-perm regression
- [ ] chunked == non-chunked
- [ ] TP rank 0/1
- [ ] first/middle/last block
- [ ] actual tensor shapes
- [ ] M=1/2/4/8/16（涉及 kernel 时）
- [ ] output slice
- [ ] same-kind merge/group
- [ ] mixed-kind shard
- [ ] per-type fallback

#### 服务加载

| 项目 | 结果 |
|---|---|
| 加载是否完成 |  |
| `/health` HTTP/内容 |  |
| `/v1/models` HTTP/内容 |  |
| 首个失败 tensor（如有） |  |
| traceback 摘要 |  |
| 各类型 native/fallback tensor 数 |  |
| 各类型 native/fallback resident bytes |  |

#### API 正确性

所有请求使用 `temperature=0`，保存完整 request/response。

| Test ID | 场景 | HTTP | 输出摘要 | 内容判断 | native/fallback 对比 | 结果 |
|---|---|---:|---|---|---|---|
| C01 | 中文 1+1 |  |  |  |  |  |
| C02 | `17*23-19` |  |  |  |  |  |
| C03 | 严格 JSON |  |  |  |  |  |
| C04 | Python 函数 |  |  |  |  |  |
| C05 | 固定常识 |  |  |  |  |  |
| C06 | 约 1K token 标记提取 |  |  |  |  |  |
| C07 | 256-token 连续 decode |  |  |  |  |  |
| C08 | concurrency 2 |  |  |  |  |  |
| C09 | concurrency 4 |  |  |  |  |  |

#### 显存

单位统一为 MiB；峰值注明采样周期。括号中可记录相对启动前增量。

| 时点 | XPU 4 | XPU 5 | XPU 6 | XPU 7 |
|---|---:|---:|---:|---:|
| 启动前 |  |  |  |  |
| 加载完成 idle |  |  |  |  |
| 1K prefill peak |  |  |  |  |
| 256-token decode steady/peak |  |  |  |  |
| 停止 30001 后 |  |  |  |  |

理论预期与实测差异解释：

#### 性能原始结果

注明客户端工具版本、tokenizer、input/output token 实际数量。每个场景 warmup 后至少三次。

| 模式 | 场景 | Run 1 | Run 2 | Run 3 | Median | 单位 | 失败数 |
|---|---|---:|---:|---:|---:|---|---:|
| native | short TTFT |  |  |  |  | ms |  |
| native | short TPOT |  |  |  |  | ms |  |
| native | decode throughput |  |  |  |  | tok/s |  |
| native | 1K prefill TTFT |  |  |  |  | ms |  |
| native | 1K prefill throughput |  |  |  |  | tok/s |  |
| native | concurrency 2 |  |  |  |  | tok/s |  |
| native | concurrency 4 |  |  |  |  | tok/s |  |
| forced fallback | 对应同场景 |  |  |  |  |  |  |

性能判断：

- native 相对 fallback：
- 波动范围：
- 是否有异常 kernel fallback/synchronization：
- 本轮数据能否用于正式结论：

#### 30001 清理与 30000 恢复

| 项目 | 结果 |
|---|---|
| 停止的 30001 PID |  |
| 30001 是否释放 |  |
| XPU 6/7 是否回落 |  |
| 30000 恢复命令/新 PID |  |
| 恢复后 30000 `/health` |  |
| 恢复后 30000 smoke 输出 |  |
| XPU 6/7 恢复前后显存差 |  |
| XPU 4/5 是否未被本任务使用 |  |

#### 问题、判断与下一步

- 观察：
- 根因证据：
- 临时绕过：
- 是否需要修改计划：
- 下一步：

## 6. Issue 登记表

| Issue ID | 首次 Run | 状态 | 现象 | 根因 | 修复 | 回归 Run |
|---|---|---|---|---|---|---|
| I-001 | Qwen3.8 首次启动 | 已修复 | `q4_k GDN out_proj col-perm unsupported` | Q4_K repack 缺少 value-head 列重排 | 阶段 1 完成压缩态重排 | `20260910-0502-S1-03` |
| I-002 | `20260910-0436-S0-01` | 外部状态变化 | 30000 `/health` 5 秒超时 | 未确认；阶段 1 开始前服务已由外部停止 | 本轮停止/恢复为 N/A | `20260910-0451-S0-04` |
| I-003 | `20260910-0446-S0-02` | 已修复 | 复制容器 build source 可能不被默认 Python 使用 | 默认 import 来自 site-packages，`PYTHONPATH` 为空 | 30001 使用 source overlay，并验证模块 `__file__` | `20260910-0502-S1-03` |
| I-004 | `20260910-0458-S1-02` | 已修复 | 首次 forward 在 `w.t().contiguous()` OOM | 大型 dense FP16 fallback 被永久缓存第二份 transpose | 仅缓存不超过 16 MiB 的小型 FP16 shard | `20260910-0502-S1-03` |
| I-005 | `20260910-0603-S2-04` | 已修复（测试流程） | 手写 fallback 启动先后因 GGUF config 缺失和 FP32 SSM state 内存不足退出 | 没有复现 `start_qwen3_6_service.sh` 设置的完整环境 | E2E 统一通过启动脚本，只增加被测 fallback 开关 | `20260910-0603-S2-04` |
| I-006 | `20260910-0526-S2-03` | 已修复，native持续回归通过 | concurrency 2 的数字序列偶发粘连、重复或提前结束 | GDN conv-state inline shift 跨 workgroup 读写竞争；vLLM fork 的历史修复漏同步 | `0a08e20` 使用既有独立 shift kernel；两种布局 red/green 及 44 请求内容检查通过 | `20260910-S5-gdn-fixed-scheduler-diag` |
| I-007 | `20260910-S5-stability-first` | 已修复，native持续回归通过 | 连续请求后空running batch仍保持full，队列饥饿 | Mamba预留槽未计入准入；heartbeat证明Gloo继续前进 | 单独补预留槽计数，保留容量上限 | `20260910-S5-gdn-fixed-scheduler-diag` |
| I-008 | `20260910-S5-gdn-snapshot-tracking` | 已修复，真实XPU/API回归通过 | 已生成前缀的 tracking slot 未更新 | 模型 ESIMD decode 提前返回跳过 backend snapshot helper | 写回 pool 后补调用；CPU red/green 通过 | `20260910-S5-gdn-snapshot-tracking` |
| I-009 | `20260910-S5-q36-performance-investigation` | 已修复，Qwen3.6性能恢复 | decode从35.30降至约24.6 | b459默认关闭fusion与1ee启动脚本显式开启未成套同步 | 同步已有canonical启动脚本，核对实际环境/ACTIVE日志 | `20260910-S5-canonical-launcher-q36` |

## 7. 阶段总结模板

### 阶段 X 总结

- 状态：
- 通过的 commit/产物：
- Reference 数值结论：
- E2E 正确性结论：
- Native/fallback 显存差：
- Native/fallback 性能差：
- 30000 停止/恢复及 XPU 4/5 隔离性：
- 已知限制：
- 未关闭 Issue：
- 是否允许进入下一阶段：
- 决策人/时间：
