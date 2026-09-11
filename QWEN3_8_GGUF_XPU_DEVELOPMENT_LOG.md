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
| 2 IQ4_NL/XS native | 未开始 | — | 未执行 | 未执行 | 未执行 | 未执行 | 当前为 FP16 fallback |
| 3 Q3_K native | 未开始 | — | 未执行 | 未执行 | 未执行 | 未执行 | 当前为 FP16 fallback |
| 4 IQ3_S native | 未开始 | — | 未执行 | 未执行 | 未执行 | 未执行 | 当前为 FP16 fallback |
| 5 全量收口 | 未开始 | — | 未执行 | 未执行 | 未执行 | 未执行 | — |
| 6 性能优化 | 未开始 | — | 未执行 | 未执行 | 未执行 | 未执行 | 正确性完成后开始 |

## 3. 已确认事实

### 3.1 环境与代码

| 项目 | 当前值 |
|---|---|
| SGLang 宿主机仓库 | `/home/intel/shaojun/sglang/sglang` |
| SGLang origin | `https://github.com/analytics-zoo/sglang` |
| SGLang branch / 当前基线 commit | `feature/qwen3.8-gguf-xpu` / `5034fd1b6d8f`（功能改动尚未提交） |
| SGLang upstream base | `origin/dev-bmg` / `66861ee2e0c485c4d34d1de56787ddfdf3fd2895` |
| llm-scaler 宿主机仓库 | `/home/intel/shaojun/sglang/llm-scaler` |
| llm-scaler origin | `https://github.com/intel/llm-scaler.git` |
| llm-scaler branch / commit | `feature/qwen3.8-gguf-xpu` / `4f47c5783f19` |
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
| 既有服务 | 30000，当前占用 XPU 6、7；E2E 前受控停止、结束后原配置恢复 |
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
- IQ4_XS、IQ4_NL、IQ3_S、Q3_K 已确认能被容器中的 `gguf.dequantize()` 解码，但当前会以 dense FP16 常驻 XPU。
- 当前没有提交 Q4_K col-perm 改动，没有同步新代码，没有启动 30001。

### 3.4 代码 provenance 与 dirty baseline

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

## 5. Run 记录模板

复制本节建立新 Run，不要覆盖旧 Run。

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
