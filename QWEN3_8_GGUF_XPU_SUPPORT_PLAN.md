# Qwen3.8-27B GGUF / XPU 支持实施与验证计划

本文是 Qwen3.8-27B GGUF 在 `sglang-dev-gguf` 中实现原生 XPU 支持的执行清单和阶段门禁。开发过程中如果实现方向、测试范围或通过标准发生变化，先更新本文，再继续开发。

每次测试的命令、代码版本、日志、正确性结果、显存和性能写入独立的 [开发验证日志](./QWEN3_8_GGUF_XPU_DEVELOPMENT_LOG.md)。本文只定义“应该做什么、怎样验证、什么条件下可以进入下一阶段”。

## 1. 目标与完成标准

目标模型：

- GGUF：`/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf`
- 宿主机文件：`/home/intel/weights/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf`
- HF config/tokenizer：`/models/Qwen3.8-27B`
- 架构：`Qwen3_5ForConditionalGeneration`
- 运行资源：XPU 6、7，`TP_SIZE=2`
- 测试服务：端口 30001

最终完成必须同时满足：

- Qwen3.8-27B GGUF 能稳定加载并完成 prefill/decode。
- Q4_K GDN `ssm_out` 的压缩态列重排正确。
- IQ4_NL、IQ4_XS、Q3_K、IQ3_S 不再作为常驻 dense FP16 权重加载。
- Native kernel 与 reference dequant + dense matmul 的数值误差在约定阈值内。
- OpenAI API 的固定测试集输出合理，无乱码、重复失控、空输出或明显错误。
- 记录 idle、prefill 峰值、decode 稳态显存以及基础延迟/吞吐。
- 每次占用卡 6、7 前，完整保存 30000 服务现场并受控停止；测试结束后按原配置恢复 30000，健康、输出和显存与基线一致。
- 所有新增路径均有可控 fallback，旧 GGUF 类型和 Qwen3.6 不回归。

“能启动”只是阶段 1 的目标，不是整个任务的完成标准。

## 2. 代码来源、目录与分支规范

### 2.1 当前代码来源

2026-09-10 使用 `git ls-remote` 确认三个远端均可读取，不需要在容器中执行 `gh login`：

| 组件 | 远端/分支 | 已确认的远端 HEAD | 本任务中的角色 |
|---|---|---|---|
| llm-scaler | `https://github.com/intel/llm-scaler.git` `main` | `5e2fea9596146af6e90038365ebd462ef59f5d23` | 镜像构建、patch、启动脚本以及 vendored ESIMD kernel 的集成仓库 |
| SGLang BMG | `https://github.com/analytics-zoo/sglang.git` `dev-bmg` | `66861ee2e0c485c4d34d1de56787ddfdf3fd2895` | SGLang Python/C++ 改动的 authoring 仓库 |
| sgl-kernel-xpu BMG | `https://github.com/analytics-zoo/sgl-kernel-xpu.git` `dev-bmg` | `fd89d0fde0b99dc8fb303945774e76a44f2f8d61` | 只有需要修改通用 XPU kernel 时才使用 |

GitHub 网页匿名访问可能返回 404，这是私有仓库的常见表现；以认证成功的 Git remote/fetch 结果为准。不要把 GitHub token 写入镜像、容器环境变量或 shell history。需要重新认证时优先在宿主机配置只读 SSH/credential helper，再由宿主机 clone/fetch。

当前 llm-scaler 镜像的实际构建方式与“容器直接 checkout 两个 analytics-zoo fork”不同：

- `Dockerfile.dev` 从 `sgl-project/sglang` 的 `v0.5.13` 构建，再应用 `sglang/patches/sglang_for_multi_arc.patch`。
- `Dockerfile.dev` 从 `sgl-project/sgl-kernel-xpu` 的 `ea5c70f0909bcd55ceaf1803302651fc0593b64d` 构建，再应用 `sglang/patches/sgl_kernel_xpu.patch`。
- `analytics-zoo/sglang:dev-bmg` 和 `analytics-zoo/sgl-kernel-xpu:dev-bmg` 是 patch authoring/同步来源；最终镜像仍回到“固定上游版本 + llm-scaler patch”的可复现流程。

未经记录不得把这两种 provenance 混用，否则本地测试通过的源码可能不是镜像实际构建的源码。

### 2.2 宿主机权威工作区

| 内容 | 权威目录 | Git 归属 |
|---|---|---|
| SGLang `gguf.py` 和 Python dispatch | `/home/intel/shaojun/sglang/sglang` | 独立 SGLang repo，origin 为 analytics-zoo |
| 目标文件 | `/home/intel/shaojun/sglang/sglang/python/sglang/srt/layers/quantization/gguf.py` | SGLang repo |
| llm-scaler Docker/patch/脚本 | `/home/intel/shaojun/sglang/llm-scaler` | llm-scaler repo |
| ESIMD kernel 源码 | `/home/intel/shaojun/sglang/llm-scaler/sglang/custom-esimd-kernels` | llm-scaler repo 中的普通 tracked 目录，不是 submodule/独立 repo |
| sgl-kernel-xpu patch | `/home/intel/shaojun/sglang/llm-scaler/sglang/patches/sgl_kernel_xpu.patch` | llm-scaler repo |
| SGLang patch | `/home/intel/shaojun/sglang/llm-scaler/sglang/patches/sglang_for_multi_arc.patch` | llm-scaler repo |

容器中的 build tree 不是权威源，不在容器内创建长期 commit。所有需要保留的修改必须先落到上述宿主机 feature branch，再同步/构建到容器。

### 2.3 ESIMD 新 kernel 的代码位置

本任务计划中的 IQ4/Q3/IQ3 GEMV 属于 `custom-esimd-kernels-sglang`，默认不改 `sgl-kernel-xpu`。至少涉及：

| 职责 | 路径 |
|---|---|
| kernel header/实现 | `llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/esimd_kernels/` |
| C++/SYCL wrapper | `llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/esimd_kernel.sycl` |
| C++ declaration | `llm-scaler/sglang/custom-esimd-kernels/include/kernel_ops.h` |
| Torch op schema/registration | `llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/torch_extension.cc` |
| Python wrapper | `llm-scaler/sglang/custom-esimd-kernels/python/custom_esimd_kernels_sglang/ops.py` |
| Python export/import | `llm-scaler/sglang/custom-esimd-kernels/python/custom_esimd_kernels_sglang/__init__.py` |
| Wheel definition | `llm-scaler/sglang/custom-esimd-kernels/setup.py` |
| Kernel tests | `llm-scaler/sglang/custom-esimd-kernels/tests/` |

现有 Q4_K 可作为结构参考：`q4_k_GEMV.h`、`esimd_gemv_q4_k[_m]` wrapper、Torch registration 和 Python export。新格式仍要使用独立 op 名称，不能伪装成 Q4_K。

只有在 profiling/接口约束证明新实现必须进入通用 `sgl_kernel` 时，才 clone `analytics-zoo/sgl-kernel-xpu:dev-bmg` 并创建第三个 feature branch。当前 Q4_K col-perm 和计划中的 IQ4/Q3/IQ3 GEMV 均不要求这样做。

### 2.4 分支与 clean baseline

不需要重新下载一整套“最新代码”再从头开始。更安全的方式是从已验证的精确 commit 建 feature branch，因为切到未经验证的新 HEAD 会同时改变大量非本任务因素。

当前任务分支：

| Repo | Feature branch | 固定起点 |
|---|---|---|
| SGLang | `feature/qwen3.8-gguf-xpu` | `b4599c0ef3270c6cc89e19b35174a89eec3e322d`，即已验证的 `origin/dev-bmg` + 1 commit |
| llm-scaler | `feature/qwen3.8-gguf-xpu` | `1ee060ea038a42f55d93794b73235161daec79a9`，即 `origin/main` + 2 commits |

开始功能代码前执行 clean-baseline gate：

- [x] 把本文档和日志作为独立文档 commit 保存。
- [x] llm-scaler 中已有的 `--skip-server-warmup` 修改单独保存，不能与 IQ kernel 混成一个 commit。
- [x] 开始功能改动前两个 repo 的 `git status --short` 为空。
- [x] 日志记录两个 repo 的 branch、HEAD、remote URL 和 merge-base。
- [x] Q4_K 与 IQ4 使用独立 commit；后续 Q3_K、IQ3_S 继续保持该边界。
- [x] 未使用 `git reset --hard`，也未覆盖用户无关修改。

如果需要完全隔离的目录，优先从上述固定 commit 使用 `git worktree add`，而不是重新 clone。新 clone 只用于本地 repo 损坏、权限域不同，或确实需要新增独立的 sgl-kernel-xpu authoring repo。

### 2.5 容器源码、安装产物与运行时解析

容器内存在三类不同位置：

| 内容 | 容器路径 | 性质 |
|---|---|---|
| SGLang build source | `/llm-scaler/sglang/sglang` | Docker build 保留的 patched clone |
| ESIMD build source | `/llm-scaler/sglang/custom-esimd-kernels` | 从 llm-scaler build context 复制的源码 |
| sgl-kernel-xpu build source | `/llm-scaler/sglang/sgl-kernel-xpu` | 固定 upstream commit + patch 的构建树 |
| 已安装 SGLang | `/usr/local/lib/python3.12/dist-packages/sglang` | 默认 `python3` 实际导入位置 |
| 已安装 ESIMD package | `/usr/local/lib/python3.12/dist-packages/custom_esimd_kernels_sglang` | 默认运行时 `.so` 位置 |
| 已安装 sgl-kernel | `/usr/local/lib/python3.12/dist-packages/sgl_kernel` | 默认运行时 `.so` 位置 |

当前容器 `PYTHONPATH` 为空，`find_spec("sglang")` 指向 site-packages。因此“把 `gguf.py` 复制到 build source”本身不足以证明测试服务使用了新代码。30001 必须使用显式 overlay：

```text
/llm-scaler/overlays/qwen3_8/              # 新 ESIMD wheel --target 安装位置
/llm-scaler/sglang/sglang/python           # 新 SGLang Python 源码
/usr/local/lib/python3.12/dist-packages    # 系统依赖；恢复后的 30000 继续使用这里
```

30001 启动前必须用与服务完全相同的环境执行：

```bash
PYTHONPATH=/llm-scaler/overlays/qwen3_8:/llm-scaler/sglang/sglang/python \
python3 -c 'import sglang.srt.layers.quantization.gguf as g; import custom_esimd_kernels_sglang as k; print(g.__file__); print(k.__file__)'
```

输出必须分别指向 `/llm-scaler/sglang/sglang/python/...` 和 `/llm-scaler/overlays/qwen3_8/...`。未命中时禁止开始 E2E，以免得到“改动实际未加载”的假结果。

### 2.6 从开发代码到可复现镜像

开发阶段的数据流固定为：

```text
SGLang feature branch
  -> 同步到容器 build source
  -> 30001 通过 PYTHONPATH 加载

llm-scaler feature branch/custom-esimd-kernels
  -> 容器内构建 wheel
  -> 安装到 /llm-scaler/overlays/qwen3_8
  -> 30001 通过 PYTHONPATH 加载

阶段验证通过
  -> 重新生成/更新 llm-scaler 的 SGLang patch
  -> 从 clean llm-scaler tree 完整构建 dev image
  -> 在新容器中做最终回归
```

30000 恢复后继续使用系统 site-packages。开发期间不对 `/usr/local/lib/python3.12/dist-packages` 执行全局 `pip install --force-reinstall`。

## 3. 不可违反的边界

- 只允许测试进程使用 `ZE_AFFINITY_MASK=6,7`；卡 4、5 不纳入本任务。
- 测试只使用端口 30001。由于 30000 当前也占用卡 6、7，两者禁止同时运行。
- 停止 30000 前必须记录其容器/进程、PID、完整命令行、环境、日志路径、健康/输出和显存；只停止精确识别的进程，并保留可复现的恢复命令。
- 禁止使用会同时命中多个服务的 `pkill -f sglang` 等宽泛命令。停止 30000/30001 时只能使用本次记录的准确 PID 或准确容器服务单位。
- 容器源码不是宿主机 bind mount。同步 Python 文件之前，必须执行 `readlink -f` 并核对目标路径；启动时还必须验证实际 import 路径。
- 原生 kernel 改动不能只复制 `gguf.py`；需要构建新的 wheel/共享库，并在 30001 专用隔离环境中加载。
- 不为赶进度跳过 reference 数值测试，也不以“HTTP 200”代替输出正确性验证。
- 不在正确性确认之前做 fusion。当前首要问题是量化格式和 GDN 列重排，不是算子融合。
- Qwen3.8 GGUF 的 `blk.64` 是 MTP 附加层。普通非 speculative 测试不设置 `SPEC_DRAFT_PATH`。

## 4. 已知模型差异与技术判断

### 4.1 Qwen3.6 与 Qwen3.8

- 两者主模型均为 64 层：48 个 GDN 层、16 个 full-attention 层。
- `hidden_size=5120`，FFN 为 17408，linear key/value heads 为 16/48，ratio 为 3，head value dim 为 128。
- 两份 GGUF 有 851 个同名同 shape tensor，其中 382 个 tensor 的量化类型发生变化。
- Qwen3.8 有 866 个 tensor，比 Qwen3.6 多出的 15 个 tensor 位于 `blk.64`，属于 MTP。
- Qwen3.8 GGUF 大小为 16,464,440,224 bytes。

Qwen3.8 的量化类型分布：

| 类型 | tensor 数量 |
|---|---:|
| F32 | 360 |
| IQ3_S | 4 |
| IQ4_NL | 7 |
| IQ4_XS | 117 |
| Q3_K | 7 |
| Q4_K | 104 |
| Q5_K | 131 |
| Q6_K | 30 |
| Q8_0 | 106 |

### 4.2 原始阻塞、已完成项与剩余 fallback

首次启动的最早阻塞点是五个 Q4_K GDN `ssm_out.weight`：

```text
blk.14.ssm_out.weight
blk.22.ssm_out.weight
blk.29.ssm_out.weight
blk.38.ssm_out.weight
blk.45.ssm_out.weight
```

Qwen3.6 对应权重主要是 Q5_K，已有压缩态 `col_perm` 支持；Qwen3.8 的这些权重是 Q4_K，原 `_xpu_prepare_shard()` 会主动触发：

```text
AssertionError: q4_k GDN out_proj col-perm unsupported
```

该阻塞已在阶段 1 修复。阶段 2 又完成了 IQ4_XS/IQ4_NL 原生常驻和 GEMV；阶段 2 结束时 IQ3_S、Q3_K 仍会先由 CPU 解量化，然后以 dense FP16 常驻设备。阶段 2 前 135 个 IQ4/Q3/IQ3 fallback tensor 的粗略成本是：

- GGUF 压缩 payload：全模型约 5.13 GiB。
- dense FP16：全模型约 19.61 GiB。
- TP2 下约 9.80 GiB/卡。
- 当前 fallback 下主模型权重估计约 15.15 GiB/卡；全部原生支持后估计约 8.10 GiB/卡。

实测阶段 2 将每 rank resident weight 从 22.08 GB 降至 12.48 GB，并将 KV capacity 从 35,136 提高到 350,016。剩余 Q3_K/IQ3_S 的 FP16 fallback 仍只能作为正确性对照和短期路径，不能作为最终生产实现。

### 4.3 原生表示的计划

IQ4_NL 与 IQ4_XS 使用不同 repack，但可以归一到同一种 kernel 输入：

```text
packed 4-bit LUT index: [N, K/2]
final fp16 scale:       [N, K/32]
weight = scale * IQ4_LUT[index]
```

- IQ4_NL：直接得到每 32 个元素一组的 scale。
- IQ4_XS：将 super-scale 与有符号 6-bit subscale 展开，预计算 final scale。
- IQ4_XS 的 `blk.13/16/17/18/33.ssm_out.weight` 同样需要压缩态 `col_perm`。

Q3_K 计划归一为低 2 bit、高/符号 mask，以及每 16 个元素的 FP16 scale。IQ3_S 计划归一为 4-bit magnitude、1-bit sign，以及每 64 个元素的 FP16 scale。最终表示以 reference 测试结果为准，不能仅凭格式推导直接冻结 ABI。

## 5. 验证原则：局部证明 + 阶段 E2E

不是每次编辑一行代码都启动服务，而是每个“可运行纵向切片”完成后必须做一次完整端到端验证。每个阶段包含：

1. repack/dequant 的 reference 数值验证；
2. kernel 对 dense matmul 的数值验证；
3. Python dispatch、混合类型 shard、输出切片和 fallback 验证；
4. 30001 服务启动、API 正确性、显存、基础性能验证；
5. 30000 的受控停止/恢复，以及卡 4、5 未被测试进程使用的复查。

如果只完成了尚未接入推理路径的 repack，启动服务不会覆盖新代码。这类中间提交只能通过局部测试，不能宣称阶段完成。

每一阶段只有在日志中写明 commit/diff、测试命令、结果和结论，并勾选阶段门禁后才能进入下一阶段。

## 6. 固定的端到端验证协议

以下协议在阶段 1～5 的每个阶段都执行一次，尽量保持参数完全一致。

### 6.1 启动前安全快照

记录 UTC 时间、git commit、dirty diff、Python 文件和 kernel wheel/`.so` 哈希：

```bash
cd /home/intel/shaojun/sglang/sglang
git rev-parse HEAD
git status --short
sha256sum python/sglang/srt/layers/quantization/gguf.py

docker inspect -f 'state={{.State.Status}} pid={{.State.Pid}} started={{.State.StartedAt}}' sglang-dev-gguf
docker exec sglang-dev-gguf readlink -f /llm-scaler/sglang/sglang/python/sglang/srt/layers/quantization/gguf.py
docker exec sglang-dev-gguf sha256sum /llm-scaler/sglang/sglang/python/sglang/srt/layers/quantization/gguf.py

ss -ltnp '( sport = :30000 or sport = :30001 )'
timeout 3s xpu-smi dump -d -1 -m 0,1,2,3,4,5,18
```

单独记录 30000 服务的 PID/命令行并请求 `/health`。健康请求超时必须写入日志并调查，不能因为端口仍监听就标为健康。只有恢复命令和基线记录齐全后才能停止 30000；确认端口释放、卡 6/7 显存回落后才能启动 30001。

### 6.2 同步与启动

仅 Python 改动时，先确认解析后的 build-source 目标就是预期文件，再同步并复核哈希：

```bash
docker exec sglang-dev-gguf readlink -f /llm-scaler/sglang/sglang/python/sglang/srt/layers/quantization/gguf.py
docker cp /home/intel/shaojun/sglang/sglang/python/sglang/srt/layers/quantization/gguf.py \
  sglang-dev-gguf:/llm-scaler/sglang/sglang/python/sglang/srt/layers/quantization/gguf.py
docker exec sglang-dev-gguf sha256sum /llm-scaler/sglang/sglang/python/sglang/srt/layers/quantization/gguf.py
```

注意：这里同步的是 build source，不是默认 site-packages。启动 30001 时必须增加 `/llm-scaler/sglang/sglang/python` 到 `PYTHONPATH`，并检查模块 `__file__`。

有 native kernel 改动时：

- 记录 kernel 源码 commit 和 diff。
- 构建新 wheel/`.so`，记录产物 SHA256。
- 使用 30001 专用 venv 或 `PYTHONPATH` overlay，不覆盖 30000 正在使用的全局安装。
- 在服务启动前用一个短 Python 程序 import 新符号并打印其实际 `.so` 路径。
- 新 kernel import 与旧 kernel import 分组。当前 `_imp_kernels()` 对同一 tuple 是 all-or-none，不能因为旧 `.so` 缺少新符号而关闭全部既有 kernel。

在容器内的专用 shell 中启动：

```bash
cd /llm-scaler/sglang
PYTHONPATH=/llm-scaler/overlays/qwen3_8:/llm-scaler/sglang/sglang/python \
MODEL_PATH=/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
GGUF_CFG_DIR=/models/Qwen3.8-27B \
SERVED_MODEL_NAME=/models/Qwen3.8-27B \
ZE_AFFINITY_MASK=6,7 \
TP_SIZE=2 \
PORT=30001 \
bash scripts/start_qwen3_6_service.sh
```

不得设置 `SPEC_DRAFT_PATH`。保存完整 stdout/stderr，记录 server PID；不要只复制最后一段 traceback。

### 6.3 服务和功能检查

至少检查：

```bash
curl -sS http://127.0.0.1:30001/health
curl -sS http://127.0.0.1:30001/v1/models
curl -sS http://127.0.0.1:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/models/Qwen3.8-27B",
    "messages": [{"role": "user", "content": "1+1 等于多少？只回答结果。"}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

固定正确性请求集至少包含：

| 类别 | 请求 | 基本通过条件 |
|---|---|---|
| 中文算术 | `1+1 等于多少？只回答结果。` | 明确回答 2 |
| 多步算术 | `17*23-19`，要求给结果 | 结果为 372 |
| 指令遵循 | 要求只输出指定 JSON | JSON 可解析、字段正确 |
| 简单代码 | 写一个 Python 素数判断函数 | 语法和基本逻辑正确 |
| 常识问答 | 固定、无时效性的事实问题 | 答案合理，无明显幻觉 |
| 长 prefill | 固定约 1K token 上下文后提取标记 | 找到正确标记 |
| 连续 decode | 固定 prompt，输出 256 token | 不乱码、不异常重复、不提前空结束 |
| 并发 | 2 路、4 路固定请求 | 全部成功且响应归属正确 |

所有请求固定 `temperature=0`；若版本支持 seed，也固定 seed。HTTP 200 只代表协议成功，必须保存并人工/脚本检查回答内容。

Native 与 fallback A/B 使用同一模型、同一请求、同一参数。文本不强制逐 token 完全相同，但答案必须语义等价；严格数值正确性由 tensor/kernel 测试保障。若接口可返回 logprob，则记录前若干 token 的 top-1 一致率和 logprob 差值。

### 6.4 显存检查

每阶段记录 XPU 4、5、6、7 的以下时点：

| 时点 | XPU 4 | XPU 5 | XPU 6 | XPU 7 |
|---|---:|---:|---:|---:|
| 启动前 |  |  |  |  |
| 模型加载完成、请求前 |  |  |  |  |
| 1K prefill 峰值 |  |  |  |  |
| 256-token decode 稳态/峰值 |  |  |  |  |
| 服务停止后 |  |  |  |  |

同时记录 allocator/log 中的权重加载统计。接入 IQ4、Q3_K、IQ3_S 后，对应 dense FP16 常驻量必须下降；若显存没有下降，优先检查：

- 是否仍构造并缓存了 dense 权重；
- `_xpu_dequant_rep_to_fp16` 是否被非预期调用；
- 混合 shard 合并是否退回 FP16；
- 实际加载的是否为新 `.so`；
- import 失败是否使整个 kernel tuple 退化。

### 6.5 基础性能检查

性能测试前完成至少一次 warmup，编译和首次初始化时间单独记录。固定请求内容、输入/输出长度、并发、服务参数和测量工具：

| 场景 | 固定条件 | 指标 |
|---|---|---|
| 短请求 | batch 1，短 prompt，输出 128 token | TTFT、TPOT、decode tok/s |
| Prefill | 约 1K input，输出 1～8 token | TTFT、input tok/s |
| Decode | 短 input，输出 256 token | TPOT、output tok/s |
| 并发 | concurrency 2、4 | 总吞吐、p50/p95、失败数 |

每项至少测 3 次，记录每次结果和中位数。阶段内以 native 对同类型强制 fallback 做 A/B；不同阶段之间只有测试配置和环境完全一致时才比较。正确性失败时性能数据只作诊断，不作优化结论。

### 6.6 30000/30001 生命周期与恢复

- 启动 30001 前，确认 30000 已按记录受控停止，30000 端口已释放，卡 6/7 显存已回落。
- 测试过程中确认 30001 进程只看到 XPU 6、7，卡 4、5 没有来自本任务的显存增长。
- 只停止记录下来的 30001 PID，并确认 30001 已释放、卡 6/7 显存回落。
- 使用启动前保存的原命令、环境和工作目录恢复 30000；不得临时猜测参数。
- 请求恢复后的 30000 `/health`，并执行与停止前相同的 smoke request。
- 比较恢复前后的模型 ID、关键启动参数、输出、卡 6/7 idle 显存和日志异常。
- 如果 30000 无法按原状态恢复，立即停止后续阶段并保留现场。

## 7. 分阶段实施计划

## 阶段 0：建立可复现基线

### 工作内容

- [x] 记录仓库 commit、dirty diff 和 `gguf.py` SHA256。
- [x] 记录容器状态、容器内实际源码路径和 SHA256。
- [x] 记录 30000 状态；开始阶段 1 时该服务已经由外部停止，端口为空，无可恢复进程。
- [x] 将“本轮无需停止/恢复 30000”作为显式 N/A 记录，而不是虚构恢复结果。
- [x] 记录 XPU 4～7 的空闲/现有服务显存。
- [x] 固化正确性 prompt、性能参数和结果保存位置。
- [x] 保存 Qwen3.8 tensor 名称、shape、量化类型清单，避免后续凭印象判断覆盖率。

### 阶段门禁

- [x] 能明确识别并分别操作 30000、30001，且不会使用宽泛进程匹配。
- [x] 已记录 30000 在阶段 1 开始前不存在，因此该轮恢复项为 N/A。
- [x] 日志中没有把未执行项目标为通过。

## 阶段 1：Q4_K GDN `ssm_out` 完整闭环

Q4_K 已有 XPU GEMV kernel。本阶段不开发新 GEMV，只补齐压缩态 value-head 列重排。

### 实现

- [x] 为 `_xpu_repack_q4_k(qweight, col_perm=None)` 增加参数。
- [x] 在 element-order nibble 重新打包之前执行 `_q5q6_col_perm_elems(nib, col_perm)`。
- [x] 以 `head_v_dim // 32` 粒度重排 scale 和 min。
- [x] `_xpu_repack_q4_k_chunked(..., col_perm=None)` 将参数传入每个 chunk。
- [x] `_xpu_prepare_shard()` 删除 Q4_K 的禁止断言并传递 `col_perm`。
- [x] 保持 row chunking，避免完整 `[N,K] int32` 临时量随大 tensor 膨胀。
- [x] 限制 large FP16 fallback 的 transpose cache，避免首次 forward 永久复制全部 fallback 权重导致 OOM。

TP2 预期 `col_perm=(3, 8, 128)`；128 可被 Q4_K 的 32-element scale group 整除。仍需在代码中校验约束，不能只依赖当前模型。

### 局部验证

- [x] 无 `col_perm` 的默认/显式 `None` repack 输出完全一致。
- [x] chunked 与 non-chunked 输出完全一致。
- [x] deterministic synthetic tensor 覆盖 nibble、scale、min 重排。
- [x] 五个实际 Q4_K `ssm_out` 的全部 5120 行均已测试。
- [x] 对原始 GGUF reference dequant 后执行 `[ratio,nk,hvd] -> [nk,ratio,hvd]`。
- [x] 分别验证 TP rank 0、rank 1 的 repack + `_xpu_dequant_q4_k`。
- [x] 记录 max abs/mean abs；最坏 max abs `4.3106e-4`，mean abs 约 `8e-6`，无 NaN/Inf。

### E2E 门禁

- [x] 30001 完整加载成功，日志不再出现 Q4_K col-perm assert。
- [x] 固定正确性请求集全部通过；thinking 模式下需给足 reasoning token，性能测试固定关闭 thinking。
- [x] 完成启动前、加载后、首次 forward 后、压力后和停止后的显存记录。
- [x] 完成短请求、约 1.5K prefill、256-token decode、并发 2/4 基础测试。
- [x] 日志明确说明 IQ4/Q3/IQ3 此时仍为 FP16 fallback。
- [x] 本轮开始前 30000 已不存在，恢复项为 N/A；30001 仅使用 `ZE_AFFINITY_MASK=6,7`。

## 阶段 2：IQ4_NL / IQ4_XS 原生纵向切片

### 实现

- [x] 分别实现 IQ4_NL、IQ4_XS 的 row-chunked canonical repack。
- [x] 将二者归一为共享的 packed LUT-index + final-scale ABI。
- [x] 实现 `esimd_gemv_iq4` 和 `esimd_gemv_iq4_m`，覆盖 M=1/2/4/8/16。
- [x] 为 IQ4_XS GDN `ssm_out` 实现压缩态 `col_perm`。
- [x] 接入 `_xpu_prepare_shard`、matmul、dense reconstruction、row permutation、merge/group 和 output-slice dispatch。
- [x] 支持与 Q4_K/Q5_K/Q6_K/Q8_0 等类型混合的输出 shard。
- [x] 增加独立的 `SGLANG_GGUF_XPU_NO_IQ4=1` fallback 开关。
- [x] 新 kernel 符号单独 import，旧 `.so` 缺少新符号时不得影响旧 kernel。

### 局部验证

- [x] synthetic block 覆盖 LUT index、负 subscale 和 canonical scale；kernel 前补充 scale 极值专项。
- [x] 与 `gguf.dequantize()` 比较 canonical dequant。
- [x] 全部 124 个实际 IQ4 tensor 验证首/中/末行；五个 `ssm_out` 全 5120 行分块验证。
- [x] kernel 在 M=1/2/4/8/16 上与 dense matmul 比较。
- [x] 验证非连续输出 slice、多个 shard 写入同一输出、same-kind merge 和 mixed-kind group。
- [x] 验证五个 IQ4_XS `ssm_out` 的 TP rank 0/1 col-perm。
- [x] 验证禁用 IQ4 后仅 IQ4 回退，其他 native kernel 不受影响。

### E2E 门禁

- [x] native IQ4 与 `SGLANG_GGUF_XPU_NO_IQ4=1` 分别启动 30001。
- [x] 两条路径固定正确性请求语义一致；并发数字序列退化在两条路径都可复现，另行记录为非 IQ4 专属问题。
- [x] native IQ4 权重常驻相对 fallback 下降约 9.60 GB/rank；固定 0.8 memory fraction 会把释放空间分给 KV cache，故服务最终显存接近。
- [x] 完成同条件性能 A/B，报告每次值和中位数；native 单路 decode 提升，但 TTFT/prefill/并发尚未优化。
- [x] 权重加载统计确认 IQ4 不再 dense 常驻；Q3_K/IQ3_S 仍为 FP16 fallback。
- [x] 本阶段开始前 30000 已不存在，因此恢复为 N/A；测试进程只设置 `ZE_AFFINITY_MASK=6,7`，未使用卡 4/5。

阶段 2 的正确性与显存门禁通过。性能结果仅说明当前 kernel 的适用区间：M=1 decode 有收益，大 M prefill 和并发仍应在阶段 6 基于 profiling 优化，不能据此提前引入 fusion。

## 阶段 3：Q3_K 原生纵向切片

### 实现

- [x] 实现 row-chunked Q3_K canonical repack。
- [x] 实现 `esimd_gemv_q3_k` 和 `esimd_gemv_q3_k_m`，覆盖 M=1/2/4/8/16。
- [x] 完整接入 rep、dense reconstruction、merge/group、mixed output-slice dispatch。
- [x] 增加 `SGLANG_GGUF_XPU_NO_Q3K=1` fallback 开关。

### 局部验证

- [x] synthetic block 覆盖 low bits、high/sign mask 和 signed scale 边界。
- [x] 七个实际 Q3_K tensor 全部做 reference dequant 检查。
- [x] 所有实际 K 和 TP-local K 的 kernel shape 均被覆盖。
- [x] M=1/2/4/8/16 与 dense matmul 对比通过。
- [x] mixed shard、output slice 和单类型 fallback 通过。

### E2E 门禁

- [x] native Q3_K 与 `SGLANG_GGUF_XPU_NO_Q3K=1` 做正确性、显存、性能 A/B。
- [x] 日志确认没有 Q3_K dense 常驻。
- [x] IQ4 native 路径没有回归。
- [x] 30000 原本不存在，恢复为 N/A；卡 4/5 未被本任务使用。

## 阶段 4：IQ3_S 原生纵向切片

### 实现

- [ ] 实现 row-chunked IQ3_S canonical repack。
- [ ] 实现 `esimd_gemv_iq3_s` 和 `esimd_gemv_iq3_s_m`，覆盖 M=1/2/4/8/16。
- [ ] 完整接入 rep、dense reconstruction、merge/group、mixed output-slice dispatch。
- [ ] 增加 `SGLANG_GGUF_XPU_NO_IQ3S=1` fallback 开关。

### 局部验证

- [ ] synthetic block 覆盖 magnitude grid、sign 和 scale 边界。
- [ ] 四个实际 IQ3_S tensor 全部做 reference dequant 检查。
- [ ] M=1/2/4/8/16 与 dense matmul 对比通过。
- [ ] mixed shard、output slice 和单类型 fallback 通过。

### E2E 门禁

- [ ] native IQ3_S 与 `SGLANG_GGUF_XPU_NO_IQ3S=1` 做正确性、显存、性能 A/B。
- [ ] 日志确认没有 IQ3_S dense 常驻。
- [ ] IQ4、Q3_K native 路径没有回归。
- [ ] 30000 已按原配置恢复；卡 4/5 未被本任务使用。

## 阶段 5：全量收口与稳定性

### 覆盖检查

- [ ] 输出每种 GGUF 类型的 tensor 数、压缩常驻数、FP16 fallback 数和字节数。
- [ ] IQ4_NL、IQ4_XS、Q3_K、IQ3_S 的 FP16 fallback 数为 0。
- [ ] 检查所有混合组合，不仅是单类型矩阵。
- [ ] Qwen3.6 GGUF 做完整回归，至少覆盖加载、固定正确性集和基础性能。
- [ ] 旧 kernel wheel 缺少新符号时能安全 fallback，且已有 Q4/Q5/Q6/Q8 kernel 不被整体禁用。

### 稳定性 E2E

- [ ] 冷启动至少 3 次。
- [ ] 连续请求至少 30 分钟，无内存持续增长、hang 或错误累积。
- [ ] 1K/4K prefill，128/256-token decode。
- [ ] concurrency 1/2/4；如资源允许再测 8。
- [ ] API 超时、取消请求、服务停止后的资源回收正常。
- [ ] 最终记录 native、逐类型 fallback 和可行的全 fallback 对照。
- [ ] 每轮 30000 均完成受控停止和同配置恢复，没有遗留状态差异。

## 阶段 6：性能分析与可选优化

只有阶段 5 全部通过后才开始：

- profile 决定瓶颈是否在 GEMV、dequant、大 M prefill、GDN、通信或 launch overhead；
- 再评估 fusion、Q8 `ssm_alpha/beta` 路径和大 M DPAS；
- 每个优化仍执行相同正确性、显存和 E2E 回归；
- 优化结果必须同时报告收益、波动、额外显存和适用 shape，不能只报告最好一次。

## 8. 数值通过标准

阈值需在第一次 reference 测试后依据量化格式和 FP16 累积误差冻结，并写入日志。冻结前采用以下原则：

- repack + dequant：与 `gguf.dequantize()` 比较；先确认元素映射完全正确，再区分 FP16 rounding。
- kernel：同一 canonical rep 下，以 FP32 accumulation 的 dense reference 为主要参考，同时记录 FP16 dense reference。
- 同时报 `max_abs`、`mean_abs`、`max_rel`、cosine similarity；只报一个最大误差不足以判断。
- 对接近零的元素，不以相对误差作为唯一标准。
- 检查 NaN/Inf，任何新增 NaN/Inf 都直接失败。
- 重要 shape 验证全输出；超大 tensor 可以按固定随机种子抽样，但 repack 的首/中/末块必须覆盖。

任何需要放宽阈值才能通过的情况，都必须先记录误差分布和原因，不能直接修改测试。

## 9. 必须覆盖的集成点

新增原生表示不能只改 `_xpu_prepare_shard()`，至少检查：

```text
_xpu_prepare_shard
_xpu_shard_matmul
_xpu_dequant_rep_to_fp16
_xpu_perm_rep_rows
_xpu_try_merge_shards
_xpu_group_shards
_xpu_groups_m_ok
_xpu_rep_gemv_into
_xpu_rep_gemv_m_into
_XPU_GEMV_OUT_KINDS
```

实际层中存在混合格式，例如：

```text
qkvz:    [Q4_K, Q4_K, Q4_K, IQ4_XS]
gate/up: [IQ4_XS, Q3_K]
qkv:     [IQ4_XS, Q6_K, Q8_0]
```

因此“单 tensor kernel 正确”不足以证明模型路径正确；output-slice 写入、分组、合并和不同 kind 混排必须单独测试。

## 10. 阻塞、回退与回滚规则

- 局部数值失败：停止 E2E，保存最小复现 tensor、shape、rank、误差和 repack 中间值。
- 服务加载失败：保存完整日志、最后成功 tensor、XPU OOM 信息和显存时间线。
- 输出错误但 kernel 单测通过：优先检查 col_perm、shard 顺序、mixed output slice、merge/group 和实际加载 `.so`。
- 显存未下降：检查 dense 缓存和 fallback 计数，不以服务能运行作为通过。
- 性能退化：先保证正确性，使用单类型 fallback A/B 定位；未经 profile 不做 fusion。
- 30000 无法恢复、卡 6/7 未释放或卡 4/5 被误用：立即停止 30001 的记录 PID，保留现场，不继续下一阶段。
- 每个新类型都保留独立环境变量 fallback；回滚不依赖覆盖全局 wheel，也不回退用户无关改动。

## 11. 每阶段交付物

每个阶段结束时必须同时存在：

- 对应源代码和测试代码；
- 可复现的构建产物及 SHA256（涉及 native kernel 时）；
- reference/kernel 测试原始输出；
- 30001 完整服务日志；
- 正确性请求和响应；
- 显存五时点数据；
- 性能每次测量及中位数；
- 30000 停止/恢复前后对比及卡 4/5 隔离性记录；
- 开发验证日志中的结论、遗留问题和下一步决定。

只有上述材料齐全，阶段状态才能从 `进行中` 改为 `通过`。
