# Onyx Tool Calling Status

更新时间：2026-07-23

本文档记录 Onyx 在 SGLang 上的 tool calling 协议、服务端行为、当前验证结果和已知边界。当前实现位于 `dev-bmg-onyx` 分支，面向 Intel XPU/BMG TP=2 online-FP8 服务。

## 1. 当前结论

当前 Onyx tool calling 已具备可交付的单调用能力：

- 支持 OpenAI-compatible `/v1/chat/completions`。
- 支持 `tool_choice=auto`、`required`、named function 和 `none`。
- 服务端强制每个 assistant response 最多返回一个工具调用，不依赖客户端的 `parallel_tool_calls` 设置。
- 支持流式参数增量传输和非流式响应。
- 工具参数必须是 JSON object，并按 Draft 2020-12 JSON Schema 完整校验。
- `to=self` 内部规划帧不会暴露给客户端；其后的 `<|eom|>`作为消息分隔符，继续生成到 `to=user` 或第一个工具调用。
- `to=user` 被解包为普通 assistant content。
- 未知 recipient、第二个工具调用、非法 JSON 和 schema 不匹配均 fail closed。
- 无 tools 或 `tool_choice=none` 时仍执行 Onyx 协议清理，避免内部 token 泄漏。

当前服务运行于：

```text
http://10.112.229.59:31888
```

服务配置为 TP=2、65,536 context、65,536 max total tokens、radix cache enabled、`max_running_requests=1`、XPU graph disabled、decoder online FP8、FP8 LM head disabled。

## 2. Onyx 原生协议

Onyx 使用 recipient 表示消息目标。

| 语义 | 原生输出 |
|---|---|
| 用户可见回复 | `assistant to=user<|message|>...<|eot|>` |
| 工具调用 | `assistant to=<tool_name><|message|>{...}<|eom|>` |
| 内部规划 | `assistant to=self<|message|>...<|eom|>` |

当前 checkpoint 的关键 special-token ID 为：

| Token | ID |
|---|---:|
| `<|eom|>` | 200007 |
| `<|eot|>` | 200008 |
| `<|start|>` | 200022 |
| `<|message|>` | 200023 |

代码不会硬编码这些 ID。OpenAI serving 层从当前 tokenizer 动态解析 `<|eom|>`、`<|eot|>`和 `<|message|>`，并编码 `to=self` recipient 前缀后传给 scheduler。

## 3. Recipient-aware 终止行为

Onyx 模型的 generation config 将 `<|end_of_text|>`、`<|eom|>`和 `<|eot|>`都声明为 EOS。直接采用普通 EOS 逻辑时，模型在输出第一个 `to=self ... <|eom|>`后就会停止，parser 隐藏 self 内容后只能返回空成功响应。

当前 scheduler 在命中 EOS token 时按当前 recipient 判断：

| 当前帧 | 结束 token | 行为 |
|---|---|---|
| `to=self` | `<|eom|>` | 隐藏该帧，将 token 作为消息分隔符并继续 decode |
| 工具 recipient | `<|eom|>` | 正常结束生成，返回工具调用 |
| `to=user` | `<|eot|>` | 正常结束生成，返回用户可见正文 |
| 任意帧 | `<|end_of_text|>` | 正常结束生成 |
| malformed/无法识别 | 任意 EOS | fail closed，按终止处理 |

判断完全基于 output token IDs：

1. 确认命中的 token 是 `<|eom|>`。
2. 从当前位置向前寻找最近的 `<|message|>` token ID。
3. 检查 `<|message|>`之前是否为 tokenizer 编码后的 `to=self` recipient 前缀。
4. 仅该条件成立时继续生成；其他 EOS 均终止。

服务端将普通 `ignore_eos`强制保持为 `false`。因此该行为不是全局忽略 EOS，也不需要 parser 在流式输出后异步 abort scheduler。

## 4. OpenAI 请求策略

### 4.1 单调用能力边界

Onyx serving 始终执行：

```text
parallel_tool_calls = false
```

即使客户端显式发送 `parallel_tool_calls=true`，服务端也会覆盖为 `false`。每个 assistant response 最多包含一个 tool call。

多步骤任务的预期循环为：

```text
assistant tool_call A
→ client/Agent 执行 A
→ tool result
→ assistant 决定下一次调用或回复用户
```

生成阶段发现第二个工具 recipient 时不会静默截断，而是返回协议错误。

### 4.2 `tool_choice`

| 模式 | 当前行为 |
|---|---|
| `auto` | XGrammar 约束完整的 `self* → user 或 schema tool` 联合结构；模型在合法分支间决定 0 或 1 个调用 |
| `required` | 使用 Onyx native structural constraint，必须生成一个合法调用 |
| named function | structural constraint 只允许指定 recipient |
| `none` | 不生成工具调用，但仍清理 `to=self`、`to=user`和协议 token |

`auto`将完整响应编译为一个 XGrammar EBNF：最多 8 个私有 `to=self`规划帧，随后必须选择显式 `to=user`、训练格式中的 bare `<|message|>`普通正文，或一个合法工具 recipient。每个工具的 recipient 前缀、JSON Schema 和 `<|eom|>`先做规则名隔离，再合并到同一个 grammar；因此选中 recipient 后不能串入其他工具的参数语法。普通正文仍是自由文本，不会把普通问答强制成工具调用。

这里刻意不再使用多个 StructuralTag `or`/`sequence`节点拼接工具 Schema。17-tool 13K 请求证明该组合会错误合并兄弟 grammar 状态：`todo_write`可能消费其他工具允许的字段顺序，最后进入无合法 next token 的死状态。单一 namespaced EBNF 在 recipient 处完成不可逆分流。

### 4.3 历史消息

模板支持标准 OpenAI history：

- `assistant.tool_calls`
- `tool_call_id`
- tool result
- legacy `recipient`

新生成严格单调用，但输入历史可以包含旧的 parallel tool calls。服务端会把多调用 assistant history 和对应 tool results 展开为顺序的 recipient/tool pairs，再渲染为 Onyx 原生协议。

## 5. 参数与 Schema 校验

服务端会将所有 Onyx tool definitions 视为 strict：

```text
tool.function.strict = true
```

请求阶段：

- 规范化常见 DB/ORM-style schema type aliases。
- 使用 `Draft202012Validator.check_schema()`检查 schema 本身。
- 无效 schema 返回请求错误。

生成阶段：

- recipient 必须对应请求中声明的工具。
- arguments 必须是完整合法 JSON。
- JSON 顶层必须是 object。
- object 必须通过所选工具的 Draft 2020-12 schema。
- 流式响应会在结束时对拼接后的完整 arguments 再做一次 finalize 校验。

不会将非法调用降级为普通正文，也不会用空 object 伪装成功。

## 6. Parser 与流式输出

### 6.1 非流式

非流式 parser 可以处理：

```text
to=self ... <|eom|>
<|start|>assistant to=self ... <|eom|>
<|start|>assistant to=<tool> ...
```

一个或多个 self frame 会被跳过，直到遇到：

- `to=user`：返回清理后的 content；
- 合法工具 recipient：返回一个 OpenAI `tool_calls` item；
- 非法协议：抛出 `ToolCallParseError`。

### 6.2 流式

流式 parser 跨 chunk 保存协议状态：

- 缓冲可能被拆分的 `<|start|>`、`<|message|>`、`<|eom|>`和 `<|eot|>`。
- self payload 全程隐藏。
- self `<|eom|>`后重置为下一 recipient frame。
- 工具名只发送一次。
- arguments 以 OpenAI tool-call delta 形式逐段发送。
- 流结束时拼接完整 arguments 并执行 JSON/schema 校验。
- 生成的 tool result header、未知 recipient 或第二次调用不会泄漏为 content。

成功工具调用的最终 `finish_reason`为：

```text
tool_calls
```

## 7. Chat Template 策略

生产模板：

```text
benchmark/onyx/onyx_tool_chat_template.jinja
```

启动脚本显式传入该模板，避免 checkpoint 内残留旧模板。模板当前规定：

- 只为确实需要外部信息或外部动作的请求调用工具。
- 先用一条简短的 `to=self`私有判断决定是否需要工具，不向用户发送工具计划旁白。
- 有对应计算或转换工具时使用该工具；没有相关工具时，算术、翻译、解释和稳定常识保持普通回答。
- 能用 status/inspection 工具获取的信息不向用户追问；请求中已有的 token/code 直接传给兼容工具。
- 每个 assistant response 最多调用一个函数。
- 多步骤任务只调用下一项，等待 tool result 后再决定后续。
- 工具失败后不得用完全相同的参数重复调用；应修改参数、调整计划或说明阻塞。
- assistant tool call 使用 `<|eom|>`，普通用户回复使用 `<|eot|>`。

这些规则是当前 SGLang 集成 prompt，不是已恢复的 Onyx 原始训练 prompt。

## 8. 当前验证结果

### 8.1 定向单测

以下三组定向测试当前通过：

```text
test/registered/unit/managers/test_conditional_eos_stop.py
test/registered/unit/function_call/test_onyx_detector.py
test/registered/unit/entrypoints/openai/test_serving_chat.py
```

结果：

```text
126 passed, 9 subtests passed
```

覆盖范围包括：

- self `<|eom|>`继续到 tool/user；
- 连续多个 self frame；
- tool `<|eom|>`和 `<|eot|>`正常终止；
- 客户端 `ignore_eos=true`不能绕过 Onyx 终止策略；
- 单调用服务端覆盖；
- `auto`零调用和单调用；
- required/named structural constraint；
- schema 合法与非法参数；
- 流式 chunk 拆分；
- protocol token 泄漏防护；
- parallel history 顺序展开。

### 8.2 原始 13K 请求

输入：

```text
/home/intel/xiangyu/13k.json
```

该请求会先生成内部规划，再调用 `todo_write`。当前结果：

| 指标 | 结果 |
|---|---|
| Prompt tokens | 14,553 |
| Completion tokens | 182 |
| Tool calls | 1 |
| Tool name | `todo_write` |
| Finish reason | `tool_calls` |
| Schema | valid |
| Protocol leakage | none |
| HTTP / SSE | 200 / complete `[DONE]` |
| Latency | 13.484 s |

这条请求验证了：

```text
to=self ... <|eom|> → to=todo_write
```

不会再出现“隐藏 self 后返回空成功响应”。

原始 SSE：

```text
/home/intel/xiangyu/copilot_workspace/onyx_13k_complete_grammar_validation.sse
```

历史 self-EOM 版本输出仍保留在：

```text
/home/intel/xiangyu/copilot_workspace/onyx_13k_self_eom_validation.sse
```

### 8.3 原始 20K 请求

输入：

```text
/home/intel/xiangyu/20k.json
```

当前结果：

| 指标 | 结果 |
|---|---|
| Prompt tokens | 19,293 |
| Cached tokens | 14,336 |
| Completion tokens | 71 |
| Tool calls | 1 |
| Tool name | `run_shell_command` |
| Finish reason | `tool_calls` |
| Schema | valid |
| Protocol leakage | none |

原始 SSE：

```text
/home/intel/xiangyu/copilot_workspace/onyx_20k_self_eom_regression.sse
```

### 8.4 BFCL 首轮空调用定向回归

首次完整 `multi_turn_base` 测试中有 63/200 条样本在首轮直接输出工具计划或追问，没有实际调用工具。修复后对这 63 条原失败样本使用最终服务、`tool_choice=auto`、`temperature=0.001`重新测试：

| 指标 | 结果 |
|---|---:|
| 实际工具调用 | 62/63 |
| 仍为普通回答 | 1/63 |
| 非法 JSON arguments | 0 |
| 非 object arguments | 0 |
| inference error | 0 |

普通回答/工具分流烟测同时通过：算术、翻译、解释和问候保持 `to=user`；实时天气请求进入 schema-constrained `get_weather`。剩余 `multi_turn_base_171`偶发错误要求重新认证，属于模型在合法 user/tool 分支间的语义选择问题，不是 recipient 或 arguments 结构错误。

### 8.5 完整 grammar 的 BFCL 定向回归

选取旧实现正确、首版 auto union 因循环而失败的 20 个 `multi_turn_base`样本复测：

| 指标 | 结果 |
|---|---:|
| Accuracy | 75.00% (15/20) |
| inference error | 0 |
| force terminated | 1 |
| execution response mismatch | 3 |
| instance state mismatch | 1 |

结果与 bare-final 修复版同为 15/20，失败集合完全相同。20 条中 17 条完整结果逐字一致；其余 3 条里，新 grammar 消除了 `base_1`和 `base_149`中的畸形/冗余参数调用。结果与评分文件：

```text
/home/intel/xiangyu/cc_workspace/gemma_splitk/_bfcl_vendor/result/onyx-complete-grammar/multi_turn/BFCL_v4_multi_turn_base_result.json
/home/intel/xiangyu/cc_workspace/gemma_splitk/_bfcl_vendor/score/onyx-complete-grammar/multi_turn/BFCL_v4_multi_turn_base_score.json
```

### 8.6 BFCL `multi_turn_base` 完整集

使用最终 complete grammar、`tool_choice=auto`和`temperature=0.001`完成全部 200 条测试：

| 指标 | 修复前 | 首版 auto union | 最终 complete grammar |
|---|---:|---:|---:|
| Accuracy | 23.00% (46/200) | 18.00% (36/200) | **46.50% (93/200)** |
| empty turn | 63 | 1 | **2** |
| instance state mismatch | 55 | 50 | 58 |
| execution response mismatch | 26 | 16 | 29 |
| force terminated | 9 | 97 | **18** |
| inference error | 1 | 0 | **0** |

相对修复前，新增正确 55 条、退化 8 条，净增 47 条。原先 63 个 empty-turn 样本中，23 条转为正确、1 条仍为 empty turn，其余进入可解析但语义或执行不正确的工具轨迹。相比首版 auto union，complete grammar 将 force termination 从 97 降至 18，说明 bare final 分支和不可串扰的 namespaced EBNF 同时避免了“大量强制工具循环”和跨工具 schema 状态泄漏。

完整结果共包含 1,852 次工具调用：非法 arguments JSON 为 0，非 object arguments 为 0，结构合法率 100%。200 条结果均成功落盘，ID 唯一，无 API/inference error；server 日志中没有 grammar accept、structural tag 或 traceback 错误。剩余失分主要是模型选错工具、错误执行规划和重复调用直到 20-step 上限，不是 arguments 结构错误。

结果与评分文件：

```text
/home/intel/xiangyu/cc_workspace/gemma_splitk/_bfcl_vendor/result/onyx-complete-grammar-full/multi_turn/BFCL_v4_multi_turn_base_result.json
/home/intel/xiangyu/cc_workspace/gemma_splitk/_bfcl_vendor/score/onyx-complete-grammar-full/multi_turn/BFCL_v4_multi_turn_base_score.json
```

BFCL 类别评分正常产出；最终 leaderboard CSV 汇总仍会因为自定义 alias 未注册而报既有`KeyError: 'onyx-hf-FC'`，不影响上述逐样本评分文件。

### 8.7 OpenGame 无 gateway 端到端复测

使用 OpenGame 0.6.0、最终 complete grammar 和本地 31888 endpoint，在全新目录中执行 focused Breakout coding run。CLI 直连 SGLang，不使用历史 first-turn/JSON-repair gateway；项目级 npm proxy 在启动前配置并验证，全程没有人工修改生成源码或打断 agent loop。

| 指标 | 结果 |
|---|---:|
| OpenGame turns | 25 |
| API logs | 25 |
| Tool calls | 24 |
| Tool results success | 24/24 |
| 非 object arguments | 0 |
| API / grammar error | 0 |
| CLI result | success / exit 0 |
| Duration | 400.987 s |
| Production build | success |
| Dev server smoke | HTTP 200 |

首轮请求包含 4 条 messages 和 17 个 tools，省略`tool_choice`（auto），直接生成合法`todo_write`，不再需要 gateway 强制`required`。一次约 14.7 KB 的`GameScene.js`长`write_file`也正常完成，没有 raw control character、invalid JSON 或 streaming finalize 错误。

模型初次 build 复现了两项跨文件一致性错误：webpack 引用了未声明的`babel-loader`，CopyWebpackPlugin 指向空 assets 目录。模型在收到真实 build 输出后自行安装 Babel 依赖、加入`noErrorOnMissing: true`并重新 build 成功，随后启动 dev server、验证 HTTP 200 并停止进程。这说明 server 修复后的 OpenGame tool loop 和 build-error recovery 可以完整运行。

该 focused run 不等同于 OpenGame 官方含 GDD/外部资产服务的完整 agent-test。生成代码仍有模型质量问题：安装的是 Phaser 3.90，但粒子代码调用了 3.60 起会抛错的`ParticleEmitter.createEmitter()`；同时球启用了四边 world bounds，导致`ball.y > 600`掉球/扣命条件不可达。它们不属于 tool-call 协议错误，也说明 build 和 HTTP 200 不能替代实际 gameplay 验证。

生成项目和 OpenAI 日志：

```text
/home/intel/xiangyu/test/opengame-breakout-direct-clean-20260723
/home/intel/xiangyu/test/opengame-breakout-direct-clean-20260723/openai-logs
```

## 9. 启动配置

启动入口：

```text
benchmark/onyx/launch_onyx_fp8_tp2.sh
```

当前 64K LAN 服务使用：

```bash
ONYX_CONTEXT_LENGTH=65536 \
ONYX_MAX_TOTAL_TOKENS=65536 \
ONYX_ALLOW_LONG_CONTEXT=1 \
ONYX_PORT=31888 \
ONYX_MAX_RUNNING_REQUESTS=1 \
bash benchmark/onyx/launch_onyx_fp8_tp2.sh
```

关键参数：

```text
--tool-call-parser onyx
--chat-template benchmark/onyx/onyx_tool_chat_template.jinja
--grammar-backend xgrammar
--sampling-backend pytorch
--tp 2
--dtype float16
--quantization fp8
--load-format layered_fp8
--max-running-requests 1
--disable-cuda-graph
--enable-cache-report
```

当前配置没有设置 `--disable-radix-cache`，因此使用 `SWARadixCache`。`enable_fp8_lm_head=false`，即 decoder linear 使用 online FP8，但 LM head 保持非 FP8 路径。

## 10. SGLang 与 Agent Runtime 的责任边界

SGLang `/v1/chat/completions`负责：

- 渲染工具定义和历史；
- 生成 0 或 1 个 tool call；
- 隐藏 self planning；
- 清理协议 token；
- 校验 recipient、JSON 和 schema；
- 返回 OpenAI-compatible content/tool-call stream。

SGLang 不负责：

- 实际执行工具；
- 自动追加 tool result；
- 完整 Agent loop；
- 相同错误连续出现时的计数、中止和重规划；
- 跨轮总体 tool budget；
- 工具副作用、权限和网络访问控制。

这些行为必须由调用方 Agent runtime 实现。

## 11. 已知限制与下一阶段

1. **只支持单调用生成。** 客户端不能通过 `parallel_tool_calls=true`启用并行生成。
2. **self-loop budget 是 grammar 上限。** `auto`最多允许 8 个 self frame、每帧最多 4,096 个非`<`字符；尚未增加重复 self 内容检测。
3. **Agent 重试策略不在服务端。** 模板只有行为提示，没有硬性的连续错误计数器。
4. **远程 JSON Schema `$ref`安全尚未加固。** resolver、网络访问和 SSRF 等安全策略按计划放到下一阶段；当前完成的是功能正确性和本地 schema 校验。
5. **`auto`的语义分支仍依赖模型。** Structural grammar 已同时约束 `to=user`、`to=self`和合法工具 recipient，并对工具 arguments 施加 Schema；但模型仍可能在应调用工具时选择合法的 `to=user`普通回答。完整 BFCL 中有 2 条 empty turn，其中原 63 条首轮空调用样本仍有 1 条残留；主要剩余失分是 58 条 state mismatch、29 条 execution mismatch 和 18 条 force termination。
6. **64K 是显式 extrapolation 配置。** 模型声明 context length 为 16K；当前 64K 服务需要 `ONYX_ALLOW_LONG_CONTEXT=1`。此前 48K 已通过，64K 边界测试等待 context 配置提升到 128K 以上后再做。

## 12. 相关文件

| 文件 | 作用 |
|---|---|
| `benchmark/onyx/onyx_tool_chat_template.jinja` | 生产 chat template 和单调用 prompt |
| `benchmark/onyx/launch_onyx_fp8_tp2.sh` | TP=2 online-FP8 服务启动入口 |
| `python/sglang/srt/function_call/onyx_detector.py` | Onyx 非流式/流式协议 parser |
| `python/sglang/srt/function_call/function_call_parser.py` | parser 集成和 streaming finalize |
| `python/sglang/srt/function_call/core_types.py` | `ToolCallParseError`及 parser 类型 |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | OpenAI policy、schema、history 和响应转换 |
| `python/sglang/srt/managers/schedule_batch.py` | recipient-aware special-token EOS 判断 |
| `python/sglang/srt/sampling/sampling_params.py` | Onyx protocol token 配置 key |
| `test/registered/unit/function_call/test_onyx_detector.py` | parser 和 template 测试 |
| `test/registered/unit/entrypoints/openai/test_serving_chat.py` | OpenAI serving policy 测试 |
| `test/registered/unit/managers/test_conditional_eos_stop.py` | special-token 终止测试 |
| `ONYX_SGLANG_SUPPORT_STATUS.md` | Onyx 全模型支持与性能总状态 |
