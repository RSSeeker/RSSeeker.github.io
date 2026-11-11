# 神经网络训练器

一个纯前端、零安装依赖的神经网络可视化训练工具。直接在浏览器中搭建并训练神经网络，实时观察 Loss 曲线、混淆矩阵和决策边界。

## 快速开始

用浏览器打开 `index.html` 即可使用，无需安装任何环境。Loss 图表依赖 Chart.js（CDN 加载，含多源备用与离线降级提示），图表以外的所有功能离线可用。

## 功能一览

### 网络搭建
- 支持回归（MSE）和分类（Softmax + Cross Entropy）两种任务
- 自由配置输入/输出维度，任意增删隐藏层
- 激活函数：ReLU / Leaky ReLU / Sigmoid / Tanh
- 优化器：SGD / Momentum / RMSProp / Adam
- 正则化：Dropout + L2 权重衰减
- 支持 He 初始化

### 训练与评估
- 训练集 / 测试集自动划分，可自定义比例
- 基于验证 Loss 的 Early Stopping
- 双 Y 轴 Loss + 准确率实时曲线（Chart.js）
- 分类任务自动生成混淆矩阵，含 Precision / Recall / F1-Score
- 2D 输入分类任务自动绘制决策边界
- WebGL 硬件加速开关

### 数据与模型
- 内置 6 个预设示例：XOR、非线性拟合、正弦曲线、鸢尾花分类、螺旋分类、四象限分类
- 手动添加 / 编辑训练数据
- 模型导出为 JSON（含权重、超参数、训练历史）
- 支持模型导入恢复
- 独立推理面板，分类任务显示各类别概率

### 语言模型（页面底部独立面板）
- 字符级因果 Transformer（GPT 式掩码自注意力），内置在页面里，纯 JS 手写
- 训练语料、轮数、学习率、模型维度、滑动窗口可调；训练进度条 + loss 实时显示
- 训练完成后按提示词续写，可调续写长度、温度、top_p（nucleus 采样）、min_p，以及重复 / 存在 / 频率惩罚与禁重复 n-gram
- 训练结束显示困惑度（exp(平均交叉熵)）
- 支持把模型（参数 + 配置 + 词表）保存为 JSON，并重新加载继续生成

### 界面
- 深色主题，响应式布局，适配手机 / 平板 / 桌面

## 与 Python 仓库的同步范围

页面只保留「浏览器里可以直接体验」的部分；Python 主仓库（`Python-standard-library-only-AI-main`）里新增的训练算法 / 指标 / 服务端组件不会全部搬进页面。当前同步状态：

| 能力 | 状态 | 说明 |
|------|------|------|
| 字符级 GPT 训练与生成、采样控制（惩罚 / top_k / top_p / min_p / typical / top_a / eta / tail_free / mirostat） | ✅ 已同步 | 与 Python `CausalLM` 同结构，`tools/llm_core_test.js` 逐位对拍 |
| 前缀 KV 缓存与缓存生成（`llmForwardStep` / `llmGenerateCached`） | ✅ 已同步 | 对应 Python `PrefixCache` / `llm_generate_cached` |
| 对比搜索、正则 / CFG / 字符级 JSON 约束解码 | ✅ 已同步 | 对应 Python `contrastive_search` / `RegexGrammar` / `CFGGrammar` / `json_grammar` |
| 指标（METEOR / chrF / WER / CER / STOI / PESQ / LPIPS / FVD 等） | ⛔ 不引入 | 纯计算在页面里没有交互价值，Python 与 C 侧已提供 |
| 投机解码（Medusa / EAGLE）、扩散 / Flow Matching 采样器、TTS、离线 RL、EGNN | ⛔ 暂不引入 | 需要训练出的草稿头 / 去噪网络等运行时状态，或与「分类训练器」主题无关；Python 与 C 侧已提供 |
| ONNX 导出、训练技术（ZeRO / 梯度检查点 / FP8 等） | ⛔ 不引入 | 属训练 / 导出工具，不是页面交互能力 |

C 侧（`AI_C` 仓库）提供上述纯计算 / 导出 / 训练模拟能力，并实现了 LLaMA / BERT / ViT / ELECTRA / RWKV / RetNet / xLSTM / Mamba / DeBERTa 的单向前向（权重由 Python 导出后逐位对拍）。最近同步的是 Python 主仓库新增的纯计算与结构类能力：NF4 / 双重量化、WeightStandardization、GradScaler、MoE 负载均衡、GRPO / Best-of-N / 拒绝采样 / 自洽性 / μP、稀疏自编码器 SAE、树状注意力草稿树与掩码、基于向量一致性的幻觉检测、PagedAttention 块分配器，以及 KAN（B 样条边函数层与多层模型，含前向/反向/SGD）；均为 C 实现并配 `tools/*_reference.py` + `tests/compare_*.c` 逐位或固定容差对拍。详见该仓库 README 的「与 Python 版本的功能对照」。

同步原则：页面里出现的数值算法必须能在 `tools/` 的 Node 校验里与 Python 参考逐位或按固定容差对齐；未同步的 Python 能力以主仓库 README / ROADMAP 为准。

## 网络架构

整体采用 **特征提取 + MLP 分类/回归** 的二段式结构：

```
输入 → [架构模块] → 特征向量 → MLP 隐藏层 → 输出层
```

架构模块支持四种可选方案：

| 架构 | 说明 | 可配参数 |
|------|------|----------|
| **MLP** | 全连接，输入直通，不做额外特征变换 | — |
| **CNN 1D** | 一维卷积，适合信号峰值检测等局部特征提取 | 卷积核大小、滤波器数量 |
| **RNN** | 循环网络，将序列逐步编码为隐藏状态，适合时间序列预测 | 隐藏维度 |
| **Attention** | 多头自注意力 + 均值池化，适合全局依赖建模与波形分类 | 嵌入维度、注意力头数 |

架构模块与后续 MLP 联合训练，共享同一优化器和损失函数。分类任务输出层经过 Softmax 得到类别概率。

## 因果语言模型

页面底部有一个独立的「因果语言模型（字符级 GPT）」面板，不依赖上面的分类/回归流水线，用来直观展示 GPT 的核心机制：

```
字符 → 词向量 + 正弦位置编码 → 因果多头自注意力 → 残差 + LayerNorm → 前馈 → 词表头 → 下一个字符
```

- **因果掩码**：注意力分数矩阵的上三角（`j > i`）在 softmax 前置为 `-Infinity`，softmax 后权重恰为 `0`，因此每个位置只能看到自己和左边的字符；反向传播时这些位置梯度天然为零，不需要额外分支。
- **滑动窗口**：窗口 `W > 0` 时，在因果掩码之外再把 `j < i - W + 1` 的位置置为 `-Infinity`，每个位置只看最近 `W` 个字符（`W = 1` 即只看自己）；`W = 0` 退回全注意力，与原来的行为逐位一致。
- **训练**：把整段语料当成一条序列，每个位置预测下一个字符，交叉熵损失 + Adam 全参数更新。
- **生成**：从提示词开始逐字符采样（温度缩放 + 累积分布），每步把最后 `maxLen` 个字符作为上下文重新前向。`top_p < 1` 时先按概率从高到低保留累计概率刚好达到 `top_p` 的 token 集合，再在集合内重新归一化采样（nucleus 采样）；`top_p = 1` 完全跳过这一步，采样结果与旧版逐位相同。
- **采样控制**：生成时还可叠加五项与 Python 版一致的策略，默认全部关闭。每一步的顺序是「重复惩罚 → 存在惩罚 → 频率惩罚 → 禁重复 n-gram → 温度缩放 → top_k / top_p → min_p」：
  - `min_p`：top_k / top_p 过滤后，丢掉概率低于 `min_p × 最大概率` 的 token（`0` 关闭）。例如 `min_p = 0.5` 在 `0.7 / 0.2 / 0.1` 上只留 `0.7`；两个等概率 token 会一起保留，因为是按相对峰值截尾。
  - `typical`（`mass`，默认 0.9）：按「惊讶度 `|−log p − H|` 从小到大」累计概率，直到 ≥ `mass`；保留的是最「典型」的那批 token，而不是最简单的一批。
  - `top_a`（默认 0.1）：阈值 `top_a · p_max²`，比它小的直接砍掉；`0` 关闭。
  - `eta`（默认 0.3）：丢弃概率 ≤ `eta` 的 token（至少保留 argmax），比 `min_p` 更硬。
  - `tft`（tail-free，默认 0.5）：按 log 概率的二阶差分找长尾拐点，`tft` 越小保留越多。
  - `mirostat_mu`：Mirostat 掩码（保留 `−log p ≤ mu`，并给出保留集熵），配合 `llmMirostatUpdate(mu, observed, target, strength)` 迭代。

过滤器顺序写死为 **top_k → typical → top_a → eta → tail_free → top_p → min_p → mirostat**；全部留空时不调用任何过滤器，采样路径与旧版逐位相同（`tools/llm_core_test.js` 有断言）。各过滤器的保留集合与 Python `AI.typical_filter` 等用同一行 logits 的实测结果对齐（`[0.5, 2.0, -1.0, 1.0, 3.0, 0.0, -0.5, 1.5]` → typical `{1,3,4,7}`、top_a `{0,1,3,4,7}`、eta `{4}`、tail_free `{1,4,7}`、mirostat `{1,4}` 且熵 `0.582203108888218`）。

### 前缀 KV 缓存（数据结构 + 可视化）

对应 Python 的 `AI.PrefixCache`，是**纯数据结构**（每层两张 `(len, d)` 行表）：

| 函数 | 作用 |
|---|---|
| `llmPrefixCache(numLayers, dModel)` | 建空缓存 |
| `llmCacheSet(c, layer, keys, values)` / `llmCacheGet(c, layer)` | 写入 / 读取（读出来是深拷贝） |
| `llmCacheLength(c, layer)` | 该层已缓存的行数 |
| `llmCacheTruncate(c, n)` | 所有层截断到 `n` 行（只截不放） |
| `llmCacheMerge(c, other)` | 追加另一个同形状缓存的全部层 |
| `llmCacheToJSON(c)` / `llmCacheFromJSON(data)` | JSON 往返（可直接塞进存档） |
| `llmCacheStats(c, promptLen)` / `llmCacheBars(stats, width)` | 每层 `{layer, len, reuse}` 与 ASCII 条形图 |
| `llmForwardStep(m, tokenId, pos, c)` | 单 token 增量前向（复用同一份缓存结构） |
| `llmGenerateCached(m, ids, n, temperature, topK, rand, topP, opts)` | 预填充 + 增量生成，返回 `{ids, stats: {prefill, steps, rebuilds}}` |

行宽、有限数字、层号范围、合并时层数/维度一致、`truncate` 非负整数、`fromJSON` 的字段与
层数都做校验——12 条非法输入在 `tools/llm_core_test.js` 里逐条断言会抛错。

缓存现在真的接进了生成循环：`llmForwardStep` 做单 token 增量前向，`llmGenerateCached` 先整段
预填充、之后每步只前向一次；窗口（`maxLen`）滑动导致位置整体左移时旧缓存作废，会按新窗口重建
一次（源码里记 `rebuilds`）。面板上的「用 KV 缓存」勾选走的就是这条路，统计行会显示
`prefill / 步进 / 重建` 次数。

数值上与整段重算**逐位一致**（`tools/llm_core_test.js` 有 `window=0` / `window=2` 的单步对比、
cached 与普通生成的逐 token 对比，含滑动重建场景）；速度上仍是教学实现，不做基准承诺。
Python 与 C 版本也都有真正的 KV cache（C 侧实测 20 倍）。

### 词表级 JSON 约束解码

页面同时给出一个**词表级**（token-level）的 JSON 约束解码器，等价于 Python 的
`AI.grammar_allowed(AI.json_grammar(), prefix, vocab)`：

| 函数 | 作用 |
|---|---|
| `llmJsonStart()` | 初始状态（期待一个顶层值） |
| `llmJsonStep(ctx, token)` | 吃一个 token，返回新状态；非法则返回 `null` |
| `llmJsonAllowed(ctx, vocab)` / `llmJsonMask(ctx, vocab)` | 当前允许的 token 列表 / 布尔掩码 |
| `llmJsonPick(logits, vocab, ctx)` | 在允许集里取 logit 最大者（同分取小下标），无解则抛错 |
| `llmJsonComplete(ctx)` | 文档是否已完整（`{"a":[1,true]}` 走完即 `true`） |

state 是一个 `{ st, stack }` 对象，`stack` 记录当前嵌套的容器（`"obj"` / `"arr"`），
所以 `st` 只有 `VAL / KCLOSE / KEY / COLON / CEND / AEND / VCLOSE / DONE` 八种。
它是**增量**的：每来一个 token 只做一步，不需要重跑整段前缀。

`tools/llm_core_test.js` 里有 15 个前缀的黄金值（同一个 15 词的词表，对照 Python
`grammar_allowed` 的允许集合逐项一致）、完整文档走到 `DONE`、`DONE` 后再喂 token 被拒、
`llmJsonPick` 的三条错误路径，以及「随机 logits 走 300 次从不产生非法 token」的性质抽查。

边界：这是**词表级**约束——词表里的每个元素被当成一个完整 token（`"a"`、`-2.5`、`true`、
`{` 等）。页面本身的 LLM 是**字符级**生成的，所以另外实现了按字符推进的语法器与约束解码（下一节），
这一节的词表级解码器保留作对照。

### 对比搜索（contrastive search）

对应 Python 的 `AI.contrastive_select` / `AI.contrastive_search`：每步在概率 top-k 候选里用
「候选自己的隐状态」与历史隐状态的最大余弦相似度做退化惩罚，`score = (1−α)·p − α·reward`，
再取分数最高者。惩罚必须按候选各自的隐状态算——共用一个隐状态时所有候选同罚，选择会退化成
argmax（`tools/llm_core_test.js` 专门断言这点）。

| 函数 | 作用 |
|---|---|
| `llmContrastiveSelect(logits, hiddens, history, alpha, topK)` | 单步选择，返回 `{token, score, prob, reward, candidates}` |
| `llmContrastiveSearch(m, ids, maxNewTokens, topK, alpha)` | 完整生成，返回 `{ids, picks, forwards}` |

代价是每生成一个字符要做 `top_k` 次候选前向（比普通续写慢一个数量级），面板上「对比搜索」按钮
与 `top_k` / `alpha` 两个输入会把这个代价显式写进统计行；提示 + 续写总长超过 `maxLen` 会直接报错
（对比搜索不做窗口截断）。

### 字符级语法约束解码（正则 / CFG / 字符级 JSON）

对应 Python 的 `RegexGrammar` / `CFGGrammar` / `json_grammar()`，但按**字符**推进、接进了生成
循环：每一步把语法不允许的字符 logit 置为 `-Infinity`，再走与普通生成相同的惩罚 / top_k /
top_p / min_p 采样路径。

| 函数 | 作用 |
|---|---|
| `llmRegexGrammar(pattern)` | 自写正则解析（字面量、`.`、`*+?`、`[abc]`、`[^abc]`、`\d`/`\w`/`\s`、`\|`、分组）+ Thompson NFA，整段匹配 |
| `llmCfgGrammar(productions, start)` | Earley 解析器；右部可混用字符串终结符与谓词函数，支持空产生式 |
| `llmJsonCharGrammar()` | 字符级完整 JSON 文法（对象 / 数组 / 字符串含转义 / 数字含整数小数指数 / true / false / null，紧凑形式） |
| `llmGrammarAllowed` / `llmGrammarAccepts` / `llmGrammarMask` / `llmGrammarPick` | 正则 / CFG 通用入口 |
| `llmGrammarGenerate(...)` / `llmConstrainedGenerate(...)` | 贪心 / 采样两种约束生成 |

字符类按 ASCII 语义：这里 `\d` 只认 `0-9`、`\w` 只认 `[A-Za-z0-9]`（Python 的 `_w7e_char_predicate`
用 `isdigit`/`isalnum`，更宽）。面板的「约束解码」输入框填 `json` 或一个正则，生成结果只含文法
允许的字符。

边界：**不包含 JSON Schema 校验**——与 Python 版边界一致（`README.md:2701` 明说「那需要
JSON Schema 层，本库未实现」）；键去重、数字范围等语义仍由 `JSON.parse` 收尾校验。
  - `repetition_penalty`：已出现 token 的 logit 若为正则除以它，否则乘以它（`1` 关闭）。例如惩罚 `2` 会把 `2 → 1`、`-1 → -2`。
  - `presence_penalty`：每个出现过的 token 的 logit 减一次，不管出现几次（`0` 关闭）。
  - `frequency_penalty`：按出现次数扣，`logit -= frequency_penalty × 次数`（`0` 关闭）。
  - `no_repeat_ngram_size = n`：最近 `n-1` 个字符的片段若在历史里出现过，就把相同的 `n`-gram 后继置为 `-Infinity`（`0` 关闭，`1` 没有意义因而非法）。
  五项留空或取默认值时，采样结果与未引入这些控制前逐位一致；`min_p ∉ [0,1)`、`repetition_penalty ≤ 0`、`no_repeat_ngram_size = 1` 或非数值输入会在面板内报错。
- **困惑度**：训练结束后用更新后的权重重新前向一遍，`困惑度 = exp(平均交叉熵)`；平均交叉熵超过 700 时显示 `inf`，避免浮点溢出。
- **存档**：模型可导出为 JSON（`format` + 配置 + 字符表 + 全部参数），重新加载时会校验格式、配置兼容性与每个参数数组的形状 / 数值，失败时在面板内给出中文错误而不是抛异常；读回后生成结果与保存前一致。
- 默认语料 `012012…` 是刻意选的周期串：几十轮就能把 loss 压到 1e-5，肉眼确认「它确实学会了转移规律」；换成一句中文也能训练，只是需要的轮数与维度更大。

- **输入校验**：token id 必须是 `[0, 词表大小)` 内的整数，`temperature` 必须 `> 0`，`top_p` 限定在 `(0, 1]`，`min_p` 必须在 `[0, 1)`，`repetition_penalty` 必须是有限正数，`presence_penalty` / `frequency_penalty` 必须是有限数，`no_repeat_ngram_size` 必须是 `0` 或 `≥ 2` 的整数；前向算完 logits 会检查是否全部有限——权重被 NaN 污染时直接报错，而不是把 NaN loss 当正常值继续训练（那会让 Adam 把整个模型写坏）。整行被掩码掉时 `softmax` 返回全 `0` 而不是 `NaN`。

这部分实现与仓库另外两个版本（Python 版 `CausalLM`、C 版 `llm.c`）是同一套结构与同一套掩码语义。

## 技术实现

| 模块 | 说明 |
|------|------|
| 神经网络 | 纯 JavaScript 手写，前向传播 + 反向传播 |
| 架构模块 | 手写 CNN 1D / RNN / Multi-Head Self-Attention |
| 语言模型 | 手写字符级因果 Transformer + Adam（前向、反向、采样全部在页面内） |
| 生成控制 | 对比搜索、正则 / CFG / 字符级 JSON 约束解码、KV 缓存增量前向（与整段重算逐位一致） |
| 采样 | 温度缩放 + top-k + nucleus（top_p）+ min_p，重复 / 存在 / 频率惩罚与禁重复 n-gram，语义与 Python 参考实现一致 |
| 模型存档 | 参数 / 配置 / 词表序列化为 JSON，导入时校验 format、形状与数值 |
| 优化器 | 手写 Adam / Momentum / RMSProp / SGD |
| 图表 | Chart.js 4（CDN） |
| 样式 | 纯 CSS |

所有运算均在浏览器本地完成，数据不会上传到任何服务器。

## 本地校验

仓库自带的 Node 校验脚本只用于开发回归，**不参与页面运行**：`index.html` 仍然是零依赖单文件，直接双击打开即可，`tools/` 不会被页面加载。需要 Node 18+。

一条命令跑完全部检查：

```bash
node tools/check.js
```

全部通过时最后一行固定为 `ALL PASS (N checks)`，退出码 0；任何一项失败都会打印 `FAIL` 行并以非 0 退出。也可以单独运行某一套：

```bash
node tools/llm_core_test.js   # 只跑 LLM 核心数值
node tools/llm_page_check.js  # 只跑页面结构与伪 DOM
```

覆盖范围：

- **LLM 核心数值**（`tools/llm_core_test.js`）：`tools/extract_page_js.js` 每次运行都从 `index.html` 现场抽出内联的 LLM 核心函数再执行，测的就是页面里的实现，不存在第二份拷贝。
  - 与 `tools/fixtures/llm_baseline.json` 逐位对比 20 步训练 loss、前向 loss / logits、三种生成序列（旧 6 参数调用、显式 `top_p=1`、`window=0`）；
  - nucleus 采样：`top_p=0.5` 只保留最高概率 token、`0.8` 保留前两名丢掉尾部、`top_p=1` 与旧采样在 100 组随机 logits 上逐位一致；
  - 采样控制：五项惩罚 / `min_p` 各有多条定向断言（repetition 正除负乘、presence 只扣一次、frequency 按次数、ngram=2/3 禁后继、min_p 只留峰值且与 top_k 叠加），显式全中性参数与黄金基线逐位一致，8 种非法参数被拒绝；
  - 滑动窗口：`window=2` 的因果性（改末位不影响前文）、`window=1` 只看自己、非法窗口被拒绝；
  - 中心有限差分梯度：`window=0` / `window=2` 各取 10 个参数，最差相对误差 ≤ 1e-8；
  - 困惑度 = `exp(平均交叉熵)`，`loss > 700` 给 `inf`；
  - 存档往返：JSON 写读后参数 / 配置 / 词表逐位一致、生成一致；7 种非法存档都被拒绝并带中文错误；
  - 对比搜索：候选打分公式、`alpha` 两端、`top_k` 约束、reward 取自候选自身隐状态、零向量不除零、
    `top_k=1` 与整段前向复核的 argmax 逐步一致、前向次数与错误路径；
  - 正则 / CFG / 字符级 JSON 文法：整段匹配、允许集与掩码、谓词终结符、空产生式、JSON 转义与数字
    分段、非法串（尾逗号 / 裸键 / 前导零 / 空白）、贪心与采样约束生成、错误路径；
  - 增量前向：`llmForwardStep` 与整段 `llmForward` 在 `window=0` / `window=2` 下逐位一致；
    `llmGenerateCached` 与 `llmGenerate` 逐 token 一致（含滑动重建），并断言 prefill / steps / rebuilds。
- **页面结构与 DOM 契约**（`tools/llm_page_check.js`）：
  - 两段内联 `<script>` 分别编译通过；
  - `div/span/button/textarea/table/select/option` 标签配平；
  - 所有 `id` 唯一，`$id(...)` / `document.getElementById` 引用的元素都存在；
  - 在伪 DOM 中真正执行页面的 LLM 面板 IIFE，跑通「训练 → 自动生成 → `top_p=0.5` 再生成 → 五项控制留空 / 显式中性生成一致 → 惩罚 + `ngram=2` 生成 → 勾选 KV 缓存生成 → 对比搜索生成 → 语法约束解码生成 → 保存为 Blob → 非法 JSON 报错 → 加载存档 → 继续生成」，并验证窗口输入 `1.5` 报错、`min_p=1.5` / `ngram=1` / 非数值惩罚在面板内报错、对比 `top_k=0` / `alpha=1.5` 与非法约束文法在面板内报错、缺少 `#llmWindow` 时退回全注意力。

`tools/fixtures/llm_baseline.json` 是引入 nucleus 采样 / 滑动窗口之前的旧实现黄金基线。除非**有意**修改数值算法，否则不要重新生成它；如果确实要改，请在提交信息里写明原因。

## 文件结构

```
├── index.html                   # 全部页面代码与 UI（单文件，运行时零依赖）
├── README.md
└── tools/                       # 开发期本地校验，不参与页面运行
    ├── check.js                 # 入口：node tools/check.js
    ├── extract_page_js.js       # 从 index.html 抽取内联脚本 / LLM 核心，提供伪 DOM
    ├── report.js                # PASS/FAIL 计数与汇总
    ├── llm_core_test.js         # LLM 核心数值检查
    ├── llm_page_check.js        # 页面结构与伪 DOM 集成检查
    └── fixtures/
        └── llm_baseline.json    # 旧实现黄金基线（逐位对比用）
```
