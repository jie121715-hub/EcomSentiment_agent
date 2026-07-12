# 🤖 电商智能客服 Agent 系统 — 面试准备

> Multi-Agent · 三层检索漏斗 · 父子块RAG · 安全门控 · BERT+LLM混合
> 面试稿 v5.0 | 基于 PDF《怎样组织项目文档》框架重构

---

## 一、开场自我介绍（2分钟）

面试官你好，我最近独立完成了一个**电商智能客服 Agent 系统**。

核心解决的问题：**传统客服机器人延迟高、回答不准、安全兜底弱**。

我的方案是构建了一个多 Agent 协作系统，核心亮点三个：

**第一，Agent 协作架构。** 用 LangGraph 编排 5 个 Agent——Perception 负责感知（BERT 本地做情感分析和意图识别）、Router 做三维决策（置信度不够反问澄清、愤怒转人工、4 意图 1:1 分发）、KnowledgeQA 管知识问答、Business 管订单物流等业务操作、KnowledgeMgmt 管知识录入。解耦后每个 Agent 可独立升级和测试。

**第二，三层检索漏斗把延迟从 3 秒降到 400 毫秒。** 80% 是高频重复问题，我加了两层快通道：Redis 精确缓存命中 30%（<5ms），BM25+MySQL FAQ 关键词匹配命中 50%（~15ms），只有 20% 复杂问题才走完整 RAG 管线。加权平均 408ms，提升 7.4 倍。

**第三，安全体系。** 高危门控防止 LLM 瞎编政策承诺；上传侧 API Key 鉴权 + 18 种恶意模式扫描，三级分流（拒绝/待审核/通过）；父子块架构保证检索精度和上下文完整性。

技术栈：FastAPI + LangGraph + BGE-M3 + Milvus + MySQL + Redis + DeepSeek + PyTorch/BERT。

---

## 二、项目一页纸（速查）

| 维度 | 内容 |
|------|------|
| **项目名称** | 电商智能客服与业务处理 Agent 系统 |
| **核心指标** | 5 Agent · 4 意图 · 7 情感标签 · 平均延迟 408ms · BERT F1 89.5% |
| **技术栈** | FastAPI · LangGraph · BGE-M3 · Milvus · MySQL · Redis · BERT · DeepSeek |
| **数据规模** | 189 条种子数据 + 9 张 MySQL 表 + Milvus 向量库 |
| **个人角色** | 独立全栈（架构设计 + Agent 开发 + RAG 管线 + 安全体系 + 前端调试面板） |

---

## 三、需求调研 & 方案选型

### 3.1 为什么做这个项目？

- 传统客服机器人基于关键词匹配，无法理解"我的快递到哪了"这类自然语言变体
- 单一 LLM 方案延迟高（每次 3s+），且无法执行精确事务（查数据库、调快递鸟 API）
- LLM 存在幻觉风险——用户问"假一赔几"，LLM 可能编造不存在的政策承诺

### 3.2 为什么选多 Agent 架构而不是单一 LLM？

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **单一 LLM + 提示词** | 实现简单 | 无法精确读写DB/调API，幻觉不可控 | ❌ |
| **多工作流串联（LangChain 旧版）** | 逻辑清晰 | 不支持动态分支/并行/循环，上下文失控 | ❌ |
| **多子 Agent（LangGraph）** | 复杂度可控、上下文隔离、方便协作和升级 | 代码量大，调试困难 | ✅ 最终选择 |
| **统一 MCP 服务** | 实现快 | 上下文过载，工具调用错误率高 | ❌ |
| **按业务切分 MCP** | 职责清晰 | 需深入理解业务切分边界 | 可作为补充方案 |

核心决策逻辑：**多 Agent 方案让每个 Agent 拥有独立上下文和可测试性**。Perception 管感知（BERT 本地推理不耗 LLM 调用）、Router 管决策（集中路由逻辑）、KnowledgeQA 管语义（独立优化缓存）、Business 管精确事务、KnowledgeMgmt 管知识录入（权限隔离）。

### 3.3 模型选型

**LLM 选型过程（类比 PDF 讲的基线→候选→评估方法）：**

| 步骤 | 内容 |
|------|------|
| **基线模型** | DeepSeek-V4-Pro——中文能力强，之前项目验证过效果 |
| **候选模型** | Kimi-K2.6 / Qwen3.7-Max / DeepSeek-V4-Pro |
| **评估方法** | 业务场景测试集（FAQ回答/意图识别/实体抽取），精确率+召回率+F1 |
| **最终选择** | DeepSeek-V4-Flash——效果接近 Pro，但价格便宜很多，综合性价比最优 |

**嵌入模型选型：**

| 步骤 | 内容 |
|------|------|
| **基线模型** | BGE-M3——中文检索 SOTA，稠密+稀疏+多向量三合一 |
| **候选模型** | text2vec-large-chinese / m3e-base / BGE-M3 |
| **评估方法** | 23 条 FAQ 检索命中率对比 |
| **最终选择** | BGE-M3——1024维，命中率最优 |

**向量库选型：**

| 候选 | 定位 | GPU | 标量字段 | 选择理由 |
|------|------|-----|---------|---------|
| Chroma | 原型开发 | ❌ | ✅ | 不适合生产 |
| Milvus | 大规模向量检索 | ✅ | ✅ | ✅ 分布式 + 高性能 + 元数据过滤（parent_id/chunk_type） |
| FAISS | 算法研究 | ✅ | ❌ | 无标量字段，无法做父子块过滤 |

---

## 四、数据组织

### 4.1 数据来源

| 数据类型 | 来源 | 规模 | 用途 |
|---------|------|------|------|
| 商品数据 | 模拟京东商品信息 | 71 条 | 商品咨询问答 |
| 订单数据 | 模拟订单记录（7 种状态） | 30 条 | 业务操作 Agent |
| 知识库 | 自建电商政策文档 | 65 条 | RAG 检索 |
| FAQ | 人工整理高频问答 | 23 条 | BM25 FAQ 层 |
| 情感训练数据 | 京东真实评论标注 | ~2000 条 | BERT 情感分类训练 |
| 意图训练数据 | 人工标注 4 类意图 | JSON 格式 | BERT 意图分类训练 |

### 4.2 数据库设计（9 张表）

| 表名 | 用途 | 设计要点 |
|------|------|---------|
| `products` | 商品缓存 | 减少对淘宝 API 的调用 |
| `orders` | 订单记录 | 7 状态枚举（pending→paid→shipped→delivered→cancelled→refunding→refunded） |
| `custom_knowledge` | 知识库原文 | source 字段追踪来源（merchant:{id} / upload:{filename}） |
| `ecom_faq` | FAQ 问答 | 5 大类别（产品咨询/订单问题/售后服务/促销活动/物流配送） |
| `shops` | 店铺信息 | 支持多租户扩展 |
| `conversation_history` | 对话持久化 | 上下文感知 + 确认操作 |
| `user_profile` | 用户画像 | 个性化回复 |
| `clarify_logs` | 反问日志 | 闭环数据→反馈训练 BERT 分类器 |
| `support_tickets` | 转人工工单 | TK-YYYYMMDD-XXXX 格式，完整审计追踪 |

---

## 五、核心亮点详解

### ⚡ 亮点1：三层检索漏斗（3s → 408ms）

**问题：** 原始 RAG 每条消息都走完整管线（Query 改写 4 次 LLM + 混合检索 + Reranker + LLM 生成），平均 3000ms。

**方案：** 发现客服场景 80% 是高频重复问题。在 RAG 前加两层快速通道：

```
用户问题
  │
  ├─ 第1层: Redis 精确缓存 (answer:{query})
  │   延迟 <5ms，命中率 ~30%
  │   命中 → 直接返回（不走LLM）
  │
  ├─ 第2层: BM25 + MySQL FAQ 关键词匹配
  │   延迟 ~15ms，命中率 ~50%
  │   jieba 分词 + BM25Okapi + 双条件门控（raw_best≥1.8 且 raw_avg≥2.0）
  │   命中 → 返回预设答案（不走LLM）
  │
  └─ 第3层: 完整 RAG（仅20%流量触发）
      延迟 ~2000ms
      Query改写 → Milvus子块检索 → parent_id去重 → BGE-Reranker精排 → LLM生成 → 异步写回Redis
```

**效果：** `0.30×5ms + 0.50×15ms + 0.20×2000ms ≈ 408ms`，加权平均提升 7.4 倍。

**BM25 双条件门控怎么调的？** 跑了一批 FAQ 查询的 BM25 分数分布，找到能区分"真命中"和"假命中"的阈值分界线—— `raw_best≥1.8`（最佳匹配分数）+ `raw_avg≥2.0`（Top-5 平均分）同时满足才判定命中。太低误匹配，太高漏召回。

**边界情况处理：** FAQ 层加了 `_is_product_inquiry()` 门控——"都有什么产品"被 FAQ 拦截时，检测到商品咨询意图自动跳过 FAQ 走 LLM 通识兜底。

---

### 🧠 亮点2：多 Agent 协作 + LangGraph 编排

**5 个 Agent 职责：**

| Agent | 职责 | 模型策略 |
|-------|------|---------|
| **PerceptionAgent** | BERT 情感(7类) + BERT 意图(4类) + LLM 实体抽取 | 本地 BERT（省调用）+ LLM sentiment_llm（thinking=true 做细粒度分类） |
| **RouterAgent** | 三维决策：置信度门控 / 情绪紧急度 / 意图分发 | LLM routing（thinking=false，轻量路由） |
| **KnowledgeQAAgent** | 三层检索（80%流量入口） | LLM qa（thinking=false，流畅生成） |
| **BusinessAgent** | 订单查询/取消/退款/物流查轨迹 | MySQL + 快递鸟 API（不消耗 LLM） |
| **KnowledgeMgmtAgent** | 知识录入 + 文件上传安全 | 文本提取 + 安全扫描 + 父子块双写 |

**为什么不同 Agent 用不同 LLM 配置？**
- 分类任务（意图/情感）→ `thinking=true`，用推理模式做精确判断
- 生成任务（问答/推荐）→ `thinking=false`，追求流畅输出
- 每个 Agent 的 LLM 调用用复合缓存键隔离：`{model}_{temperature}_{streaming}_{thinking}`，避免 PyTorch 图缓存冲突

**LangGraph 编排细节：**

```
Perception → Router → 条件边路由
                         ├─ knowledge_qa → KnowledgeQA
                         ├─ business → Business
                         ├─ knowledge_mgmt → KnowledgeMgmt
                         └─ escalate → Escalate
```

**并行节点状态合并（PDF 强调的面试难点）：**

如果有并行执行的 Agent 节点，LangGraph 用 `Annotated[type, reducer_function]` 解决状态冲突：

```python
# State 中字段的合并策略
class AgentState(TypedDict):
    sentiment: str                              # 单节点写入，无需 reducer
    intent: str                                 # 单节点写入
    messages: Annotated[list, operator.add]      # 多节点并行 → 追加合并
    confidence: Annotated[float, max]            # 多节点并行 → 取最大值
```

**关键设计决策：**
- Router 三维决策优先级：**紧急接管 > 反问澄清 > 正常分发**
- 反问有智能跳过：订单号/快递单号/确认回复场景自动跳过（从 context 判断）
- `_correct_product_inquiry_intent`：BERT 误判修正——"查下苹果18参数"含"查"字易误判 BUSINESS，检查商品词+无业务词→强制 KnowledgeQA

---

### 🛡️ 亮点3：安全体系

**问答侧——高危门控：**

| 问题类型 | 知识库有文档 | 知识库无文档 |
|----------|-------------|-------------|
| 商品参数/通识 | RAG + LLM 正常回答 | LLM 通识兜底 + 自然语言免责声明 |
| 退款政策/法律/承诺类 | 基于文档精确回答 | 🚨 强制转人工，绝不用 LLM 编造 |

检测逻辑：30+ 高危关键词（退款、赔偿、假一赔几、法律条款…）+ RAG 检索结果相似度 < 阈值 → 触发门控。

**上传侧——三道防线：**

| 防线 | 机制 |
|------|------|
| ① API Key 鉴权 | `X-API-Key` → `merchant`(可上传) / `admin`(全部) / 无 Key(403) |
| ② 内容安全扫描 | 18 种恶意模式 + 白名单（7天无理由/运费险等合法术语） |
| ③ 三级分流 | 🔴 严重(假一赔十/永久保修) → 422拒绝 · 🟡 可疑(品牌直营) → pending_review · 🟢 正常 → 入库 |

**为什么需要白名单？** "7天无理由退货""运费险"是合法电商术语，不加白名单会被误杀，影响正常商户上传。

---

### 🧬 亮点4：父子块 RAG 架构

**传统切块的问题：** 固定 500 字 → 上下文碎片 → LLM 拿到不完整段落 → 回答质量下降。

**ParentChildChunker 方案：**

```
原始知识（MySQL custom_knowledge）
  │
  ▼
ParentChildChunker 切分（上传和RAG共用）
  ├─ 父块: 1000 字 rolling window → 给 LLM 看（完整上下文）
  └─ 子块: 300 字 sliding window → 给向量检索用（精准匹配）
  │
  ▼
写入 Milvus（同一 Collection，chunk_type 区分）
  - parent: chunk_type="parent", parent_id=null
  - child: chunk_type="child", parent_id=<parent_id>
  │
  ▼
检索时：
  Query → Embedding → Milvus 搜子块 → parent_id 去重 → 取父块1000字 → LLM
```

**4 层父块回退策略**（保证新旧数据兼容）：
1. 内存 `_parent_map` 字典（最快，同进程写入后可用）
2. Milvus 按 `parent_id` 查询（Milvus 可用时）
3. 检查搜索结果是否已含父块（已有数据）
4. 降级直接用子块（无父块元数据时）

---

### 🔧 亮点5：工程化细节

| 维度 | 做法 |
|------|------|
| **预加载防崩溃** | Windows DLL 竞态问题 → 单线程预加载 pyarrow/pandas/sklearn/torch |
| **优雅启动** | lifespan 预热：MySQL → 5个Agent → BM25索引，每步独立 try/except 降级 |
| **异步全链路** | FastAPI async/await + asyncmy 异步MySQL + asyncio.create_task 异步写缓存 |
| **重试机制** | 三层：指数退避重试(1s→2s→4s) → Agent降级返回 → 系统兜底(客服电话) |
| **SSE 生产适配** | `X-Accel-Buffering: no` 头，防止 Nginx 缓冲破坏实时性 |
| **模型降级** | DeepSeek 主用 + DashScope(阿里云) 备用——多提供商防止单点故障 |
| **网络适配** | `trust_env=False` 跳过系统代理直连国内 LLM API |
| **配置管理** | Pydantic Settings + .env.local 环境隔离，所有 RAG 参数可配置开关 |
| **全链路调试** | `/debug` 三栏式可视化面板 + SSE 结构化事件(perception→route→token→done) |

---

## 六、面试高频问答

### Q1: 三层检索怎么把延迟降到 400ms 的？
> 客服场景 80% 是高频重复问题。我在 RAG 前面加了 Redis 精确缓存（30%命中，<5ms）和 BM25+MySQL FAQ 关键词匹配（50%命中，~15ms），只有 20% 复杂问题走完整 RAG。加权平均 408ms。FAQ 命中率是关键——我用 jieba 中文分词 + BM25 + 双条件门控阈值调优，23 条 FAQ 已经能正确区分命中/未命中。

### Q2: 为什么用多 Agent 而不是一个 LLM？
> 单一 LLM 无法精确读写数据库、调快递鸟 API、做安全门控。PDF 里讲：多子 Agent 方案复杂度可控、上下文隔离、方便协作和升级。我拆成 5 个 Agent：Perception 管感知（本地 BERT 省 LLM 调用）、Router 管决策、KnowledgeQA 管语义（独立优化缓存）、Business 管精确事务、KnowledgeMgmt 管知识录入（权限隔离）。各自独立升级测试。

### Q3: LangGraph 并行节点状态合并怎么处理？
> PDF 专门提到这是面试难点。核心是用 Python 的 `Annotated[type, reducer]`：比如 messages 字段用 `operator.add` 做追加合并，confidence 字段用 `max` 取最高置信度。每个并行节点写入同名字段时，LangGraph 自动调用对应的 reducer 函数合并，不会出现后写入覆盖先写入的问题。

### Q4: 父子块检索怎么做的？
> 传统等分切块上下文碎片化。我用 ParentChildChunker 切成父块 1000 字（给 LLM）和子块 300 字（向量检索）。子块检索精准，parent_id 去重后取父块完整上下文。而且有 4 层父块回退——内存缓存→Milvus 查询→结果自带→降级用子块，保证新旧数据兼容。

### Q5: 怎么防止 LLM 捏造答案？
> 高危门控：检测退款/法律关键词 → RAG 检索相似度 < 阈值 → 强制转人工。PDF 里讲的合规性检查：规则过滤 + 大模型过滤。我的实现是：上传侧 18 种模式扫描 + 问答侧高危门控 + 相似度门控，三道防线。

### Q6: 大模型生成的工具参数格式错误怎么办？
> PDF 提到三种兜底：① 参数不合法→把错误信息+原始 tool_call 还给大模型让它修正；② MCP 超时→指数退避重试（1s→2s→4s）；③ 返回空结果→让大模型宽泛条件重试。我的实现：3 次重试 + Agent 降级返回友好提示 + 系统兜底返回客服电话。

### Q7: BERT 意图识别不准怎么办？
> 双层保障：① Router 置信度门控（<50% 反问澄清，但订单号/确认回复场景自动跳过）；② 商品咨询智能修正——"查下苹果18参数"BERT 可能因"查"字判 BUSINESS，加 `_correct_product_inquiry_intent` 二次修正。

### Q8: 你的模型选型过程是怎样的？
> PDF 强调要有基线→候选→评估的完整链路。我的 LLM：基线 DeepSeek-V4-Pro（之前项目验证过）→ 候选 Kimi-K2.6/Qwen3.7-Max → 评估后 DeepSeek 最优 → 综合性价比选 flash 版。嵌入模型：基线 BGE-M3 → 候选 text2vec/m3e → 用 23 条 FAQ 测检索命中率 → BGE-M3 最优。

### Q9: 如果让你改进，你会做什么？
> ① 降本：不同难度问题灵活切换轻重模型（PDF：平时全量回答，峰值压缩版回答）；② 压峰：高峰时段先给精简答案，用户追问再展开（实测追问率 27.3%）；③ 数据迭代：积累 clarify_logs 数据反馈训练 BERT 分类器；④ FAQ 自动发现：从对话日志聚类高频问法自动生成 FAQ；⑤ 引入 Langfuse/LangSmith 做全链路监控。

### Q10: 项目有什么产出？
> **技术效益**：延迟 3s→408ms（7.4x），BERT F1 89.5%，QPS 提升显著；**运营效益**：三层漏斗使 80% 问题免走 LLM，GPU 消耗大幅下降；**知识沉淀**：9 张 MySQL 表 + Milvus 向量库 + 父子块检索架构可复用到其他领域。

---

## 七、STAR 故事库

### 故事1：延迟优化
> 初版每条消息都走完整 RAG 管线——Query 改写调 4 次 LLM + 混合检索 + Reranker + LLM 生成——平均 3 秒。我分析流量发现客服场景 80% 是"怎么退货""多久发货"这类高频问题。方案是在 RAG 前加 Redis 缓存 + BM25 FAQ 匹配，只有 20% 长尾走完整 RAG。FAQ 层的双条件门控阈值（raw_best≥1.8 + raw_avg≥2.0）是基于 23 条 FAQ 反复调参确定的。最终 3s → 408ms，提升 7.4 倍。

### 故事2：高危门控
> 测试时发现 LLM 会编造政策——用户问"假一赔几"，LLM 直接答"假一赔十"，但知识库没这条政策。我在 RAG 后处理加了高危门控：检测到退款/赔偿关键词 + 检索相似度 < 阈值 → 强制转人工。上传侧也加了 18 种恶意模式 + 白名单扫描（防止"7天无理由"被误杀）。三道防线保证答案可信。

### 故事3：父子块检索
> 最初用 RecursiveCharacterTextSplitter 等分 500 字，LLM 经常拿到半句话。调研后实现 ParentChildChunker：父块 1000 字（LLM 上下文），子块 300 字（向量检索）。子块匹配精准、父块上下文完整。关键是用 4 层回退保证新旧数据兼容——从内存缓存到降级子块，逐层兜底。

---

## 八、技术决策速查（"为什么用 X 而不是 Y"）

| 问题 | 我的选择 | 理由 |
|------|---------|------|
| Agent 编排 | LangGraph（非旧版 LangChain） | 旧版只支持 DAG，LangGraph 支持条件边+循环+并行+动态路由 |
| 向量库 | Milvus | 分布式+高性能+元数据过滤（parent_id/chunk_type），Chroma 只适合原型 |
| 嵌入模型 | BGE-M3 | 稠密+稀疏+多向量三合一，中文 SOTA，23条FAQ测试命中率最优 |
| Reranker | BGE-Reranker | Cross-encoder 精排，失败自动降级按原始分数排序 |
| FAQ 检索 | BM25 | 关键词匹配比向量检索更精准（"退货"vs"退款"不会被 embedding 模糊） |
| 缓存 | Redis | TTL 自动过期 + 毫秒响应 + 两级键空间（rag 答案+FAQ精确匹配） |
| 本地 NLP | BERT | 省 LLM 调用 + 低延迟（~100ms），失败降级 LLM 零样本 |
| LLM | DeepSeek-V4-Flash | 性价比最高，分类任务开 thinking 模式，生成任务关 thinking |
| API 框架 | FastAPI | async/await 原生 + Pydantic 校验 + SSE 流式 |
| 混合检索 | 4路并行 + RRF 融合 | 稠密+稀疏+混合+多向量 → asyncio.gather 并行 → RRF k=60 融合 |

---

## 九、上线方案（规划）

| 维度 | 方案 |
|------|------|
| **部署方式** | FastAPI + Uvicorn，Nginx 反向代理，Docker 容器化 |
| **冗余** | 绝对不单点部署——至少双备份，重要服务三备份 |
| **弹性伸缩** | 常驻 1 实例，压力 > 阈值 5 分钟自动拉起新实例，上限 10 个；高峰时段无条件预留 5 实例 |
| **备份** | 3-2-1 原则：3 份副本 + 2 种介质 + 至少 1 份异地。日志和 DB 每日归档 |
| **监控** | 结构化日志（trace_id 全链路追踪），计划接入 Langfuse 全链路监控 |
| **Nginx 适配** | SSE 流式设置 `X-Accel-Buffering: no`，防止缓冲破坏实时推送 |

---

## 十、可能被追问的技术细节

- **BM25 阈值怎么调的？** 跑 FAQ 查询的 BM25 分数分布 → 找"真命中"和"假命中"分界线 → raw_best≥1.8 + raw_avg≥2.0 双条件
- **子块多匹配同一父块怎么去重？** `{r["parent_id"] for r in results}` set 去重 → 取对应父块全文
- **Query 改写 4 策略？** ① 原始 query ② 关键词提取 ③ 口语转书面 ④ 多视角改写。已命中缓存则不触发
- **并行检索怎么做的？** 4 路 `asyncio.gather`（稠密/稀疏/混合/多向量）→ RRF(k=60) 融合，单路失败返回空列表不阻塞其他
- **异步写回 Redis？** `asyncio.create_task()` 后台写，不阻塞主链路，写失败打日志不影响用户
- **BERT 模型管理？** transformers 本地加载，首次 ~2s，常驻内存。失败 → LLM 零样本兜底
- **MCP 工具参数不合法？** 错误信息+原始 tool_call 还给 LLM 修正；超时→指数退避重试；空结果→宽泛条件重试
- **复合缓存键设计？** `{model}_{temperature}_{streaming}_{thinking}`——不同参数的 LLM 隔离缓存，防止 PyTorch 图缓存冲突
- **预加载 DLL 是什么？** Windows 多线程 DLL 加载竞态 `WinError 6714` → 启动时单线程预加载 pyarrow/pandas/sklearn/torch
- **4 路混合检索如果某一类挂了怎么办？** try/except 返回空列表，RRF 融合自动降维——某个检索源不可用，其他三个继续工作

---

## 十一、一句话总结

> 我做了一个电商智能客服 Agent 系统，用 5 个 Agent 协作 + 三层检索漏斗，把问答延迟从 3 秒降到 400 毫秒，通过高危门控和上传安全扫描解决了 LLM 幻觉和安全兜底问题。

---

> 📋 使用建议：面试前熟读「一、开场自我介绍」「三、方案选型」「六、高频问答」，做到脱口而出。
> 其余部分作为知识储备，被追问时自然展开。STAR 故事选 1-2 个最熟悉的准备好细节。
