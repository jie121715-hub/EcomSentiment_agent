# 云答智能客服 — 完整执行流程图

## 一、顶层 LangGraph 拓扑（graph.py 编译图）

```
                          ┌─────────────────────────────────────┐
                          │         POST /api/v1/chat           │
                          │    chat.py → run_shopping_guide()   │
                          └─────────────────┬───────────────────┘
                                            │
                          ┌─────────────────▼───────────────────┐
                          │        START: ShoppingGuideState    │
                          │   query, user_id, session_id,       │
                          │   history=[], perception=None,      │
                          │   route_decision=None, ...          │
                          └─────────────────┬───────────────────┘
                                            │
          ╔═════════════════════════════════╧═════════════════════════════════╗
          ║                        【节点 1】perceive                        ║
          ║                   backend/agents/perception/nodes.py             ║
          ╠══════════════════════════════════════════════════════════════════╣
          ║                                                                  ║
          ║  ┌───────────────────────────────────────────────────────────┐  ║
          ║  │              PerceptionAgent.perceive(query)              │  ║
          ║  │                                                           │  ║
          ║  │  ┌─────────────────────┐   ┌──────────────────────────┐  │  ║
          ║  │  │  analyze_sentiment  │   │ extract_intent_and_entities│  │  ║
          ║  │  │  (asyncio 并行)      │   │ (asyncio 并行)             │  │  ║
          ║  │  │                     │   │                            │  │  ║
          ║  │  │  优先级：            │   │  优先级：                   │  │  ║
          ║  │  │  ① 训练7分类BERT     │   │  ① 训练BERT意图分类器       │  │  ║
          ║  │  │  ② ModelScope 7class │   │  ② LLM结构化输出            │  │  ║
          ║  │  │  ③ 本地 BERT 模型    │   │  映射: 10分类→4分类         │  │  ║
          ║  │  │  ④ LLM零样本         │   │                            │  │  ║
          ║  │  └─────────┬───────────┘   └────────────┬─────────────┘  │  ║
          ║  │            │                             │                │  ║
          ║  │            ▼                             ▼                │  ║
          ║  │  Sentiment(极性) + SentimentLabel(细粒度)  IntentCategory │  ║
          ║  │            │                             │                │  ║
          ║  │            └──────────┬──────────────────┘                │  ║
          ║  │                       ▼                                   │  ║
          ║  │              refine_sentiment (LLM细分)                   │  ║
          ║  │              高置信度(>0.9) 跳过LLM                        │  ║
          ║  └───────────────────────────────────────────────────────────┘  ║
          ║                                                                  ║
          ║  输出: PerceptionResult { sentiment, sentiment_label,            ║
          ║         intent, entities, fine_intent, confidence }              ║
          ╚══════════════════════════════════════════════════════════════════╝
                                            │
          ╔═════════════════════════════════╧═════════════════════════════════╗
          ║                        【节点 2】route                           ║
          ║                   backend/agents/router/nodes.py                 ║
          ╠══════════════════════════════════════════════════════════════════╣
          ║                                                                  ║
          ║  ┌───────────────────────────────────────────────────────────┐  ║
          ║  │            RoutingAgent.route(perception, history)        │  ║
          ║  │                                                           │  ║
          ║  │  维度0: 知识管理关键词预检 → 重定向到 KNOWLEDGE_QA          │  ║
          ║  │                                                           │  ║
          ║  │  情感×意图修正:                                            │  ║
          ║  │    • 正面+escalate → knowledge_qa                         │  ║
          ║  │    • 订单号格式 → business                                │  ║
          ║  │    • 商品咨询词 → knowledge_qa                             │  ║
          ║  │                                                           │  ║
          ║  │  维度2: 情绪紧急度检测 ──→ CRITICAL → escalate(短路返回)    │  ║
          ║  │    ANGRY + (退款/投诉) → CRITICAL                          │  ║
          ║  │    ANXIOUS + 修改订单 → ELEVATED                           │  ║
          ║  │                                                           │  ║
          ║  │  维度1: 意图置信度门控 ──→ confidence < 0.50 → clarify      │  ║
          ║  │    例外: 包含订单号/快递单号时跳过门控                       │  ║
          ║  │                                                           │  ║
          ║  │  维度3: 意图→Agent分发 (INTENT_AGENT_MAP)                   │  ║
          ║  │    KNOWLEDGE_QA    → TargetAgent.KNOWLEDGE_QA             │  ║
          ║  │    BUSINESS        → TargetAgent.BUSINESS                 │  ║
          ║  │    KNOWLEDGE_MGMT  → TargetAgent.KNOWLEDGE_QA (已合并)    │  ║
          ║  │    ESCALATE        → TargetAgent.ESCALATE                 │  ║
          ║  │                                                           │  ║
          ║  │  动态指令构建: _build_extra_instruction(感知×意图)          │  ║
          ║  │    安抚焦虑 / 道歉愤怒 / 通俗解释困惑 / 顺势推荐开心        │  ║
          ║  └───────────────────────────────────────────────────────────┘  ║
          ║                                                                  ║
          ║  输出: RouteDecision { target_agent, needs_clarification,        ║
          ║         urgency, escalate_to_human, strategy, tone_instruction } ║
          ╚══════════════════════════════════════════════════════════════════╝
                                            │
                          ┌─────────────────▼───────────────────┐
                          │     resolve_context (shop_id)       │
                          │   backend/agents/context_resolver   │
                          └─────────────────┬───────────────────┘
                                            │
                          ┌─────────────────▼───────────────────┐
                          │       dispatch_router(state)        │
                          │        条件路由: 读取 target_agent    │
                          └─────────────────┬───────────────────┘
                                            │
               ┌───────────┬───────────────┼───────────────┐
               ▼           ▼               ▼               ▼
          clarify      escalate        business      knowledge_qa
               │           │               │               │
               ▼           ▼               ▼               ▼
        ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────┐
        │ clarify  │ │escalate  │ │  business    │ │ retrieve │
        │  _node   │ │  _node   │ │   _node      │ │  _node   │
        │          │ │          │ │               │ │          │
        │ 直接返回 │ │ 返回转人 │ │ LLM解析+确认 │ │ RAG检索  │
        │ 澄清消息 │ │ 工引导   │ │ +MySQL操作   │ │          │
        └────┬─────┘ └────┬─────┘ └──────┬───────┘ └────┬─────┘
             │           │              │               │
             │           │              │               ▼
             │           │              │        ┌──────────┐
             │           │              │        │  answer  │
             │           │              │        │  _node   │
             │           │              │        │          │
             │           │              │        │ 缓存检查 │
             │           │              │        │ 闲聊快速 │
             │           │              │        │ BM25 FAQ │
             │           │              │        │ 高危门控 │
             │           │              │        │ LLM生成  │
             │           │              │        │ 后处理   │
             │           │              │        └────┬─────┘
             │           │              │             │
             ▼           ▼              ▼             ▼
        ┌──────────────────────────────────────────────────────┐
        │                         END                          │
        │  返回 AgentResponse { success, message }             │
        └──────────────────────────────────────────────────────┘
```

---

## 二、KnowledgeQA 内部流程（核心路径，处理 ~80% 流量）

```
                      ┌───────────────────┐
                      │  knowledge_qa 路径  │
                      └─────────┬─────────┘
                                │
                      ┌─────────▼─────────┐
                      │  _retrieve_node   │ ← graph.py 节点适配
                      └─────────┬─────────┘
                                │
          ╔═════════════════════╧═════════════════════════════╗
          ║         retrieve_node(state) — nodes.py          ║
          ╠═══════════════════════════════════════════════════╣
          ║                                                   ║
          ║  skip_rag? ──YES──→ 返回 []                        ║
          ║     │                                             ║
          ║    NO                                             ║
          ║     │                                             ║
          ║     ▼                                             ║
          ║  retrieve_with_pipeline(query, source_filter)      ║
          ║     │                                             ║
          ║     ├──① Query Rewriter (4策略并行)               ║
          ║     │    • 指代消解 (coreference)                  ║
          ║     │    • 同义词扩展 (3 variations)               ║
          ║     │    • 子问题拆分 (sub-question)              ║
          ║     │    • 关键词+NER增强                          ║
          ║     │                                             ║
          ║     ├──② Hybrid Searcher (4路并行 + RRF融合)      ║
          ║     │    G1: Dense向量 (BGE-M3, Top50)            ║
          ║     │    G2: Sparse向量 (关键词, Top30)           ║
          ║     │    G3: Dense+Sparse混合 (加权, Top40)       ║
          ║     │    G4: 多向量检索 (Top30)                   ║
          ║     │    └── RRF k=60 融合去重                     ║
          ║     │                                             ║
          ║     ├──③ 父子块映射                               ║
          ║     │    子块检索 → parent_id去重 → 父块替换       ║
          ║     │                                             ║
          ║     ├──④ BGE-Reranker v2-m3 (Cross-Encoder精排)   ║
          ║     │                                             ║
          ║     └──⑤ Quality Check (首条 score ≥ 0.6 ?)      ║
          ║                                                   ║
          ║  输出: context_docs (Top-K), retrieval_meta        ║
          ╚═══════════════════════════════════════════════════╝
                                │
                      ┌─────────▼─────────┐
                      │   _answer_node    │ ← graph.py 节点适配
                      └─────────┬─────────┘
                                │
          ╔═════════════════════╧═════════════════════════════╗
          ║          answer_node(state) — nodes.py           ║
          ╠═══════════════════════════════════════════════════╣
          ║                                                   ║
          ║  ┌─────────────────────────────────────────────┐ ║
          ║  │ ① Redis 缓存检查 (MD5 hash, TTL=1h)         │ ║
          ║  │    命中 ──→ 直接返回                         │ ║
          ║  ├─────────────────────────────────────────────┤ ║
          ║  │ ② 闲聊快速通道 (_quick_reply_check)          │ ║
          ║  │    "你好"/"谢谢"/"再见" → 模板回复            │ ║
          ║  ├─────────────────────────────────────────────┤ ║
          ║  │ ③ BM25+MySQL FAQ (_try_fast_path)           │ ║
          ║  │    命中 ──→ 直接返回 (~10ms)                │ ║
          ║  │    商品咨询类自动跳过FAQ                     │ ║
          ║  ├─────────────────────────────────────────────┤ ║
          ║  │ ④ 高危门控 (is_high_risk)                   │ ║
          ║  │    退款/赔付/法律问题 + 无精准文档            │ ║
          ║  │    ──→ 强制转人工，不编造数字                │ ║
          ║  ├─────────────────────────────────────────────┤ ║
          ║  │ ⑤ 上下文构建                                 │ ║
          ║  │    知识库文档 + 对话历史格式化                │ ║
          ║  ├─────────────────────────────────────────────┤ ║
          ║  │ ⑥ Prompt 组装                              │ ║
          ║  │    • 推荐类 → recommend_prompt()             │ ║
          ║  │    • 问答类 → rag_prompt()                   │ ║
          ║  │    注入: tone + extra_instruction + context  │ ║
          ║  ├─────────────────────────────────────────────┤ ║
          ║  │ ⑦ LLM 生成 (temperature=0.1, DeepSeek)      │ ║
          ║  │    异常 → _get_fallback_message()            │ ║
          ║  ├─────────────────────────────────────────────┤ ║
          ║  │ ⑧ 答案后处理 (AnswerPostProcessor)          │ ║
          ║  │    • 幻觉检测 (LLM-as-judge)                 │ ║
          ║  │    • 敏感词过滤                              │ ║
          ║  │    • 格式规范化                              │ ║
          ║  ├─────────────────────────────────────────────┤ ║
          ║  │ ⑨ 异步写回 Redis 缓存                       │ ║
          ║  └─────────────────────────────────────────────┘ ║
          ║                                                   ║
          ║  输出: AgentResponse { success, message, ... }     ║
          ╚═══════════════════════════════════════════════════╝
```

---

## 三、Business Agent 内部流程

```
                      ┌───────────────────┐
                      │   business 路径    │
                      └─────────┬─────────┘
                                │
          ╔═════════════════════╧═════════════════════════════╗
          ║       BusinessAgent.handle(query, history)       ║
          ╠═══════════════════════════════════════════════════╣
          ║                                                   ║
          ║  ┌─────────────────────────────────────────────┐ ║
          ║  │ ① LLM 解析 (_parse_query)                   │ ║
          ║  │    action: logistics|order|stock|cancel|... │ ║
          ║  │    params: {order_id, sku, product_name...} │ ║
          ║  │    needs_confirm: true/false                │ ║
          ║  │    异常 → 关键词兜底 (_keyword_fallback)     │ ║
          ║  ├─────────────────────────────────────────────┤ ║
          ║  │ ② 参数校验                                  │ ║
          ║  │    缺order_id → _missing_info_response()    │ ║
          ║  │    确认场景 → 从历史/数据库提取order_id      │ ║
          ║  ├─────────────────────────────────────────────┤ ║
          ║  │ ③ 写操作 → 二次确认 (_build_confirm_message) │ ║
          ║  │    "确定要取消订单xxx吗？请回复确认"          │ ║
          ║  │    modify_address 无需确认                    │ ║
          ║  ├─────────────────────────────────────────────┤ ║
          ║  │ ④ 执行 (_execute)                           │ ║
          ║  │    ┌─ 读操作 ────────────────────────────┐  │ ║
          ║  │    │ logistics → _query_logistics (MySQL) │  │ ║
          ║  │    │ order    → _query_order (MySQL)     │  │ ║
          ║  │    │ stock    → _query_stock (MySQL)     │  │ ║
          ║  │    └─────────────────────────────────────┘  │ ║
          ║  │    ┌─ 写操作 ────────────────────────────┐  │ ║
          ║  │    │ cancel_order    → MySQL UPDATE      │  │ ║
          ║  │    │ apply_refund     → MySQL UPDATE      │  │ ║
          ║  │    │ modify_address   → MySQL UPDATE      │  │ ║
          ║  │    │ modify_quantity  → 生成工单(人工)     │  │ ║
          ║  │    └─────────────────────────────────────┘  │ ║
          ║  └─────────────────────────────────────────────┘ ║
          ╚═══════════════════════════════════════════════════╝
```

---

## 四、Clarify + Escalate 流程

```
  ┌─────────────────────────────────┐    ┌─────────────────────────────┐
  │         clarify 路径             │    │       escalate 路径          │
  └────────────────┬────────────────┘    └────────────┬────────────────┘
                   │                                  │
  ╔════════════════╧════════════════════╗  ╔══════════╧════════════════╗
  ║ _clarify_node                      ║  ║ _escalate_node             ║
  ╠════════════════════════════════════╣  ╠═══════════════════════════╣
  ║                                    ║  ║                           ║
  ║ route_decision 中已有澄清消息       ║  ║ ① 输出转人工引导消息       ║
  ║                                    ║  ║    客服电话 + 工单号      ║
  ║ 生成方式:                          ║  ║    安抚语句               ║
  ║ ① 模板匹配 (毫秒级)                ║  ║                           ║
  ║    5对常见意图混淆模板             ║  ║ ② 异步记录澄清日志到MySQL  ║
  ║ ② LLM动态生成 (兜底)              ║  ║    (ClarifyLog 表)        ║
  ║    生成2-3个选项反问               ║  ║                           ║
  ║                                    ║  ╚═══════════════════════════╝
  ║ 异步写澄清日志到MySQL (ClarifyLog)  ║
  ╚════════════════════════════════════╝
```

---

## 五、SSE 流式路径（run_shopping_guide_stream）

```
  POST /api/v1/chat/stream
          │
          ▼
  run_shopping_guide_stream(query, user_id, session_id, history)
          │
          ├── perception_agent.perceive(query)
          │       │
          │       ▼
          │   yield ChatEvent(event="perception", data={sentiment, intent, entities, confidence})
          │
          ├── routing_agent.route(perception, history)
          │       │
          │       ▼
          │   yield ChatEvent(event="route", data={target_agent, strategy, urgency, escalate, clarify, skip_rag})
          │
          ├── context_resolver.resolve()
          │       │
          │       ▼
          │   shop_id 确定（多租户隔离）
          │
          ├── needs_clarification?
          │       │
          │       YES → for char in clarification_question: yield ChatEvent(event="token", data=char)
          │              yield ChatEvent(event="done", data="clarify")
          │
          ├── target=ESCALATE?
          │       │
          │       YES → yield token 流式输出转人工消息
          │              yield ChatEvent(event="done", data="escalate")
          │
          ├── target=BUSINESS?
          │       │
          │       YES → biz_agent.handle() → yield token → done("business")
          │
          └── target=KNOWLEDGE_QA? (默认)
                  │
                  ├── _try_fast_path(Redis+BM25)
                  │       │
                  │       命中 → yield token → done("knowledge_qa_fast")
                  │
                  ├── kqa_agent.retrieve(query, decision, history, shop_id)
                  │       │
                  │       ▼
                  │   context_docs (RAG检索结果)
                  │
                  └── kqa_agent.answer_stream(query, perception, decision, context_docs, history)
                          │
                          ├── LLM.astream() → 逐token yield
                          │       yield ChatEvent(event="token", data=token)
                          │
                          └── yield ChatEvent(event="done", data="knowledge_qa")

  异常: yield ChatEvent(event="error", data=str(e))
```

---

## 六、RAG 三层检索漏斗

```
用户Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  第一层: Redis 精确缓存                                │
│  延迟:  毫秒级  |  命中率: ~30%                          │
│  Key = MD5(query + source_filter)                    │
│  命中 → 直接返回，跳过后续所有步骤                       │
└────────────┬────────────────────────────────────────┘
             │ 未命中
             ▼
┌─────────────────────────────────────────────────────┐
│  第二层: BM25 + MySQL FAQ                            │
│  延迟: ~10ms  |  命中率: ~50%                         │
│  jieba分词 → BM25Okapi打分 → softmax门控(>0.85)       │
│  命中 → MySQL取答案 → 写回Redis → 直接返回              │
└────────────┬────────────────────────────────────────┘
             │ 未命中
             ▼
┌─────────────────────────────────────────────────────┐
│  第三层: RAG 语义检索                                  │
│  延迟: ~2s  |  覆盖率: ~10%                           │
│  Query改写 → 混合检索 → RRF融合 → Reranker → LLM生成   │
└─────────────────────────────────────────────────────┘
```

---

## 七、文件对应关系

| 层 | 文件 | 职责 |
|---|---|---|
| **入口** | `main.py` | Uvicorn启动 |
| **API层** | `backend/main.py` | FastAPI应用入口, lifespan预加载 |
| | `backend/api/v1/chat.py` | POST /chat, POST /chat/stream |
| | `backend/api/v1/auth.py` | 认证接口 (login/register/me) |
| | `backend/api/v1/admin_knowledge.py` | 知识库 CRUD + 同步 |
| | `backend/api/v1/admin_upload.py` | 文件上传 (PDF/DOCX) |
| | `backend/api/v1/admin_import.py` | 淘宝商品导入 |
| **编排** | `backend/agents/graph.py` | 7节点LangGraph编译图, 4路分发 |
| **感知** | `backend/agents/perception/` | BERT情感+意图+LLM NER |
| **路由** | `backend/agents/router/` | 三维决策(置信度+紧急度+分发) |
| **上下文** | `backend/agents/context_resolver.py` | 多租户shop_id解析 |
| **知识问答** | `backend/agents/knowledge_qa/` | 三层检索漏斗→RAG→生成→后处理 |
| **业务** | `backend/agents/business/` | MySQL读写合一+确认流程 |
| **RAG** | `backend/rag/` | 检索/分块/缓存/重排/后处理/改写 |
| **模型** | `backend/models/` | Pydantic Schema + SQLAlchemy ORM |
| **核心** | `backend/core/` | LLM工厂/日志/重试/异常/数据库 |
| **训练** | `backend/training/` | BERT情感7分类+意图4分类+二分类 |
| **工具** | `backend/utils/` | 物流/淘宝导入/微信桥接/DLL预加载 |
| **前端** | `frontend/` | Vue 3 + Element Plus + Pinia SPA |
| **脚本** | `scripts/` | 环境检查/种子数据/模型下载/预处理 |
