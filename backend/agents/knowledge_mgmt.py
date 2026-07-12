# backend/agents/knowledge_mgmt.py
# 🆕 知识收纳Agent：负责知识库的录入、更新、查询管理。
#
# 能力：
#   1. 接收商户提供的商品/政策知识，写入知识库
#   2. 更新已有知识（覆盖旧版）
#   3. 查看当前知识库内容摘要
#
# 写入策略（双写）：
#   MySQL  → 持久化备份 + 审计追溯
#   Milvus/Chroma → 向量语义检索（让 KnowledgeQAAgent 的 RAG 能搜到）
#
# 权限约束：本Agent仅供商户使用（前端/鉴权层控制），普通用户不应触发此流程。

import time
from backend.core.logger import get_logger
from backend.core.retry import with_retry_async
from backend.models.schemas import AgentResponse, AgentMessage, ConversationHistory

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = get_logger(__name__)

# ── 父子块切分器 ──────────────────────────────────────────
from backend.rag.chunker import ParentChildChunker


class KnowledgeMgmtAgent:
    """知识收纳Agent — 管理电商知识库的录入与更新。

    供商户使用：上传店铺政策、商品说明、售后规则等。
    """

    def __init__(self):
        self.retriever = None  # 懒加载，避免启动时联网

    def _get_retriever(self):
        if self.retriever is None:
            from backend.rag.retriever import EcomRetriever  # 惰性导入，避免触发重依赖
            self.retriever = EcomRetriever()
        return self.retriever

    async def handle(
        self,
        query: str,
        history: list[ConversationHistory] | None = None,
        merchant_id: str | None = None,
    ) -> AgentResponse:
        """处理商户知识管理请求。

        :param query: 用户输入
        :param history: 对话历史
        :param merchant_id: 商户标识（用于标注知识来源和按商户隔离检索）
        """
        start = time.time()
        logger.info(
            "knowledge_mgmt.started",
            query=query[:50],
            merchant_id=merchant_id or "unknown",
        )

        # 判断子意图
        is_add = any(w in query for w in ["添加", "新增", "加入", "记录", "记下来", "保存", "存入", "上传"])
        is_view = any(w in query for w in ["查看", "有哪些", "列出", "看看", "有什么"])
        is_delete = any(w in query for w in ["删除", "移除", "去掉"])

        if is_view:
            return await self._view_knowledge()
        elif is_delete:
            return await self._delete_knowledge(query)
        elif is_add:
            return await self._add_knowledge(query, merchant_id=merchant_id)
        else:
            # 默认：查看现有知识
            return await self._view_knowledge()

    # ── 添加知识 ─────────────────────────────────────────────────

    async def _add_knowledge(
        self,
        query: str,
        merchant_id: str | None = None,
    ) -> AgentResponse:
        """从商户消息中提取知识，双写到 MySQL + 向量库。

        流程：
        1. LLM 提取结构化知识
        2. 写入 MySQL（持久化）
        3. 向量化后写入 Milvus/Chroma（供 RAG 检索）
        """
        start = time.time()
        from backend.core.llm_factory import get_llm

        # 步骤1：LLM 提取知识
        extract_prompt = f"""你是一个电商知识提取助手。请从以下商户消息中提取电商知识，格式化为一段流畅的话（50-300字）。
提取范围：
- 商品信息：材质、尺码、颜色、适用人群、使用方法、注意事项
- 政策规则：退换货规则、运费规则、优惠活动、质保政策
- 售后流程：退换货步骤、联系方式、处理时效

如果消息中没有可提取的实质性知识，回复"无知识"。

商户消息：{query}

提取的知识："""

        llm = get_llm("qa", temperature=0)
        response = await with_retry_async(llm.ainvoke, extract_prompt)
        knowledge = response.text.strip() if hasattr(response, 'text') else str(response).strip()

        if "无知识" in knowledge or len(knowledge) < 10:
            elapsed = (time.time() - start) * 1000
            return AgentResponse(
                success=True,
                message=AgentMessage(
                    role="assistant",
                    content=(
                        "我没有从您的消息中提取到可记录的知识。请描述具体的商品信息、政策规则或操作流程，例如：\n\n"
                        "💡 \"这款T恤是100%纯棉的，支持机洗，建议冷水洗涤\"\n"
                        "💡 \"退货规则：7天内无理由退换，需保持吊牌完整，通过订单页申请即可\"\n"
                        "💡 \"满99元包邮，不满的收8元运费，偏远地区加收5元\"\n\n"
                        "请重新发送，我会帮您录入知识库。"
                    ),
                    intent_detected="knowledge_mgmt",
                ),
                processing_time_ms=elapsed,
            )

        # 步骤2：写入 MySQL（持久化备份）
        from backend.core.database import get_session
        from backend.models.db_models import CustomKnowledge

        merchant_tag = merchant_id or "默认商户"
        mysql_ok = False
        kb_id = None

        try:
            async with get_session() as session:
                record = CustomKnowledge(
                    content=knowledge,
                    source=f"merchant:{merchant_tag}",
                    category="general",
                    merchant_id=merchant_tag,
                )
                session.add(record)
                await session.commit()
                kb_id = record.id
                mysql_ok = True
                logger.info("knowledge_mgmt.written_to_mysql", id=kb_id, merchant=merchant_tag)
        except Exception as e:
            logger.error("knowledge_mgmt.mysql_write_failed", error=str(e))

        # 步骤3：写入向量库（Milvus/Chroma，供 RAG 语义检索）
        vector_ok = False
        chunk_count = 0
        try:
            retriever = self._get_retriever()
            if retriever.vector_store is not None:
                # 分块
                chunks = _KNOWLEDGE_SPLITTER.split_text(knowledge)
                # 转为 LangChain Document，带上来源元数据
                docs = [
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": f"merchant:{merchant_tag}",
                            "knowledge_id": str(kb_id) if kb_id else "unknown",
                            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                    )
                    for chunk in chunks
                ]
                retriever.vector_store.add_documents(docs)
                vector_ok = True
                chunk_count = len(chunks)
                logger.info(
                    "knowledge_mgmt.written_to_vector",
                    backend=retriever._backend,
                    mysql_id=kb_id,
                    chunks=chunk_count,
                )
            else:
                logger.warning("knowledge_mgmt.vector_store_unavailable")
        except Exception as e:
            logger.error("knowledge_mgmt.vector_write_failed", error=str(e))

        # 步骤4：汇总结果
        elapsed = (time.time() - start) * 1000

        status_parts = []
        if mysql_ok:
            status_parts.append(f"MySQL (ID:{kb_id})")
        if vector_ok:
            backend_name = self._get_retriever()._backend or "向量库"
            status_parts.append(f"{backend_name} ({chunk_count}个分块)")
        if not mysql_ok and not vector_ok:
            return AgentResponse(
                success=False,
                message=AgentMessage(
                    role="assistant",
                    content=(
                        f"知识提取成功但写入失败，请联系管理员检查数据库和向量库连接。\n\n"
                        f"待写入内容：\n{knowledge[:200]}..."
                    ),
                    intent_detected="knowledge_mgmt",
                ),
                processing_time_ms=elapsed,
            )

        # 统计总数
        total = "?"
        try:
            from backend.core.database import get_session
            from backend.models.db_models import CustomKnowledge
            from sqlalchemy import select, func
            async with get_session() as session:
                total = str((await session.execute(select(func.count(CustomKnowledge.id)))).scalar())
        except Exception:
            pass

        status_text = "、".join(status_parts)
        return AgentResponse(
            success=True,
            message=AgentMessage(
                role="assistant",
                content=(
                    f"✅ 知识已录入！\n\n"
                    f"📝 内容：{knowledge}\n\n"
                    f"📦 写入位置：{status_text}\n"
                    f"📊 知识库当前共 {total} 条记录\n\n"
                    f"后续用户咨询相关问题时，我会自动从知识库中检索这些内容。"
                ),
                intent_detected="knowledge_mgmt",
            ),
            processing_time_ms=elapsed,
        )

    # ── 查看知识 ─────────────────────────────────────────────────

    async def _view_knowledge(self) -> AgentResponse:
        """查看当前知识库摘要。"""
        from backend.core.database import get_session
        from backend.models.db_models import CustomKnowledge
        from sqlalchemy import select, func

        async with get_session() as session:
            total = (await session.execute(select(func.count(CustomKnowledge.id)))).scalar()

            if total == 0:
                return AgentResponse(
                    success=True,
                    message=AgentMessage(
                        role="assistant",
                        content=(
                            "📚 知识库目前还没有商户录入的自定义知识。\n\n"
                            "您可以这样添加：\n"
                            "• \"帮我记一下：这款防晒霜SPF50，适合油性皮肤\"\n"
                            "• \"添加知识：退货需要在7天内申请，超期不受理\"\n"
                            "• \"上传规则：满199包邮，生鲜类不支持无理由退货\""
                        ),
                        intent_detected="knowledge_mgmt",
                    ),
                )

            result = await session.execute(
                select(CustomKnowledge).order_by(CustomKnowledge.id.desc()).limit(10)
            )
            entries = result.scalars().all()

        summary = f"📚 知识库当前共 {total} 条自定义知识（最近10条）：\n\n"
        for i, entry in enumerate(entries, 1):
            content_preview = entry.content[:100] + ('...' if len(entry.content) > 100 else '')
            source_tag = getattr(entry, 'source', '未知来源') or '未知来源'
            summary += f"{i}. {content_preview}\n"
            summary += f"   来源: {source_tag}"
            if entry.created_at:
                summary += f" | {entry.created_at.strftime('%Y-%m-%d %H:%M')}"
            summary += "\n\n"

        if total > 10:
            summary += f"...还有 {total - 10} 条更早的知识"

        return AgentResponse(
            success=True,
            message=AgentMessage(
                role="assistant",
                content=summary,
                intent_detected="knowledge_mgmt",
            ),
        )

    # ── 批量同步：MySQL → 向量库 ──────────────────────────────

    async def sync_all_to_vector(self) -> dict:
        """将 MySQL 中所有知识全量同步到向量库（父子块策略）。

        流程：
          1. 读 MySQL custom_knowledge → 转为 Document
          2. ParentChildChunker 切分 → 父块 + 子块（带 parent_id 关联）
          3. 清空 collection → 写入子块（检索用）+ 父块（LLM上下文用）
          4. 更新 retriever._parent_map（内存映射表）

        :return: {"success": bool, "synced": int, "chunks": int, "backend": str, "error": str}
        """
        from backend.core.database import get_session
        from backend.models.db_models import CustomKnowledge
        from sqlalchemy import select

        try:
            retriever = self._get_retriever()
            if retriever.vector_store is None:
                return {"success": False, "synced": 0, "chunks": 0, "backend": "none", "error": "向量库未初始化"}

            async with get_session() as session:
                result = await session.execute(select(CustomKnowledge))
                records = result.scalars().all()

            if not records:
                logger.info("knowledge_mgmt.sync_empty")
                return {"success": True, "synced": 0, "chunks": 0, "backend": retriever._backend or "unknown"}

            # ── 构建 Document 列表 ────────────────────
            documents = []
            for record in records:
                documents.append(Document(
                    page_content=record.content,
                    metadata={
                        "source": record.source or "merchant:default",
                        "knowledge_id": str(record.id),
                        "category": record.category or "general",
                        "merchant_id": record.merchant_id or "default",
                    },
                ))

            # ── 父子块切分 ────────────────────────────
            chunker = ParentChildChunker()
            chunk_result = chunker.split_documents(documents)

            logger.info("knowledge_mgmt.chunking_done",
                       records=len(records),
                       parents=chunk_result.total_parents,
                       children=chunk_result.total_children,
                       ratio=f"1:{chunk_result.total_children // max(chunk_result.total_parents, 1)}")

            # ── 子块写入向量库（检索用）───────────────
            # 每个子块标记 chunk_type="child" + parent_id
            retriever.vector_store.add_documents(chunk_result.child_chunks)

            # ── 父块也写入向量库（完整上下文用）────────
            # 标记 chunk_type="parent" + parent_id
            retriever.vector_store.add_documents(chunk_result.parent_chunks)

            # ── 更新 retriever 的父块映射表（内存缓存）─
            retriever._parent_map = chunk_result.parent_map

            logger.info("knowledge_mgmt.sync_completed",
                       synced=len(records),
                       parents=chunk_result.total_parents,
                       children=chunk_result.total_children,
                       backend=retriever._backend)
            return {
                "success": True,
                "synced": len(records),
                "chunks": chunk_result.total_children,
                "parents": chunk_result.total_parents,
                "backend": retriever._backend or "unknown",
            }

        except Exception as e:
            logger.error("knowledge_mgmt.sync_failed", error=str(e))
            return {"success": False, "synced": 0, "chunks": 0, "backend": "unknown", "error": str(e)}

    # ── 删除知识 ─────────────────────────────────────────────────

    async def _delete_knowledge(self, query: str) -> AgentResponse:
        """删除指定知识（从 MySQL + 向量库同步删除）。

        v1 简化：按 ID 精确删除。后续可扩展为语义匹配删除。
        """
        import re
        from backend.core.database import get_session
        from backend.models.db_models import CustomKnowledge
        from sqlalchemy import select

        # 尝试从消息中提取 ID
        id_match = re.search(r'\b(\d+)\b', query)
        if not id_match:
            return AgentResponse(
                success=True,
                message=AgentMessage(
                    role="assistant",
                    content=(
                        "请告诉我您要删除的知识 ID。\n"
                        "例如：\"删除知识 3\"\n\n"
                        "💡 您可以用\"查看知识\"先确认要删除的编号。"
                    ),
                    intent_detected="knowledge_mgmt",
                ),
            )

        kb_id = int(id_match.group(1))

        async with get_session() as session:
            result = await session.execute(
                select(CustomKnowledge).where(CustomKnowledge.id == kb_id)
            )
            record = result.scalar_one_or_none()

            if record is None:
                return AgentResponse(
                    success=False,
                    message=AgentMessage(
                        role="assistant",
                        content=f"未找到 ID 为 {kb_id} 的知识记录，请确认编号是否正确。",
                        intent_detected="knowledge_mgmt",
                    ),
                )

            content_snapshot = record.content[:100]
            await session.delete(record)
            await session.commit()

        logger.info("knowledge_mgmt.deleted_from_mysql", id=kb_id)

        # 向量库删除：v1 暂不支持按 metadata 精确删除单个文档，
        # 通知用户向量库中可能仍有残留（会在下次全量重建时清理）
        return AgentResponse(
            success=True,
            message=AgentMessage(
                role="assistant",
                content=(
                    f"✅ 知识 ID:{kb_id} 已从 MySQL 删除。\n\n"
                    f"被删除的内容：{content_snapshot}...\n\n"
                    f"⚠️ 向量库中的对应分块可能需要手动清理或等待下次全量重建时自动移除。"
                ),
                intent_detected="knowledge_mgmt",
            ),
        )


# ── 测试代码 ──
if __name__ == "__main__":
    import asyncio
    from backend.core.logger import configure_logging
    configure_logging()

    async def test():
        agent = KnowledgeMgmtAgent()

        # 测试添加
        print("=" * 60)
        print("测试：商户添加退货规则")
        result = await agent.handle(
            "帮我记一下：本店支持7天无理由退换货，退货需保持吊牌和包装完整，"
            "在订单页申请后24小时内审核，审核通过后3个工作日内退款到原账户。",
            merchant_id="shop_test_001",
        )
        print(f"添加结果: {result.message.content[:300]}...")

        # 测试查看
        print()
        print("=" * 60)
        print("测试：查看知识库")
        result = await agent.handle("查看有哪些知识")
        print(f"查看结果: {result.message.content[:500]}...")

        print("\nknowledge_mgmt.py 自测完成")

    asyncio.run(test())
