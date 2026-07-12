# backend/rag/reranker.py
# 🆕 v3 BGE-Reranker 精排器：对混合检索结果进行 Cross-Encoder 深度打分。
#
# 流程：
#   粗排(Top 50~100) → RRF融合 → Reranker精排 → Top 5 → 质检(首条 > 0.6?)
#
# 模型：BAAI/bge-reranker-v2-m3 (Cross-Encoder, 多语言)
# 降级：模型不可用时用 RRF 分数代替

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


class EcomReranker:
    """BGE-Reranker 精排器 — Cross-Encoder 深度相关性打分。

    粗排是双塔模型（query/doc 独立编码 → 余弦相似度），速度快但精度有限。
    精排是 Cross-Encoder（query+doc 拼接 → 联合编码），精度高但速度慢。
    在对 50~100 个粗排候选做精排后取 Top 5，兼顾精度和效率。

    使用方式：
        reranker = EcomReranker()
        ranked = await reranker.rerank(query, docs, top_n=5)
    """

    def __init__(self):
        self.settings = get_settings()
        self.enabled = self.settings.rag_reranker_enabled
        self._model = None
        self._model_available = False
        self._init_model()

    def _init_model(self):
        """尝试加载 BGE-Reranker 模型。"""
        if not self.enabled:
            logger.info("reranker.disabled_by_config")
            return

        try:
            from FlagEmbedding import FlagReranker
            model_name = self.settings.rag_reranker_model
            self._model = FlagReranker(
                model_name,
                use_fp16=True,
                devices=["cpu"],  # 线上可改为 cuda
            )
            self._model_available = True
            logger.info("reranker.model_loaded", model=model_name)
        except ImportError:
            logger.warning("reranker.flag_embedding_not_installed",
                          hint="pip install FlagEmbedding 以启用精排")
        except Exception as e:
            logger.warning("reranker.init_failed", error=str(e))

    async def rerank(
        self,
        query: str,
        docs: list,
        top_n: int | None = None,
    ) -> list:
        """对文档列表进行精排。

        :param query: 用户问题
        :param docs: 粗排候选文档列表
        :param top_n: 返回 Top-N 文档
        :return: 按相关性降序排列的文档列表
        """
        k = top_n or self.settings.rag_retrieval_k

        if not docs:
            return []

        if not self._model_available or len(docs) <= k:
            return docs[:k]

        try:
            # 构建 (query, doc) 对
            pairs = []
            for doc in docs:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                pairs.append([query, content])

            # Cross-Encoder 打分
            scores = self._model.compute_score(pairs, normalize=True)

            # 按分数排序
            scored = list(zip(docs, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            top = [doc for doc, _ in scored[:k]]

            logger.info(
                "reranker.done",
                candidates=len(docs),
                top_score=round(scored[0][1], 4) if scored else 0,
                top_n=len(top),
            )
            return top

        except Exception as e:
            logger.warning("reranker.rerank_failed", error=str(e), fallback="rrf_order")
            return docs[:k]

    def check_quality(self, docs: list, threshold: float | None = None) -> dict:
        """检索质量检查：首条文档分数是否达标。

        :param docs: 精排后的文档列表
        :param threshold: 质量阈值（默认从 config 读取）
        :return: {"passed": bool, "top_score": float, "reason": str}
        """
        thresh = threshold or self.settings.rag_relevance_threshold

        if not docs:
            return {
                "passed": False,
                "top_score": 0.0,
                "reason": "检索结果为空，无可用文档",
            }

        # 如果有 reranker 分数，用 reranker 分数判断
        # 否则用文档相似度（如果有的话）
        top_doc = docs[0]
        score = getattr(top_doc, '_score', 0.0)

        if score >= thresh:
            return {"passed": True, "top_score": score, "reason": ""}

        return {
            "passed": False,
            "top_score": score,
            "reason": f"首条相关性 {score:.2f} < 阈值 {thresh}，检索质量不足",
        }

    def get_fallback_message(self) -> str:
        """检索质量不足时的兜底话术。"""
        from backend.config import get_settings
        phone = get_settings().customer_service_phone
        return (
            "非常抱歉，我暂时无法为您找到准确的答案。\n\n"
            "建议您：\n"
            f"• 换个方式描述您的问题，我再帮您查\n"
            f"• 拨打客服电话 {phone} 联系人工客服\n"
            f"• 在「我的订单」中查看相关信息\n\n"
            "我会继续学习，争取下次能更好地为您服务～"
        )


# ── 测试代码 ──
if __name__ == "__main__":
    print("EcomReranker — 需 FlagEmbedding 依赖")
    reranker = EcomReranker()
    print(f"精排可用: {reranker._model_available}")
    if not reranker._model_available:
        print("提示: pip install FlagEmbedding 以启用 BGE-Reranker 精排")
