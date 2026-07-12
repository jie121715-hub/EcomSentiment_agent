# backend/rag/chunker.py
# 🆕 v3 父子块切分策略：子块检索(精准) → 父块返回(完整)。
#
# 核心思想（RAG 最佳实践）：
#   小块 = 语义更聚焦 → 检索精度更高（减少噪声）
#   大块 = 上下文更完整 → LLM 生成质量更高（避免信息截断）
#
# 策略：
#   父块 (Parent): 800~1200 字，overlap 200  — 提供给 LLM 的完整上下文
#   子块 (Child):  200~400 字，overlap 50   — 存入向量库用于语义检索
#   关联:           每个子块在 metadata 中记录 parent_id
#
# 检索流程：
#   用户query → 向量检索子块(Top-K children) → 提取parent_id去重
#            → 返回父块完整内容 → LLM生成
#
# 存储策略（Milvus/Chroma）：
#   - 子块存入向量库，带 metadata: {chunk_type:"child", parent_id:"xxx", doc_id:"yyy"}
#   - 父块也存入向量库，带 metadata: {chunk_type:"parent", parent_id:"xxx"}
#   - 检索时 filter: chunk_type=="child" → 找到子块 → 二次查父块

from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkResult:
    """父子块切分结果。"""
    parent_chunks: list[Document] = field(default_factory=list)   # 父块（大）
    child_chunks: list[Document] = field(default_factory=list)    # 子块（小，用于检索）
    parent_map: dict[str, str] = field(default_factory=dict)      # parent_id → parent_content
    total_docs: int = 0
    total_parents: int = 0
    total_children: int = 0


class ParentChildChunker:
    """父子块切分器。

    两步切分：
      1. 父块切分：大块（保证上下文完整，给 LLM 看）
      2. 子块切分：小块（保证检索精度，给向量库检索用）

    使用方式：
        chunker = ParentChildChunker()
        result = chunker.split_documents(raw_docs)
        # result.child_chunks  → 存入向量库用于检索
        # result.parent_map    → 检索后按 parent_id 取父块内容
    """

    def __init__(
        self,
        parent_size: int | None = None,
        parent_overlap: int | None = None,
        child_size: int | None = None,
        child_overlap: int | None = None,
    ):
        settings = get_settings()
        self.parent_size = parent_size or settings.rag_parent_chunk_size
        self.parent_overlap = parent_overlap or settings.rag_parent_chunk_overlap
        self.child_size = child_size or settings.rag_child_chunk_size
        self.child_overlap = child_overlap or settings.rag_child_chunk_overlap

        # 父块切分器
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_size,
            chunk_overlap=self.parent_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

        # 子块切分器
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_size,
            chunk_overlap=self.child_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

    def split_documents(self, documents: list[Document]) -> ChunkResult:
        """对一批文档执行父子块切分。

        :param documents: 原始 LangChain Document 列表
        :return: ChunkResult 包含父块、子块、映射表
        """
        if not documents:
            logger.warning("chunker.no_documents")
            return ChunkResult()

        logger.info("chunker.started", docs=len(documents))

        all_parents = []
        all_children = []
        parent_map = {}
        parent_counter = 0

        for doc_idx, doc in enumerate(documents):
            doc_id = doc.metadata.get("source", f"doc_{doc_idx}")

            # 步骤1: 切分为父块
            raw_parents = self.parent_splitter.split_documents([doc])

            for p_idx, parent in enumerate(raw_parents):
                parent_id = f"{doc_id}_p{parent_counter}"
                parent_counter += 1

                # 父块元数据
                parent.metadata.update({
                    "chunk_type": "parent",
                    "parent_id": parent_id,
                    "doc_id": doc_id,
                    "chunk_size": "parent",
                })
                all_parents.append(parent)

                # 记录 parent_id → 父块内容（用于检索后快速映射）
                parent_map[parent_id] = parent.page_content

                # 步骤2: 每个父块再切分为子块
                raw_children = self.child_splitter.split_documents([parent])

                for c_idx, child in enumerate(raw_children):
                    # 子块继承父块的 source 元数据 + 新增关联字段
                    child.metadata.update({
                        "chunk_type": "child",
                        "parent_id": parent_id,
                        "doc_id": doc_id,
                        "chunk_size": "child",
                        "child_index": c_idx,
                        # 保留原始来源信息
                        "source": parent.metadata.get("source", doc_id),
                    })
                    all_children.append(child)

        result = ChunkResult(
            parent_chunks=all_parents,
            child_chunks=all_children,
            parent_map=parent_map,
            total_docs=len(documents),
            total_parents=len(all_parents),
            total_children=len(all_children),
        )

        logger.info(
            "chunker.done",
            docs=len(documents),
            parents=result.total_parents,
            children=result.total_children,
            ratio=f"1:{result.total_children / max(result.total_parents, 1):.1f}",
        )
        return result

    def map_children_to_parents(
        self, child_docs: list[Document], parent_map: dict[str, str]
    ) -> list[str]:
        """将检索到的子块映射回父块内容（去重）。

        :param child_docs: 检索返回的子块文档
        :param parent_map: parent_id → parent_content 映射表
        :return: 去重后的父块内容列表
        """
        seen_parents = set()
        parent_contents = []

        for child in child_docs:
            parent_id = child.metadata.get("parent_id", "")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_content = parent_map.get(parent_id, "")
                if parent_content:
                    parent_contents.append(parent_content)

        logger.info(
            "chunker.child_to_parent",
            children=len(child_docs),
            unique_parents=len(parent_contents),
        )
        return parent_contents

    def map_children_to_parent_docs(
        self, child_docs: list[Document], parent_map: dict[str, str]
    ) -> list[Document]:
        """将检索到的子块映射回父块 Document 对象（去重，保留元数据）。

        :return: 父块 Document 列表
        """
        seen_parents = set()
        parent_docs = []

        for child in child_docs:
            parent_id = child.metadata.get("parent_id", "")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_content = parent_map.get(parent_id, "")
                if parent_content:
                    # 构建父块 Document，继承子块的 source 等元数据
                    parent_docs.append(Document(
                        page_content=parent_content,
                        metadata={
                            "parent_id": parent_id,
                            "source": child.metadata.get("source", "未知"),
                            "doc_id": child.metadata.get("doc_id", ""),
                            "retrieved_via": f"child_{child.metadata.get('child_index', '?')}",
                        },
                    ))

        return parent_docs


# ── 测试代码 ──
if __name__ == "__main__":
    from backend.core.logger import configure_logging
    configure_logging()

    chunker = ParentChildChunker()

    # 模拟文档
    test_doc = Document(
        page_content=(
            "商品名称：夏季纯棉T恤。材质：100%新疆长绒棉，柔软透气不起球。"
            "尺码：S/M/L/XL/XXL，请参考尺码表选购。颜色：白色、黑色、灰色、藏青。"
            "价格：99元/件，买二送一活动进行中。洗涤建议：冷水机洗，不可漂白，中温熨烫。"
            "售后政策：支持7天无理由退换货，退货需保持吊牌和包装完整。"
            "在订单页申请后24小时内审核，审核通过后3个工作日内退款。"
            "物流：默认中通快递，全国包邮（新疆西藏除外），下单后48小时内发货。"
            "优惠活动：满199减20，满299减50，叠加店铺券更优惠。"
            "好评返现：五星好评+晒图返5元红包。店铺评分4.9分，复购率35%。"
        ),
        metadata={"source": "products/t-shirt.txt"},
    )

    result = chunker.split_documents([test_doc])
    print(f"文档数: {result.total_docs}")
    print(f"父块数: {result.total_parents}")
    print(f"子块数: {result.total_children}")
    print(f"父子比: 1:{result.total_children // max(result.total_parents, 1)}")
    print(f"父块映射数: {len(result.parent_map)}")

    # 检查子块的 parent_id
    for i, child in enumerate(result.child_chunks[:3]):
        pid = child.metadata.get("parent_id", "?")
        print(f"\n子块[{i}]: parent_id={pid}")
        print(f"  内容: {child.page_content[:80]}...")
        print(f"  元数据: chunk_type={child.metadata.get('chunk_type')}")

    # 模拟检索后映射
    retrieved_children = result.child_chunks[:2]
    parents = chunker.map_children_to_parents(retrieved_children, result.parent_map)
    print(f"\n检索到 {len(retrieved_children)} 个子块 → 映射到 {len(parents)} 个父块")
    if parents:
        print(f"父块内容预览: {parents[0][:120]}...")

    print("\nchunker.py 自测完成")
