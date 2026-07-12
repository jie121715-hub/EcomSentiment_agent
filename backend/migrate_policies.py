"""
迁移脚本：ecom_knowledge_v1 → ecom_policies_v1

步骤:
  1. 从旧Collection提取所有 text + source
  2. 合并为完整文档，用ParentChildChunker做父子块切分
  3. BGE-M3 重新编码子块 (1024维)
  4. 子块写入 ecom_policies_v1, 父块映射表留 retriever 用
"""

import asyncio
from pymilvus import MilvusClient

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


async def migrate():
    s = get_settings()
    client = MilvusClient(uri=f"http://{s.milvus_host}:{s.milvus_port}")

    # ── 1. 提取旧数据 ──────────────────────────────
    print("Step 1: Extracting old data...")
    all_texts = []

    # 分批 query (pk >= offset)
    batch_size = 500
    offset = 0
    while True:
        res = client.query(
            collection_name="ecom_knowledge_v1",
            filter=f"pk >= {offset}",
            output_fields=["pk", "source", "text"],
            limit=batch_size,
        )
        if not res:
            break
        for r in res:
            pk = r["pk"]
            txt = r.get("text", "")
            src = r.get("source", "unknown")
            if txt and len(txt.strip()) > 10:
                all_texts.append({"pk": pk, "text": txt.strip(), "source": src})
        offset = max(r["pk"] for r in res) + 1
        print(f"  extracted {len(all_texts)} so far (offset={offset})...")
        if len(res) < batch_size:
            break

    print(f"  Total: {len(all_texts)} text chunks")

    if not all_texts:
        print("No data to migrate.")
        return

    # ── 2. 合并为文档 + 父子块切分 ──────────────────
    print("\nStep 2: Parent-child chunking...")
    from langchain_core.documents import Document
    from backend.rag.chunker import ParentChildChunker

    # 按 source 聚合内容
    docs_by_source = {}
    for item in all_texts:
        src = item["source"]
        if src not in docs_by_source:
            docs_by_source[src] = []
        docs_by_source[src].append(item["text"])

    documents = []
    for src, texts in docs_by_source.items():
        full_text = "\n\n".join(texts)
        # 截断过长的文档（Milvus VARCHAR max 4096, 父块1000字即可）
        documents.append(Document(
            page_content=full_text[:10000],
            metadata={"source": src, "knowledge_type": "policy"},
        ))

    chunker = ParentChildChunker()
    result = chunker.split_documents(documents)
    print(f"  Parents: {result.total_parents}, Children: {result.total_children}")

    # ── 3. BGE-M3 编码子块 ────────────────────────
    print("\nStep 3: Encoding with BGE-M3...")
    from langchain_community.embeddings import HuggingFaceEmbeddings

    model_path = s.embedding_model_name
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    children_texts = [c.page_content for c in result.child_chunks]
    print(f"  Encoding {len(children_texts)} chunks (this may take a while)...")

    # 分批编码
    vectors = []
    batch = 50
    for i in range(0, len(children_texts), batch):
        batch_texts = children_texts[i:i+batch]
        batch_vecs = embeddings.embed_documents(batch_texts)
        vectors.extend(batch_vecs)
        if (i + batch) % 200 == 0:
            print(f"    {i + len(batch_texts)}/{len(children_texts)} encoded")

    print(f"  Encoded {len(vectors)} vectors, dim={len(vectors[0])}")

    # ── 4. 写入 ecom_policies_v1 ───────────────────
    print("\nStep 4: Inserting into ecom_policies_v1...")
    shop_id = s.default_shop_id

    # 推断 policy_type
    def infer_policy_type(source: str) -> str:
        src_lower = source.lower()
        if any(w in src_lower for w in ["after_sales", "售后", "退货", "退款", "换货"]):
            return "after_sales"
        if any(w in src_lower for w in ["shipping", "物流", "配送", "运费"]):
            return "shipping"
        if any(w in src_lower for w in ["warranty", "质保", "保修"]):
            return "warranty"
        if any(w in src_lower for w in ["promotion", "活动", "优惠", "促销"]):
            return "promotion"
        return "general"

    insert_data = []
    for i, child in enumerate(result.child_chunks):
        src = child.metadata.get("source", "unknown")
        insert_data.append({
            "vector": vectors[i],
            "shop_id": shop_id,
            "content": child.page_content[:4000],  # truncate to max_length
            "knowledge_type": "policy",
            "policy_type": infer_policy_type(src),
            "source": src[:120],
            "category": "policy",
        })

    # 分批插入
    insert_batch = 200
    for i in range(0, len(insert_data), insert_batch):
        batch_data = insert_data[i:i+insert_batch]
        client.insert(
            collection_name="ecom_policies_v1",
            data=batch_data,
        )
        print(f"  inserted {i+len(batch_data)}/{len(insert_data)}")

    print(f"\nDone! Migrated {len(insert_data)} child chunks into ecom_policies_v1")
    print(f"Parent map has {len(result.parent_map)} entries (kept in retriever memory on next load)")
    client.close()


if __name__ == "__main__":
    from backend.core.logger import configure_logging
    configure_logging()
    asyncio.run(migrate())
