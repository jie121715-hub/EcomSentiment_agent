#!/usr/bin/env python
# scripts/seed_tenant_data.py
# 给 a001 企业写入测试政策和商品（父子块切分 + BGE-M3 编码）
#
# 用法: python scripts/seed_tenant_data.py

import sys, os, time, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHOP_ID = "a001"
DB_NAME = "eco_rag"


# ── 测试数据 ──
TEST_POLICIES = [
    {
        "doc_id": f"{SHOP_ID}_policy_001",
        "content": """【退货政策】
1. 自签收之日起7天内，商品未使用、包装完好可申请无理由退货。
2. 退货运费由买家承担，除非商品存在质量问题。
3. 退货时请保留原始包装及所有配件。
4. 特殊商品（内衣、食品、定制商品）不支持无理由退货。
5. 退款将在收到退货商品后48小时内原路退回。
6. 如有疑问请联系在线客服，服务时间：每天9:00-21:00。""",
        "category": "after_sales",
    },
    {
        "doc_id": f"{SHOP_ID}_policy_002",
        "content": """【换货与保修】
1. 商品存在质量问题时，7天内可申请换货，15天内可申请维修。
2. 电子类产品提供2年质保服务，非人为损坏免费维修。
3. 换货商品须保留完整包装，配件齐全。
4. 质保期内在指定维修点凭购买凭证享受服务。
5. 人为损坏（进水、摔落、私自拆修）不在质保范围。
6. 质保期内维修超过3次的商品可申请换新。""",
        "category": "warranty",
    },
    {
        "doc_id": f"{SHOP_ID}_policy_003",
        "content": """【物流配送政策】
1. 全国大部分地区下单后48小时内发货，偏远地区72小时内。
2. 默认使用顺丰速运，偏远地区使用EMS。
3. 订单金额满99元包邮，不满99元收取8元运费。
4. 支持实时物流查询，可在订单详情中查看物流轨迹。
5. 发货后如7天未收到货，请联系客服核实。
6. 节假日发货可能延迟1-2天，敬请谅解。""",
        "category": "logistics",
    },
]

TEST_PRODUCTS = [
    {
        "doc_id": f"{SHOP_ID}_product_001",
        "content": """【智能蓝牙耳机 Pro】
价格：¥299 | 原价：¥499
品牌：SoundMax
颜色：曜石黑 / 珍珠白
主要特点：
- 蓝牙5.3芯片，连接距离20米
- ANC主动降噪，降噪深度-42dB
- 续航8小时，充电仓额外24小时
- IPX5防水等级，运动无惧汗水
- 13mm动圈单元，Hi-Res认证音质
适用场景：通勤、运动、办公、游戏
保修：2年质保，7天无理由退货""",
        "category": "electronics",
    },
    {
        "doc_id": f"{SHOP_ID}_product_002",
        "content": """【纯棉四件套 北欧简约风】
价格：¥259 | 原价：¥399
品牌：HomeLife
颜色：灰白条纹 / 雾霾蓝 / 奶油杏
材质：100%新疆长绒棉，60支高密度
包含：床单×1、被套×1、枕套×2
尺寸：1.5m床款（200×230cm）/ 1.8m床款（220×240cm）
特点：
- A类安全标准，母婴可用
- 水洗不易缩水，柔软亲肤
- 不起球不褪色，持久如新
洗涤建议：30℃温水洗涤，不可漂白""",
        "category": "home",
    },
]


def main():
    # ── 1. 加载 BGE-M3 ──
    print("加载 BGE-M3 模型...")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from FlagEmbedding import BGEM3FlagModel
    from backend.config import get_settings

    settings = get_settings()
    model_path = settings.embedding_model_name
    print(f"  模型路径: {model_path}")
    model = BGEM3FlagModel(model_path, use_fp16=False, local_files_only=True)
    print("  模型加载完成\n")

    # ── 2. 连接 Milvus ──
    from pymilvus import connections, Collection
    connections.connect(host="localhost", port=19530, db_name=DB_NAME)
    print(f"连接 Milvus 数据库: {DB_NAME}")

    # ── 3. 切分 + 编码 + 写入 ──
    from backend.rag.chunker import ParentChildChunker

    chunker = ParentChildChunker(
        parent_size=settings.rag_parent_chunk_size,
        parent_overlap=settings.rag_parent_chunk_overlap,
        child_size=settings.rag_child_chunk_size,
        child_overlap=settings.rag_child_chunk_overlap,
    )

    for col_name, datasets in [("policies", TEST_POLICIES), ("products", TEST_PRODUCTS)]:
        col = Collection(col_name)
        total_inserted = 0

        for data in datasets:
            doc_id = data["doc_id"]
            category = data["category"]
            content = data["content"]

            # 父子块切分
            from langchain_core.documents import Document
            raw_doc = Document(page_content=content, metadata={"doc_id": doc_id, "shop_id": SHOP_ID})
            result = chunker.split_documents([raw_doc])

            ts = int(time.time())
            rows = []

            # ── 父块 ──
            for i, p in enumerate(result.parent_chunks):
                pid = f"{doc_id}_parent"
                dense, sparse = _encode(model, p.page_content)
                sparse_dict = _sparse_to_dict(sparse)
                rows.append({
                    "id": pid,
                    "shop_id": SHOP_ID,
                    "doc_id": doc_id,
                    "chunk_type": "parent",
                    "parent_id": pid,
                    "content": p.page_content,
                    "dense_vector": dense.tolist(),
                    "sparse_vector": sparse_dict,
                    "category": category,
                    "created_at": ts,
                })

            # ── 子块 ──
            for j, c in enumerate(result.child_chunks):
                pid = f"{doc_id}_parent"
                cid = f"{doc_id}_child_{j}"
                dense, sparse = _encode(model, c.page_content)
                sparse_dict = _sparse_to_dict(c)
                rows.append({
                    "id": cid,
                    "shop_id": SHOP_ID,
                    "doc_id": doc_id,
                    "chunk_type": "child",
                    "parent_id": pid,
                    "content": c.page_content,
                    "dense_vector": dense.tolist(),
                    "sparse_vector": sparse_dict,
                    "category": category,
                    "created_at": ts,
                })

            if rows:
                col.insert(rows)
                total_inserted += len(rows)
                print(f"  [{col_name}] {doc_id}: {len(result.parent_chunks)}父块 + {len(result.child_chunks)}子块 → {len(rows)}条")

        col.flush()
        print(f"  [{col_name}] 总计写入 {total_inserted} 条, 实体数: {col.num_entities}\n")

    connections.disconnect("default")
    print("[OK] 种子数据写入完成")


def _encode(model, text: str):
    """BGE-M3 编码单条文本，返回 (dense_1d, sparse_dict)。"""
    result = model.encode([text], return_dense=True, return_sparse=True)
    dense = result["dense_vecs"][0]           # numpy 1D (1024,)
    sparse_list = result["lexical_weights"]    # list of defaultdict
    sparse = _sparse_to_dict(sparse_list[0]) if sparse_list else {}
    return dense, sparse


def _sparse_to_dict(sparse) -> dict:
    """BGE-M3 稀疏输出（defaultdict/稀疏矩阵）→ {int: float}。"""
    import numpy as np
    result = {}
    if sparse is None:
        pass
    elif isinstance(sparse, (dict, collections.defaultdict)):
        result = {int(k): float(v) for k, v in sparse.items()}
    elif hasattr(sparse, "indices") and hasattr(sparse, "data"):
        result = {int(i): float(v) for i, v in zip(sparse.indices, sparse.data)}
    elif hasattr(sparse, "_data"):
        row = sparse._data[0] if sparse._data else {}
        if isinstance(row, dict):
            result = {int(k): float(v) for k, v in row.items()}
        elif hasattr(row, "indices"):
            result = {int(i): float(v) for i, v in zip(row.indices, row.data)}
    # Milvus 不允许空稀疏向量，加占位符
    if not result:
        result = {0: 0.001}
    return result


if __name__ == "__main__":
    main()
