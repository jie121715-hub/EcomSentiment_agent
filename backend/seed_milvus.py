"""
Milvus Collection 初始化 — Collection级多租户

两个 Collection (pymilvus 3.0 动态 Schema):
  ecom_products_v1  — 商品描述向量 (shop_id/product_id/category 动态字段)
  ecom_policies_v1  — 店铺政策向量 (shop_id/policy_type 动态字段)

每条记录插入时带 shop_id 实现多租户隔离。
检索时用 filter='shop_id=="xxx"' 过滤。

用法:
  python -m backend.seed_milvus             # 创建（幂等）
  python -m backend.seed_milvus --drop       # 删除重建
"""

import argparse
from pymilvus import MilvusClient

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

def _coll_cfg(name: str, desc: str) -> dict:
    s = get_settings()
    return {
        "name": name,
        "dimension": s.milvus_embedding_dim,
        "description": desc,
        "metric": "COSINE",
    }

COLLECTIONS = [
    _coll_cfg("ecom_products_v1", "商品描述向量 — product_inquiry/recommend_request"),
    _coll_cfg("ecom_policies_v1", "店铺政策向量 — after_sales/complaint"),
]


def create_collections(drop_first: bool = False):
    """创建 Milvus Collections (pymilvus 3.0 动态 Schema, 幂等)"""
    s = get_settings()
    client = MilvusClient(uri=f"http://{s.milvus_host}:{s.milvus_port}")

    existing = client.list_collections()
    print(f"Existing: {existing}")

    for cfg in COLLECTIONS:
        name = cfg["name"]

        if name in existing:
            if drop_first:
                client.drop_collection(name)
                print(f"  DROPPED: {name}")
            else:
                print(f"  SKIP (exists): {name}")
                continue

        # pymilvus 3.0: create_collection(dimension=...) 自动建索引，无需手动 create_index
        client.create_collection(
            collection_name=name,
            dimension=cfg["dimension"],
            metric_type=cfg["metric"],
            auto_id=True,
            description=cfg["description"],
        )
        client.load_collection(name)

        print(f"  CREATED: {name} (dim={cfg['dimension']}, {cfg['metric']})")

    final = client.list_collections()
    print(f"\nFinal collections: {final}")
    for cn in final:
        try:
            desc = client.describe_collection(cn)
            print(f"  {cn}: {desc.get('num_entities', '?')} entities")
        except Exception:
            print(f"  {cn}: (describe failed)")

    client.close()
    print("Done.")


if __name__ == "__main__":
    from backend.core.logger import configure_logging
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()

    create_collections(drop_first=args.drop)
