#!/usr/bin/env python
# scripts/init_milvus_v2.py
# 创建 eco_rag 数据库 + policies/products 两个 Collection
# Schema: 稠密(1024) + 稀疏 + shop_id隔离 + 父子块
#
# 用法: python scripts/init_milvus_v2.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import (
    connections, utility, Collection, CollectionSchema,
    FieldSchema, DataType, MilvusClient,
)

DB_NAME = "eco_rag"
COLLECTIONS = ["policies", "products"]


def init():
    client = MilvusClient(uri="http://localhost:19530", timeout=15)

    # ── 1. 创建数据库 ──
    dbs = client.list_databases()
    if DB_NAME not in dbs:
        client.create_database(DB_NAME, properties={"database.replica.number": "1"})
        print(f"[OK] 数据库 '{DB_NAME}' 已创建")
    else:
        print(f"[SKIP] 数据库 '{DB_NAME}' 已存在")

    # 切换到新数据库（用 connections API）
    connections.connect(host="localhost", port=19530, db_name=DB_NAME)

    # ── 2. 定义 Schema ──
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128,
                    description="主键: {shop_id}_{type}_{doc_id}_{chunk_id}"),
        FieldSchema(name="shop_id", dtype=DataType.VARCHAR, max_length=64,
                    description="企业编号，租户隔离 key"),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128,
                    description="原始文档 ID"),
        FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=10,
                    description="parent 或 child"),
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=128,
                    description="子块指向父块 ID"),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535,
                    description="文本内容"),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024,
                    description="BGE-M3 稠密向量"),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR,
                    description="BGE-M3 稀疏向量"),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64,
                    description="分类标签"),
        FieldSchema(name="created_at", dtype=DataType.INT64,
                    description="创建时间戳"),
    ]

    schema = CollectionSchema(
        fields,
        description="电商RAG知识库 — 父子块+稠密稀疏+多租户",
        enable_dynamic_field=False,
    )

    # ── 3. 创建两个 Collection ──
    for col_name in COLLECTIONS:
        if utility.has_collection(col_name):
            print(f"[SKIP] Collection '{col_name}' 已存在")
            continue

        col = Collection(name=col_name, schema=schema)

        # 稠密向量索引
        dense_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        col.create_index(field_name="dense_vector", index_params=dense_params)

        # 稀疏向量索引
        sparse_params = {
            "metric_type": "IP",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"drop_ratio_build": 0.2},
        }
        col.create_index(field_name="sparse_vector", index_params=sparse_params)

        col.load()
        print(f"[OK] Collection '{DB_NAME}.{col_name}' 创建完成 (1024维稠密 + 稀疏)")

    connections.disconnect("default")
    print(f"\n[OK] Milvus v2 init done: {DB_NAME}.policies + {DB_NAME}.products")


if __name__ == "__main__":
    init()
