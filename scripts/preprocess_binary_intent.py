# scripts/preprocess_binary_intent.py
# 将 4 分类意图数据 + Chinese-EcomQA 数据集合并为二分类训练数据
#
# 二分类定义：
#   Class 0 — 信息咨询 (info): 商品/政策/知识问答/闲聊 → RAG 检索
#   Class 1 — 业务办理 (action): 查订单/物流/退货退款/投诉 → Business API
#
# 输出格式：JSONL，每行 {"query": "用户问题", "label": "info|action"}
#
# 用法：
#   python scripts/preprocess_binary_intent.py

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 4分类 → 2分类 映射 ──
LABEL_MAP_4TO2 = {
    "知识问答": "info",
    "知识管理": "info",
    "业务处理": "action",
    "工单处理": "action",
}

# ── Chinese-EcomQA task → 2分类 映射 ──
# 基于 task 定义合理映射
TASK_MAP = {
    "BC":  "info",    # 品牌知识问答
    "IDC": "info",    # 行业知识问答
    "SC":  "info",    # 拼写纠错 → info
    "IC":  "info",    # 商品类目 → info
    "CC":  "info",    # 品类细分 → info
    "ITC": "info",    # 意图分类训练数据 → 可用于 info
    "AC":  "action",  # 属性抽取 → 偏操作
    "RLC": "action",  # 相关性判断 → 偏操作
    "PC":  "info",    # 个性化推荐 → info
    "RVC": "action",  # 评论审核 → action
}

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "intent_binary_train.jsonl"
)


def load_existing_4class(data_path: str) -> list[dict]:
    """加载现有 4 分类数据，转为二分类。"""
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                item = json.loads(line)
                query = item.get("query", "").strip()
                label_4 = item.get("label", "").strip()
                label_2 = LABEL_MAP_4TO2.get(label_4)
                if query and label_2:
                    records.append({"query": query, "label": label_2})
            except json.JSONDecodeError:
                continue
    return records


def load_ecomqa_dataset() -> list[dict]:
    """从 Chinese-EcomQA 提取可用的意图数据。"""
    records = []
    try:
        # 去掉 HF 镜像，走代理直读
        os.environ.pop("HF_ENDPOINT", None)
        os.environ.pop("HF_HUB_OFFLINE", None)

        from datasets import load_dataset
        dataset = load_dataset("OpenStellarTeam/Chinese-EcomQA", split="train")

        for item in dataset:
            task = item.get("task", "")
            label_2 = TASK_MAP.get(task)
            if not label_2:
                continue

            # 提取 query 文本（prompt 字段包含 query）
            prompt = item.get("prompt", "")
            # 尝试提取 ***query*** 中的内容
            import re
            query_match = re.search(r'\*\*\*query\*\*\*[：:]\s*(.+?)(?:\*\*\*|\n)', prompt)
            if query_match:
                query = query_match.group(1).strip()
            else:
                # 试试简单的 prompt 清理
                query = prompt.replace("***query***", "").replace("***", "").strip()
                if len(query) > 200:
                    query = query[:200]

            if query and len(query) >= 3:
                records.append({"query": query, "label": label_2})
    except Exception as e:
        print(f"  [WARN] 加载 Chinese-EcomQA 失败: {e}，跳过")

    return records


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    all_records = []

    # 1. 现有 4 分类数据
    data_path = os.path.join(base_dir, "data", "intent_train_v4.json")
    if os.path.exists(data_path):
        existing = load_existing_4class(data_path)
        print(f"  现有4分类数据: {len(existing)} 条")
        all_records.extend(existing)
    else:
        print(f"  [WARN] 未找到 {data_path}")

    # 2. Chinese-EcomQA 数据
    print("  加载 Chinese-EcomQA 数据集...")
    ecomqa = load_ecomqa_dataset()
    print(f"  Chinese-EcomQA: {len(ecomqa)} 条")

    # 去重：按 query 去重，保留第一次出现的标签
    seen = set()
    deduped = []
    for r in all_records + ecomqa:
        key = r["query"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # 统计
    from collections import Counter
    label_counts = Counter(r["label"] for r in deduped)
    print(f"\n  === 最终数据集 ===")
    print(f"  info (信息咨询):  {label_counts.get('info', 0)}")
    print(f"  action (业务办理): {label_counts.get('action', 0)}")
    print(f"  总计: {len(deduped)}")

    # 写入
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
