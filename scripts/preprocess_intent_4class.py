# scripts/preprocess_intent_4class.py
# 合并旧数据(10分类→4分类) + v4数据 → 统一4分类 + LLM扩充少数类

import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 旧10类中文 → 4类中文映射
_OLD_10_TO_4 = {
    "产品咨询": "知识问答", "价格优惠": "知识问答", "求推荐": "知识问答",
    "闲聊": "知识问答", "其他": "知识问答",
    "查订单": "业务处理", "改订单": "业务处理", "售后": "业务处理",
    "知识管理": "知识管理",
    "投诉": "工单处理",
}

_LABEL_EN = {"知识问答": "knowledge_qa", "业务处理": "business",
             "知识管理": "knowledge_mgmt", "工单处理": "escalate"}
_LABEL_ID = {"知识问答": 0, "业务处理": 1, "知识管理": 2, "工单处理": 3}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
random.seed(42)


def load_file(path: str) -> list[dict]:
    """加载 JSONL（每行一个JSON对象，末尾可能有逗号）。"""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 去掉行尾逗号（JSON数组元素格式）
            if line.endswith(","):
                line = line[:-1]
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def main():
    # 1. 加载旧数据并映射
    old_raw = load_file(os.path.join(DATA_DIR, "intent_train.json"))
    old_mapped = []
    for d in old_raw:
        new_label = _OLD_10_TO_4.get(d["label"])
        if new_label:
            old_mapped.append({"query": d["query"].strip(), "label": new_label})

    # 2. 加载v4数据
    v4_raw = load_file(os.path.join(DATA_DIR, "intent_train_v4.json"))
    v4_mapped = [{"query": d["query"].strip(), "label": d["label"]} for d in v4_raw]

    # 3. 合并去重
    seen = set()
    all_data = []
    for d in old_mapped + v4_mapped:
        q = d["query"]
        if q and q not in seen:
            seen.add(q)
            all_data.append(d)

    # 4. 统计
    from collections import Counter
    counts = Counter(d["label"] for d in all_data)
    print(f"合并后总数据: {len(all_data)} 条")
    for label in ["知识问答", "业务处理", "知识管理", "工单处理"]:
        print(f"  {label} ({_LABEL_EN[label]}): {counts.get(label, 0)}")

    # 5. LLM 扩充少数类（知识管理 < 200条时扩充到 ~400条）
    target_min = 300
    for label in ["知识管理", "工单处理"]:
        current = counts.get(label, 0)
        if current < target_min:
            need = target_min - current
            print(f"\nLLM扩充 '{label}': 当前{current}条, 需要+{need}条...")
            augmented = augment_with_llm(label, all_data, need)
            all_data.extend(augmented)
            print(f"  扩充完成: +{len(augmented)}条")

    # 重新统计
    counts = Counter(d["label"] for d in all_data)
    print(f"\n最终数据: {len(all_data)} 条")
    for label in ["知识问答", "业务处理", "知识管理", "工单处理"]:
        print(f"  {label}: {counts.get(label, 0)}")

    # 6. 打乱+拆分
    random.shuffle(all_data)
    n = len(all_data)
    train_end = int(n * 0.8)
    dev_end = int(n * 0.9)
    splits = {"train": all_data[:train_end], "dev": all_data[train_end:dev_end],
              "test": all_data[dev_end:]}

    # 7. 写CSV
    csv_path = os.path.join(DATA_DIR, "intent_4class_train.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("sentence,label,label_name,dataset\n")
        for ds_name, items in splits.items():
            for d in items:
                text = d["query"].replace(",", "，").replace("\n", " ")
                lid = _LABEL_ID[d["label"]]
                en = _LABEL_EN[d["label"]]
                f.write(f"{text},{lid},{en},{ds_name}\n")

    print(f"\n保存: {csv_path}")
    for ds_name in ["train", "dev", "test"]:
        print(f"  {ds_name}: {len(splits[ds_name])} 条")


def augment_with_llm(label: str, existing: list[dict], need: int) -> list[dict]:
    """用LLM生成指定标签的云答客服query。"""
    from backend.core.llm_factory import get_llm

    examples = [d["query"] for d in existing if d["label"] == label]
    sample_examples = random.sample(examples, min(5, len(examples)))

    prompt = f"""你是一个云答客服意图识别数据生成器。
请生成 {need} 条属于"{label}"类别的用户消息。

类别说明：
- 知识问答: 商品咨询、价格询问、求推荐、闲聊
- 业务处理: 查物流、改订单、退款退货、售后操作
- 知识管理: 商户录入/更新/删除知识、政策规则管理
- 工单处理: 投诉、严重不满、要求转人工

参考示例:
{chr(10).join(f'  - {e}' for e in sample_examples)}

请直接输出JSON数组，每个元素: {{"query":"用户消息"}}
只输出JSON数组，不要其他内容。"""

    llm = get_llm(temperature=0.8)
    try:
        import asyncio
        resp = asyncio.get_event_loop().run_until_complete(
            llm.ainvoke(prompt)
        )
        text = resp.content if hasattr(resp, 'content') else str(resp)
        # 提取JSON数组
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            items = json.loads(match.group())
            return [{"query": item["query"].strip(), "label": label}
                    for item in items if item.get("query", "").strip()]
    except Exception as e:
        print(f"  LLM扩充失败: {e}")

    return []


if __name__ == "__main__":
    main()
