# scripts/preprocess_emotion_7class.py
# NLPCC2014 8类情绪 → 系统 7 类情感标签，生成训练/验证/测试集

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 原始标签 → 系统标签映射（数据层映射，训练后模型直接输出系统标签）
LABEL_REMAP = {
    "happiness": "happy",
    "like": "grateful",
    "none": "neutral",
    "surprise": "confused",
    "fear": "anxious",
    "anger": "angry",
    "sadness": "disappointed",
    "disgust": "disappointed",
}

# 目标标签列表
TARGET_LABELS = ["happy", "grateful", "neutral", "confused", "anxious", "angry", "disappointed"]
# 对应的数字标签
LABEL_TO_ID = {label: i for i, label in enumerate(TARGET_LABELS)}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_FILE = os.path.join(DATA_DIR, "Nlpcc2014Train.tsv")
OUTPUT_CSV = os.path.join(DATA_DIR, "sentiment_7class_train.csv")

random.seed(42)


def main():
    # 1. 读取原始数据
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    samples = []
    skipped = 0
    for i, line in enumerate(lines[1:], start=1):  # 跳过 header
        parts = line.split(",", 1)
        if len(parts) != 2:
            skipped += 1
            continue
        label_raw, text = parts[0], parts[1].strip()
        if not text or label_raw not in LABEL_REMAP:
            skipped += 1
            continue
        target_label = LABEL_REMAP[label_raw]
        label_id = LABEL_TO_ID[target_label]
        samples.append((text, target_label, label_id))

    print(f"原始行数: {len(lines) - 1}")
    print(f"有效样本: {len(samples)}, 跳过: {skipped}")

    # 2. 打乱
    random.shuffle(samples)

    # 3. 拆分 train(80%) / dev(10%) / test(10%)
    n = len(samples)
    train_end = int(n * 0.8)
    dev_end = int(n * 0.9)

    splits = {
        "train": samples[:train_end],
        "dev": samples[train_end:dev_end],
        "test": samples[dev_end:],
    }

    # 4. 写入 CSV（兼容现有 train_sentiment.py 格式）
    csv_path = OUTPUT_CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("sentence,label,label_name,dataset\n")
        for dataset_name, data in splits.items():
            for text, label_name, label_id in data:
                # 转义逗号和换行符
                text_escaped = text.replace(",", "，").replace("\n", " ").replace("\r", " ")
                f.write(f"{text_escaped},{label_id},{label_name},{dataset_name}\n")

    print(f"\n数据集已保存: {csv_path}")
    print(f"  train: {len(splits['train'])} 条")
    print(f"  dev:   {len(splits['dev'])} 条")
    print(f"  test:  {len(splits['test'])} 条")
    print(f"\n标签映射: {TARGET_LABELS}")
    print(f"标签ID:   {LABEL_TO_ID}")

    # 5. 保存标签映射（供训练脚本使用）
    import json
    map_path = os.path.join(DATA_DIR, "sentiment_7class_labels.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump({
            "labels": TARGET_LABELS,
            "label_to_id": LABEL_TO_ID,
            "id_to_label": {v: k for k, v in LABEL_TO_ID.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"标签配置已保存: {map_path}")


if __name__ == "__main__":
    main()
