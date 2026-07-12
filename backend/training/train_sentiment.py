# backend/training/train_sentiment.py
# 情感分类模型训练脚本（BERT 二分类）
#
# 数据格式：CSV，列名 sentence(文本), label(0/1), dataset(train/dev/test)
# 训练命令：
#   cd EcomSentiment_agent
#   python -m backend.training.train_sentiment --data ./data/sentiment_train.csv

import csv
import json
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments,
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from backend.training.config import (
    BERT_BASE_PATH, SENTIMENT_MODEL_DIR, SENTIMENT_LABELS,
    BATCH_SIZE, MAX_LENGTH, EPOCHS, LEARNING_RATE, WARMUP_STEPS, WEIGHT_DECAY, DEVICE,
)
from backend.core.logger import get_logger

logger = get_logger(__name__)


class SentimentDataset(Dataset):
    """情感分类数据集。"""

    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_data(data_path: str) -> tuple:
    """从 CSV 加载情感数据。

    CSV 列: sentence(文本), label(0=负面/1=正面), dataset(train/dev/test)

    返回: (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels)
    """
    all_data = {"train": ([], []), "dev": ([], []), "test": ([], [])}

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, 2):
            sentence = row.get("sentence", "").strip()
            label_str = row.get("label", "").strip()
            dataset = row.get("dataset", "train").strip().lower()

            if not sentence or label_str == "":
                continue

            try:
                label = int(label_str)
                if label not in (0, 1):
                    logger.warning(f"跳过第{row_num}行: label必须是0或1，实际是{label}")
                    continue
            except ValueError:
                logger.warning(f"跳过第{row_num}行: label无法转为整数")
                continue

            if dataset not in all_data:
                logger.warning(f"未知dataset值 '{dataset}'，归入train")
                dataset = "train"

            all_data[dataset][0].append(sentence)
            all_data[dataset][1].append(label)

    total = sum(len(v[0]) for v in all_data.values())
    logger.info(
        f"从 {data_path} 加载了 {total} 条数据: "
        f"train={len(all_data['train'][0])}, "
        f"dev={len(all_data['dev'][0])}, "
        f"test={len(all_data['test'][0])}"
    )

    return all_data["train"], all_data["dev"], all_data["test"]


def compute_metrics(eval_pred) -> dict:
    """计算评估指标。"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def train(data_path: str):
    """训练情感分类模型。"""
    logger.info("开始训练情感分类模型")

    # 1. 加载数据
    (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels) = load_data(data_path)

    if len(train_texts) < 100:
        raise ValueError(f"训练数据量太少 ({len(train_texts)}条)，至少需要100条")

    # 统计正负比例
    pos_count = sum(1 for lb in train_labels if lb == 1)
    neg_count = sum(1 for lb in train_labels if lb == 0)
    logger.info(f"训练集: 正面={pos_count}, 负面={neg_count}", ratio=f"1:{neg_count/pos_count:.1f}" if pos_count > 0 else "all_neg")

    # 2. 加载分词器和模型
    logger.info(f"加载基座模型: {BERT_BASE_PATH}")
    tokenizer = BertTokenizer.from_pretrained(BERT_BASE_PATH)
    model = BertForSequenceClassification.from_pretrained(
        BERT_BASE_PATH,
        num_labels=2,
    )
    model.to(DEVICE)

    # 3. 构建数据集
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    dev_dataset = SentimentDataset(dev_texts, dev_labels, tokenizer) if dev_texts else None
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer) if test_texts else None

    # 4. 训练
    training_args = TrainingArguments(
        output_dir=os.path.join(SENTIMENT_MODEL_DIR, "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_dir=os.path.join(SENTIMENT_MODEL_DIR, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="macro_f1",
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("开始训练...")
    trainer.train()

    # 5. 评估
    eval_ds = test_dataset or dev_dataset or train_dataset
    eval_result = trainer.evaluate(eval_ds)
    logger.info(f"评估结果: acc={eval_result.get('eval_accuracy', 0):.4f}, f1={eval_result.get('eval_macro_f1', 0):.4f}")

    predictions = trainer.predict(eval_ds)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    report = classification_report(eval_ds.labels, pred_labels, target_names=SENTIMENT_LABELS)
    print("\n" + "=" * 60)
    print("情感分类 - 评估报告")
    print("=" * 60)
    print(report)
    print("混淆矩阵:")
    print(confusion_matrix(eval_ds.labels, pred_labels))

    # 6. 保存模型
    os.makedirs(SENTIMENT_MODEL_DIR, exist_ok=True)
    model.save_pretrained(SENTIMENT_MODEL_DIR)
    tokenizer.save_pretrained(SENTIMENT_MODEL_DIR)

    with open(os.path.join(SENTIMENT_MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"0": "负面", "1": "正面"}, f, ensure_ascii=False, indent=2)

    logger.info(f"模型已保存到: {SENTIMENT_MODEL_DIR}")
    logger.info("训练完成!")

    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练情感分类模型")
    parser.add_argument("--data", type=str, default=r"C:\Users\23387\Desktop\新建文件夹\EcomSentiment_agent\data\sentiment_train.csv", help="CSV训练数据路径")
    args = parser.parse_args()

    train(args.data)
