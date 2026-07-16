# backend/training/train_intent.py
# 意图识别模型训练脚本（BERT 多分类，4 类）🆕 v4.0
#
# 数据格式：JSONL，每行 {"query": "用户消息", "label": "知识问答|业务处理|知识管理|工单处理"}
# 训练命令：
#   cd EcomSentiment_agent
#   python -m backend.training.train_intent --data ./data/intent_train.json

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from backend.training.config import (
    BERT_BASE_PATH, INTENT_MODEL_DIR, INTENT_LABELS,
    BATCH_SIZE, MAX_LENGTH, EPOCHS, LEARNING_RATE, WARMUP_STEPS, WEIGHT_DECAY, DEVICE,
)
from backend.core.logger import get_logger

logger = get_logger(__name__)


class IntentDataset(Dataset):
    """意图分类数据集。"""

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


def load_data(data_path: str) -> tuple[list[str], list[int]]:
    """从 JSONL 加载训练数据。

    每行格式: {"query": "用户消息", "label": "意图类别"}
    """
    texts, labels = [], []
    # 标签别名映射（处理标注不一致的边界情况）
    _LABEL_ALIAS = {"产品": "产品咨询", "商品咨询": "产品咨询", "物流": "查物流", "退货": "售后问题"}
    label2id = {label: i for i, label in enumerate(INTENT_LABELS)}

    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip().rstrip(",")  # 容错：去掉末尾逗号（JSONL不支持尾逗号）
            if not line:
                continue
            try:
                item = json.loads(line)
                query = item.get("query", "").strip()
                label = item.get("label", "").strip()

                if not query or not label:
                    logger.warning(f"跳过第{line_num}行: 缺少query或label")
                    continue

                if label not in label2id:
                    # 尝试别名映射
                    label = _LABEL_ALIAS.get(label, label)
                if label not in label2id:
                    logger.warning(f"跳过第{line_num}行: 未知标签 '{label}'，有效标签: {INTENT_LABELS}")
                    continue

                texts.append(query)
                labels.append(label2id[label])

            except json.JSONDecodeError as e:
                logger.warning(f"跳过第{line_num}行: JSON解析失败 - {e}")

    logger.info(f"从 {data_path} 加载了 {len(texts)} 条数据")
    return texts, labels


def compute_metrics(eval_pred) -> dict:
    """计算评估指标。"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def train(data_path: str):
    """训练意图识别模型。"""
    logger.info("开始训练意图识别模型")

    # 1. 加载数据
    texts, labels = load_data(data_path)
    if len(texts) < 100:
        raise ValueError(f"数据量太少 ({len(texts)}条)，至少需要100条")

    # 统计每类数量
    label_counts = {}
    for lb in labels:
        label_counts[INTENT_LABELS[lb]] = label_counts.get(INTENT_LABELS[lb], 0) + 1
    logger.info("类别分布:", distribution=label_counts)

    # 2. 拆分训练/验证/测试 (70/15/15)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    logger.info(f"拆分: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")

    # 3. 加载分词器和模型
    logger.info(f"加载基座模型: {BERT_BASE_PATH}")
    tokenizer = BertTokenizer.from_pretrained(BERT_BASE_PATH)
    model = BertForSequenceClassification.from_pretrained(
        BERT_BASE_PATH,
        num_labels=len(INTENT_LABELS),
    )
    model.to(DEVICE)

    # 4. 构建数据集
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer)
    test_dataset = IntentDataset(test_texts, test_labels, tokenizer)

    # 5. 训练
    training_args = TrainingArguments(
        output_dir=os.path.join(INTENT_MODEL_DIR, "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_dir=os.path.join(INTENT_MODEL_DIR, "logs"),
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
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("开始训练...")
    trainer.train()

    # 6. 评估测试集
    logger.info("测试集评估...")
    test_result = trainer.evaluate(test_dataset)
    logger.info(f"测试集结果: acc={test_result.get('eval_accuracy', 0):.4f}, f1={test_result.get('eval_macro_f1', 0):.4f}")

    # 详细分类报告
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    report = classification_report(test_labels, pred_labels, target_names=INTENT_LABELS)
    print("\n" + "=" * 60)
    print("意图分类 - 测试集评估报告")
    print("=" * 60)
    print(report)
    print("混淆矩阵:")
    print(confusion_matrix(test_labels, pred_labels))

    # 7. 保存模型
    os.makedirs(INTENT_MODEL_DIR, exist_ok=True)
    model.save_pretrained(INTENT_MODEL_DIR)
    tokenizer.save_pretrained(INTENT_MODEL_DIR)

    # 保存标签映射
    with open(os.path.join(INTENT_MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({str(i): label for i, label in enumerate(INTENT_LABELS)}, f, ensure_ascii=False, indent=2)

    logger.info(f"模型已保存到: {INTENT_MODEL_DIR}")
    logger.info("训练完成!")

    return test_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练意图识别模型")
    parser.add_argument("--data", type=str, default=r"D:\电商项目文件\EcomSentiment_agent\data\intent_train_v4.json", help="JSONL训练数据路径")
    args = parser.parse_args()

    train(args.data)
