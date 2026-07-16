# backend/training/train_binary_intent.py
# 二分类意图识别模型训练（Qwen2.5-0.5B + QLoRA）
#
# 二分类定义：
#   Class 0 — 信息咨询 (info):  商品/政策/知识问答/闲聊 → RAG 检索
#   Class 1 — 业务办理 (action): 查订单/物流/退货退款/投诉 → Business API
#
# 训练命令：
#   cd EcomSentiment_agent
#   python -m backend.training.train_binary_intent
#
# 首次使用先运行预处理：
#   python scripts/preprocess_binary_intent.py

import json
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from backend.core.logger import get_logger

logger = get_logger(__name__)

# ── 路径配置 ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "qwen2.5-0.5b-instruct")
DATA_PATH = os.path.join(BASE_DIR, "data", "intent_binary_train.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "saved_models", "intent_binary")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# ── 标签 ──
LABELS = ["info", "action"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ── 超参 ──
MAX_LENGTH = 256
BATCH_SIZE = 8          # QLoRA 可用较小 batch
GRAD_ACCUM = 4          # 等效 batch=32
EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01


# ═══════════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════════

def load_data(data_path: str) -> tuple[list[str], list[int]]:
    """加载二分类 JSONL 数据。"""
    texts, labels = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            query = item.get("query", "").strip()
            label = item.get("label", "").strip()
            if query and label in LABEL2ID:
                texts.append(query)
                labels.append(LABEL2ID[label])
    logger.info(f"加载 {len(texts)} 条数据")
    return texts, labels


class BinaryIntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════
# 指标
# ═══════════════════════════════════════════════════════════════

def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


# ═══════════════════════════════════════════════════════════════
# 训练主函数
# ═══════════════════════════════════════════════════════════════

def train(data_path: str = DATA_PATH):
    logger.info("=" * 50)
    logger.info("二分类意图识别训练 — Qwen2.5-0.5B + QLoRA")
    logger.info("=" * 50)

    # 1. 加载数据
    texts, labels = load_data(data_path)
    if len(texts) < 100:
        raise ValueError(f"数据量太少 ({len(texts)}条)，至少需要100条")

    # 统计
    from collections import Counter
    dist = Counter(ID2LABEL[lb] for lb in labels)
    logger.info("类别分布", info=dist.get("info", 0), action=dist.get("action", 0))

    # 2. 拆分 (70/15/15)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    logger.info(f"拆分: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")

    # 3. 加载模型 (4-bit QLoRA)
    logger.info(f"加载模型: {MODEL_PATH}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=len(LABELS),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # 4-bit 训练准备
    model = prepare_model_for_kbit_training(model)

    # LoRA 配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. 构建数据集
    train_dataset = BinaryIntentDataset(train_texts, train_labels, tokenizer)
    val_dataset = BinaryIntentDataset(val_texts, val_labels, tokenizer)
    test_dataset = BinaryIntentDataset(test_texts, test_labels, tokenizer)

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="macro_f1",
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 6. 训练
    logger.info("开始训练...")
    trainer.train()

    # 7. 评估测试集
    logger.info("测试集评估...")
    test_result = trainer.evaluate(test_dataset)
    logger.info(f"测试集: acc={test_result.get('eval_accuracy', 0):.4f}, f1={test_result.get('eval_macro_f1', 0):.4f}")

    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    print("\n" + "=" * 60)
    print("二分类意图识别 — 测试集评估报告")
    print("=" * 60)
    print(classification_report(test_labels, pred_labels, target_names=LABELS))
    print("混淆矩阵:")
    print(confusion_matrix(test_labels, pred_labels))

    # 8. 保存模型
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 保存标签映射
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(ID2LABEL, f, ensure_ascii=False, indent=2)

    logger.info(f"模型已保存到: {OUTPUT_DIR}")
    logger.info("训练完成!")

    return test_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练二分类意图识别模型")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="JSONL训练数据路径")
    args = parser.parse_args()
    train(args.data)
