# backend/training/config.py
# 训练模块共享配置

import os
import torch

# ── 基座模型（中文 BERT，含完整 tokenizer + 模型权重）──
# config.py 在 backend/training/ 下，往上2级到项目根，再往上1级到 新建文件夹/
_BERT_RELATIVE = "../../../TMFCode_随堂代码/04-bert/bert-base-chinese"
# 解析为绝对路径，兼容 PyCharm / 终端 / 不同工作目录
BERT_BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), _BERT_RELATIVE))

# ── 输出路径 ──
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
INTENT_MODEL_DIR = os.path.join(OUTPUT_DIR, "intent_classifier")
SENTIMENT_MODEL_DIR = os.path.join(OUTPUT_DIR, "sentiment_classifier")

# ── 训练超参 ──
BATCH_SIZE = 16                    # 加大 batch，梯度更稳定
MAX_LENGTH = 128
EPOCHS = 5                         # 多跑两轮，充分收敛
LEARNING_RATE = 5e-6               # 降低学习率，避免震荡
WARMUP_STEPS = 100                 # 更多预热步数
WEIGHT_DECAY = 0.01

# ── 设备 ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 🆕 v4.0 意图标签（4 类，与 schemas.py IntentCategory 对应）──
INTENT_LABELS = [
    "知识问答",      # → knowledge_qa    (商品咨询/价格/推荐/闲聊/政策)
    "业务处理",      # → business        (查物流/改订单/退款/售后操作)
    "知识管理",      # → knowledge_mgmt  (商户录入/查看/删除知识)
    "工单处理",      # → escalate        (投诉/愤怒情绪/高危转人工)
]

# ── 情感标签（2 类）──
SENTIMENT_LABELS = ["负面", "正面"]
