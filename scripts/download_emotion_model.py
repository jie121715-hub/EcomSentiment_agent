# scripts/download_emotion_model.py
# 下载 ModelScope StructBERT 情绪7分类模型到项目 models/ 目录

import os
import sys

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelscope.hub.snapshot_download import snapshot_download

MODEL_NAME = "iic/nlp_structbert_emotion-classification_chinese-large"
TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "emotion_7class")

print(f"正在下载 {MODEL_NAME} ...")
print(f"目标目录: {TARGET_DIR}")
os.makedirs(TARGET_DIR, exist_ok=True)

model_dir = snapshot_download(MODEL_NAME, cache_dir=TARGET_DIR)
print(f"\n✅ 模型下载完成: {model_dir}")

# 列出文件
for f in sorted(os.listdir(model_dir)):
    fpath = os.path.join(model_dir, f)
    size = os.path.getsize(fpath) if os.path.isfile(fpath) else "<DIR>"
    print(f"  {f} ({size})")
