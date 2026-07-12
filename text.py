import sys;

sys.path.insert(0, '.')
import csv, random

# 1. 检查训练数据label是不是反的
with open('./data/sentiment_train.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# label=0 和 label=1 各抽3条看看
neg_samples = [r['sentence'] for r in rows if r['label'] == '0'][:5]
pos_samples = [r['sentence'] for r in rows if r['label'] == '1'][:5]
print('label=0 (训练为负面) 示例:')
for s in neg_samples: print(f'  [{s[:40]}]')
print('label=1 (训练为正面) 示例:')
for s in pos_samples: print(f'  [{s[:40]}]')
print(f'\n总分布: 0={sum(1 for r in rows if r["label"] == "0")}, 1={sum(1 for r in rows if r["label"] == "1")}')

# 2. 快速重训（5 epochs, 更小学习率）
from backend.training.config import EPOCHS, LEARNING_RATE

print(f'\n当前配置: EPOCHS={EPOCHS}, LR={LEARNING_RATE}')
print('建议改为: EPOCHS=5, LR=5e-6 重训')