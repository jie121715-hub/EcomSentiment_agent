# backend/rag/preprocess.py
# 文本预处理工具：jieba分词 + 小写化，用于 BM25 检索。
#
# 从 EcomSentiment_RAG/mysql_qa/utils/preprocess.py 移植。

import jieba


def preprocess_text(text: str) -> str:
    """对文本进行预处理：jieba精确模式分词 + 小写化。

    :param text: 原始文本字符串
    :return: 分词后空格分隔的小写字符串（供 BM25Okapi 使用）
    """
    # 1. jieba精确模式分词
    words = jieba.lcut(text)
    # 2. 转为小写并过滤空字符串
    words = [w.lower().strip() for w in words if w.strip()]
    # 3. 返回空格分隔的字符串
    return " ".join(words)


if __name__ == "__main__":
    test = "我想申请退货退款，请问怎么操作？"
    result = preprocess_text(test)
    print(f"原文: {test}")
    print(f"处理后: {result}")
