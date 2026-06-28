# -*- coding: utf-8 -*-
"""
text_utils.py —— 中文文本预处理工具

提供统一的 jieba 分词函数，供分类训练与检索共用，
保证训练与推理时的特征处理方式一致。
"""
import re
import jieba

# 简单停用词表（中文常见虚词 + 标点），降低噪声、提升 TF-IDF 区分度
STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "也", "对", "把", "被", "为", "之",
    "其", "中", "等", "即", "及", "并", "而", "则", "就", "都", "这", "那", "有",
    "个", "我", "你", "他", "它", "们", "一个", "一种", "一些", "可以", "通过",
    "使得", "使", "能", "会", "上", "下", "时", "如", "若", "用", "做", "什么",
    "怎么", "如何", "为什么", "请问", "请", "吗", "呢", "啊", "吧",
}


def clean_text(text: str) -> str:
    """去除多余空白和特殊符号，保留中英文、数字。"""
    if not isinstance(text, str):
        text = str(text)
    # 保留中文、英文字母、数字、空格
    text = re.sub(r"[^一-龥a-zA-Z0-9\s]", " ", text)
    return text.strip()


def tokenize(text: str):
    """对中文文本分词并过滤停用词，返回词列表（供 TfidfVectorizer 使用）。"""
    text = clean_text(text)
    words = jieba.lcut(text)
    return [w for w in words if w.strip() and w not in STOPWORDS and len(w.strip()) > 0]


def tokenize_join(text: str) -> str:
    """分词后用空格拼接，方便交给以空格切分的向量器。"""
    return " ".join(tokenize(text))
