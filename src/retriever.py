# -*- coding: utf-8 -*-
"""
retriever.py —— 知识点检索（两级检索提升准确率）
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from text_utils import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "knowledge_base.csv")
MODELS_DIR = os.path.join(ROOT, "models")
RETRIEVER_PATH = os.path.join(MODELS_DIR, "tfidf_retriever.pkl")
NB_PATH = os.path.join(MODELS_DIR, "nb_final.pkl")


class KnowledgeRetriever:
    """知识库检索器：封装 TF-IDF 向量空间 + 朴素贝叶斯类别预测。"""

    def __init__(self):
        self.df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
        self.df["doc"] = (self.df["title"] + " " + self.df["title"] + " "
                          + self.df["keywords"] + " " + self.df["content"])
        self._fit_or_load_tfidf()
        self._load_nb()

    def _fit_or_load_tfidf(self):
        """加载缓存的 TF-IDF 向量器；不存在或加载失败则现场拟合并缓存。"""
        loaded = False
        if os.path.exists(RETRIEVER_PATH):
            try:
                obj = joblib.load(RETRIEVER_PATH)
                self.vectorizer = obj["vectorizer"]
                self.doc_matrix = obj["doc_matrix"]
                loaded = True
            except Exception:
                # pkl 版本不兼容或损坏，重新拟合
                pass

        if not loaded:
            self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                              ngram_range=(1, 2), min_df=1)
            self.doc_matrix = self.vectorizer.fit_transform(self.df["doc"])
            os.makedirs(MODELS_DIR, exist_ok=True)
            try:
                joblib.dump({"vectorizer": self.vectorizer, "doc_matrix": self.doc_matrix},
                            RETRIEVER_PATH)
            except Exception:
                pass  # 保存失败不影响运行

    def _load_nb(self):
        """加载朴素贝叶斯分类器；缺失或损坏则置 None。"""
        self.nb = None
        if os.path.exists(NB_PATH):
            try:
                self.nb = joblib.load(NB_PATH)["model"]
            except Exception:
                self.nb = None

    def predict_category(self, query: str):
        if self.nb is None:
            return None
        try:
            return self.nb.predict([query])[0]
        except Exception:
            return None

    def search(self, query: str, top_k: int = 3):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]
        pred_cat = self.predict_category(query)

        order = np.argsort(sims)[::-1]
        in_cat, others = [], []
        for idx in order:
            row = self.df.iloc[idx]
            item = {
                "id": int(row["id"]),
                "category": row["category"],
                "title": row["title"],
                "content": row["content"],
                "keywords": row["keywords"],
                "score": round(float(sims[idx]), 4),
                "in_pred_category": (pred_cat is not None and row["category"] == pred_cat),
            }
            if item["in_pred_category"]:
                in_cat.append(item)
            else:
                others.append(item)

        results = (in_cat + others)[:top_k]
        return {"predicted_category": pred_cat, "results": results}


_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever


def _cli():
    if len(sys.argv) < 2:
        print('用法：python src/retriever.py "你的问题"')
        return
    query = sys.argv[1]
    r = get_retriever()
    out = r.search(query, top_k=3)
    print(f"\n问题：{query}")
    print(f"预测类别：{out['predicted_category']}")
    print("=" * 50)
    for i, item in enumerate(out["results"], 1):
        flag = "★类内" if item["in_pred_category"] else " 全局"
        print(f"[{i}] {flag} | {item['category']} | 相似度={item['score']}")
        print(f"    {item['title']}")
        print(f"    {item['content'][:60]}...")
        print("-" * 50)


if __name__ == "__main__":
    _cli()
