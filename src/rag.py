# -*- coding: utf-8 -*-
"""
rag.py —— RAG 检索增强生成（进阶①：PDF 文档上传问答）

流程：
  PDF 解析(pdfplumber) -> 按字数分块 -> TF-IDF 向量化（轻量，无需 GPU/embedding 服务）
  -> 提问时余弦相似度检索 Top-K 相关块 -> 返回检索依据

降级策略：
  - pdfplumber 未安装：提示安装，无法解析 PDF（可改用纯文本输入）。

命令行用法：
  python src/rag.py 某文档.pdf "你的问题"
"""
import os
import sys
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from text_utils import tokenize

# pdfplumber 为可选依赖
try:
    import pdfplumber
    _PDF_OK = True
except Exception:
    _PDF_OK = False


def pdf_available() -> bool:
    return _PDF_OK


def extract_pdf_text(pdf_path: str) -> str:
    """解析 PDF 全部文本。"""
    if not _PDF_OK:
        raise RuntimeError("未安装 pdfplumber，无法解析 PDF。请先 pip install pdfplumber")
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                texts.append(t)
    return "\n".join(texts)


def split_chunks(text: str, chunk_size: int = 300, overlap: int = 50):
    """
    把长文本切成带重叠的块。先按段落切，再按字数合并，
    重叠部分有助于跨块语境的检索召回。
    """
    # 先按换行/句号粗分
    paragraphs = re.split(r"[\n。！？!?]", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) <= chunk_size:
            buf += p + "。"
        else:
            if buf:
                chunks.append(buf)
            # 重叠：保留上一块尾部
            tail = buf[-overlap:] if buf else ""
            buf = tail + p + "。"
    if buf.strip():
        chunks.append(buf)
    return [c for c in chunks if len(c.strip()) > 10]


class RAGEngine:
    """对单份文档（或文本）建立 TF-IDF 索引，支持检索增强问答。"""

    def __init__(self, text: str):
        self.chunks = split_chunks(text)
        if not self.chunks:
            raise ValueError("文档内容为空或无法提取有效文本。")
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                          ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    @classmethod
    def from_pdf(cls, pdf_path: str):
        return cls(extract_pdf_text(pdf_path))

    def retrieve(self, question: str, top_k: int = 3):
        """检索与问题最相关的若干文本块。
        对于泛化问题（如"讲了什么"），TF-IDF 相似度可能接近 0，
        此时直接返回相似度最高的 top_k 个块，不做阈值过滤。
        """
        q = self.vectorizer.transform([question])
        sims = cosine_similarity(q, self.matrix)[0]
        idx = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], round(float(sims[i]), 4)) for i in idx]

    def answer(self, question: str, top_k: int = 3):
        """
        检索增强问答：返回拼接的检索原文。
        返回 (answer_text, retrieved_chunks)
        """
        hits = self.retrieve(question, top_k=top_k)
        if not hits:
            return "未在文档中检索到相关内容。", []

        context = "\n\n".join([f"【片段{i+1}】{c}" for i, (c, _) in enumerate(hits)])
        note = "（以下为文档中最相关的原文片段）\n\n"
        return note + context, hits

    def summarize(self, top_k: int = None):
        """对文档做概要性回答：返回前 top_k 个文本块作为文档概览。"""
        if top_k is None:
            top_k = min(3, len(self.chunks))
        # 按块长度排序，优先返回信息量大的块
        ranked = sorted(enumerate(self.chunks), key=lambda x: len(x[1]), reverse=True)
        hits = [(self.chunks[i], 0.0) for i, _ in ranked[:top_k]]
        context = "\n\n".join([f"【片段{i+1}】{c}" for i, (c, _) in enumerate(hits)])
        note = "（以下为文档的核心内容片段）\n\n"
        return note + context, hits


def _cli():
    if len(sys.argv) < 3:
        print('用法：python src/rag.py 文档.pdf "你的问题"')
        return
    pdf_path, question = sys.argv[1], sys.argv[2]
    engine = RAGEngine.from_pdf(pdf_path)
    print(f"文档已切分为 {len(engine.chunks)} 块")
    ans, hits = engine.answer(question)
    print("=" * 50)
    print(f"问题：{question}\n")
    print(ans)


if __name__ == "__main__":
    _cli()
