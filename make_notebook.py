# -*- coding: utf-8 -*-
"""生成提交用 notebook：notebook/机器学习答疑助手.ipynb"""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "notebook", "机器学习答疑助手.ipynb")

nb = nbf.v4.new_notebook()
cells = []

def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# 机器学习专升本知识点智能答疑助手 —— 完整流程演示

《机器学习基础及应用》期末大作业 · 方向3：教育服务类（进阶版）

本 Notebook 串联：知识库构建（213 个知识点，6 大类别）→ 数据预处理 → 3 种算法训练评估 → 两级检索 → RAG 文档问答演示。""")

md("## 0. 环境与路径准备")
code("""import os, sys, json
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.getcwd(), "..", "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import pandas as pd, numpy as np
print("环境就绪")""")

md("""## 1. 知识库构建与数据预处理

调用 `build_kb.py` 生成 213 个机器学习知识点，并完成缺失值/重复值/类别校验。""")
code("""import build_kb
df = build_kb.build_dataframe()
print("知识点总数：", len(df))
print("缺失值数量：", int(df.isnull().sum().sum()))
print("重复标题数：", int(df.duplicated(subset=['title']).sum()))
df['category'].value_counts()""")

code("""# 查看知识库样例
df.head(3)""")

md("""## 2. 特征工程：jieba 中文分词 + TF-IDF

中文需先分词。这里展示分词效果。""")
code("""from text_utils import tokenize
sample = "朴素贝叶斯是基于贝叶斯定理的分类算法，假设特征条件独立。"
print("分词结果：", tokenize(sample))""")

md("""## 3. 三种分类算法训练、调优与多指标评估

实现**朴素贝叶斯、决策树、随机森林**，用 GridSearchCV 网格搜索调参，
用准确率/精确率/召回率/F1/AUC 多指标评估，绘制混淆矩阵与 ROC。""")
code("""import train_classify as tc
tc.main()   # 训练三模型并保存评估报告与图表""")

code("""# 读取评估报告，对比三算法
with open(os.path.join("..", "models", "eval_report.json"), encoding="utf-8") as f:
    report = json.load(f)
rows = []
for name in ["朴素贝叶斯", "决策树", "随机森林"]:
    if name in report:
        rows.append({"模型": name, **report[name]["metrics"]})
pd.DataFrame(rows).set_index("模型")""")

md("""**结论**：朴素贝叶斯在该文本分类任务上表现最佳（F1、AUC 最高），符合其在高维稀疏文本上的优势，故作为知识点分类与检索的主模型。""")

md("""## 4. 两级检索演示

先用朴素贝叶斯预测问题类别缩小范围，再在 TF-IDF 空间用余弦相似度排序。""")
code("""import retriever
r = retriever.get_retriever()
for q in ["什么是过拟合", "K-means如何选择K值", "ReLU激活函数有什么优点"]:
    out = r.search(q, top_k=2)
    print(f"问题：{q}")
    print(f"  预测类别：{out['predicted_category']}")
    for it in out["results"]:
        print(f"  - [{it['category']}] {it['title']} (相似度 {it['score']})")
    print()""")

md("""## 5. RAG 文档问答演示（进阶①）

把一段文本作为"文档"，演示分块 → 检索 → 返回原文片段。""")
code("""import rag
text = ("机器学习分为监督学习和无监督学习。监督学习需要带标签的数据，可做分类与回归。"
        "无监督学习不需要标签，常用于聚类与降维。过拟合指模型在训练集表现好但测试集差，"
        "可通过正则化、增加数据、Dropout 等方法缓解。")
engine = rag.RAGEngine(text)
ans, hits = engine.answer("如何缓解过拟合？")
print("文档块数：", len(engine.chunks))
print("回答：\\n", ans)""")

md("""## 6. 模拟试卷生成与评分演示（进阶③）""")
code("""import exam
qs = exam.generate_exam(n=3, use_llm=False)   # 离线模板出题
for i, q in enumerate(qs, 1):
    print(f"[{i}] ({q['type']}/{q['category']}) {q['question'][:50]}... 答案={q['answer']}")
score, total, details = exam.grade(qs, [q['answer'] for q in qs])  # 全答对演示
print(f"\\n全部答对得分：{score}/{total}")""")

md("""## 7. 总结

- 知识库：213 个知识点，6 大类别，完成预处理校验。
- 模型：朴素贝叶斯/决策树/随机森林，网格搜索调优，多指标评估，朴素贝叶斯最优。
- 检索：两级检索提升准确率。
- 进阶：RAG 文档问答、模拟试卷生成评分、错题薄弱项分析（见 Streamlit 应用）。

完整智能体运行：`streamlit run app.py`""")

nb["cells"] = cells
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("notebook 已生成：", OUT)
