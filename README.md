# 机器学习专升本知识点智能答疑助手（进阶版）

《机器学习基础及应用》课程期末大作业 —— **方向3：教育服务类——专升本知识点智能答疑助手**

一个面向"机器学习"专升本考试的智能答疑 AI 智能体，覆盖**基础版（必做）+ 进阶版（选做）**全部要求。

---

## 一、功能总览

| 模块 | 说明 | 对应作业要求 |
|------|------|------------|
| 知识库构建 | 213 个机器学习核心知识点，6 大类别结构化 | 必做①（≥100 知识点） |
| 知识点分类 | 朴素贝叶斯 + 决策树 + 随机森林，网格搜索调优，多指标评估 | 必做②（NB 分类检索）|
| 智能答疑 | 朴素贝叶斯分类 + 两级 TF-IDF 检索 + 本地知识库回答 | 必做③（自然语言问答）|
| PDF 文档问答 | RAG：PDF 解析→分块→向量检索→检索增强生成 | 进阶①（RAG）|
| 错题/薄弱项 | 错题持久化、薄弱项统计柱状图、复习建议 | 进阶②（错题分析）|
| 模拟考试 | 自动出题→作答→评分→错题入库 | 进阶③（试卷生成评分）|

---

## 二、环境安装

```bash
# 1. 安装依赖（建议使用清华镜像）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 若镜像受限，可逐个安装：
pip install scikit-learn pandas numpy jieba matplotlib streamlit joblib
pip install pdfplumber   # RAG 功能所需
```

> 说明：`pdfplumber`（PDF 解析）为**可选依赖**。未安装时，答疑、PDF 问答和模拟考试仍可基于本地知识库/检索/模板完整演示。

---

## 三、运行步骤

```bash
# 第 1 步：构建知识库（生成 data/knowledge_base.csv）
python src/build_kb.py

# 第 2 步：训练分类模型（生成 models/ 下的 pkl + 混淆矩阵 + ROC + 评估报告）
python src/train_classify.py

# 第 3 步（可选命令行自测）：
python src/retriever.py "什么是过拟合"

# 第 4 步：启动智能体 Web 应用
streamlit run app.py
```

浏览器打开后即可使用五大功能标签页。

---

## 五、项目结构

```
jiqixuexi/
├── data/
│   ├── knowledge_base.csv      # 知识库（213 条）
│   └── mistakes.csv            # 错题本（运行时生成）
├── models/                     # 训练产物（运行 train_classify.py 后生成）
│   ├── nb_classifier.pkl       # 最优模型
│   ├── nb_final.pkl            # 朴素贝叶斯最终模型（检索用）
│   ├── tfidf_retriever.pkl     # 检索向量器缓存
│   ├── eval_report.json        # 多指标评估报告
│   ├── confusion_*.png         # 混淆矩阵
│   └── roc_*.png               # ROC 曲线
├── src/
│   ├── build_kb.py             # 知识库构建与校验
│   ├── text_utils.py           # jieba 中文分词工具
│   ├── train_classify.py       # 3 算法训练+调优+评估
│   ├── retriever.py            # 两级检索
│   ├── rag.py                  # RAG 文档问答
│   ├── exam.py                 # 模拟试卷生成与评分
│   └── mistakes.py             # 错题与薄弱项分析
├── notebook/
│   └── 机器学习答疑助手.ipynb   # 完整流程演示 notebook
├── app.py                      # Streamlit 智能体主程序
├── requirements.txt
├── 数据集使用说明.md
├── 项目文档.md
└── README.md
```

---

## 六、技术栈

- **机器学习**：scikit-learn（朴素贝叶斯 / 决策树 / 随机森林 / TF-IDF / GridSearchCV）
- **中文处理**：jieba 分词
- **检索 / RAG**：TF-IDF + 余弦相似度，pdfplumber 解析 PDF
- **智能体界面**：Streamlit
