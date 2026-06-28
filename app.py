# -*- coding: utf-8 -*-
"""
app.py —— 专升本机器学习知识点智能答疑助手（Streamlit 智能体主程序）

五大功能 Tab：
  1. 智能答疑     —— 朴素贝叶斯分类 + 两级检索
  2. PDF文档问答  —— RAG（上传 PDF -> 检索增强生成）
  3. 模拟考试     —— 自动出题 -> 作答 -> 评分 -> 错题入库
  4. 错题/薄弱项  —— 错题本 + 薄弱项柱状图 + 复习建议
  5. 模型评估看板 —— 三算法多指标对比 + 混淆矩阵 + ROC 曲线

运行：streamlit run app.py
"""
import os
import sys
import json

import pandas as pd
import streamlit as st

# 让 app 能 import src 下的模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import retriever as retr
import mistakes
import exam as exam_mod

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")
KB_PATH = os.path.join(DATA_DIR, "knowledge_base.csv")

st.set_page_config(page_title="机器学习专升本智能答疑助手", page_icon="🎓", layout="wide")


# ---------------- 缓存资源 ----------------
@st.cache_resource
def get_retriever():
    return retr.get_retriever()


@st.cache_resource
def get_rag_engine(text: str):
    import rag
    return rag.RAGEngine(text)


def build_kb_answer(question: str, out: dict) -> str:
    """基于本地知识库检索结果组织答疑内容。"""
    results = out.get("results", [])
    if not results:
        return "未在本地知识库中检索到相关知识点，建议换一种问法或补充课程资料。"

    primary = results[0]
    related = "、".join([it["title"] for it in results[1:]]) if len(results) > 1 else "暂无"
    keywords = primary.get("keywords", "")
    return (
        f'**问题理解**：你问的是"{question}"。\n\n'
        f"**知识点定位**：{primary['category']} / {primary['title']}\n\n"
        f"**核心回答**：{primary['content']}\n\n"
        f"**关键记忆**：{keywords}\n\n"
        f"**相关知识点**：{related}\n\n"
        "以上回答来自本地结构化知识库。"
    )


def build_doc_answer(question: str, hits) -> str:
    """基于RAG检索片段组织文档问答内容，不依赖大模型。"""
    if not hits:
        return "未在文档中检索到足够相关的内容。"

    blocks = [
        f"**问题**：{question}",
        "",
        "**文档依据**：以下内容来自上传文档的相似片段检索结果。",
        "",
    ]
    for i, (chunk, score) in enumerate(hits, 1):
        snippet = chunk[:360] + ("..." if len(chunk) > 360 else "")
        blocks.append(f"**片段{i}（相似度 {score}）**：{snippet}")
        blocks.append("")
    blocks.append("当前为文档检索回答模式。")
    return "\n".join(blocks)


# ---------------- 顶部状态 ----------------
st.title("🎓 机器学习专升本知识点智能答疑助手")
st.caption("课程期末大作业 · 方向3：教育服务类（进阶版） | 知识库分类(朴素贝叶斯) + 两级检索 + RAG")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["💬 智能答疑", "📄 PDF文档问答(RAG)", "📝 模拟考试", "📚 错题/薄弱项", "📊 模型评估看板"])


# ============================================================
# Tab 1：智能答疑
# ============================================================
with tab1:
    st.subheader("智能答疑")
    st.write("输入机器学习相关问题，系统会自动判别知识类别、检索最相关知识点，并基于本地知识库组织解答。")
    q = st.text_input("请输入你的问题：", placeholder="例如：什么是过拟合？怎么解决？", key="qa_q")
    top_k = st.slider("检索知识点数量", 1, 5, 3, key="qa_k")
    if st.button("提问", key="qa_btn") and q.strip():
        try:
            r = get_retriever()
            out = r.search(q, top_k=top_k)
        except Exception as e:
            st.error(f"知识库加载失败：{e}\n\n请确认 data/knowledge_base.csv 和 models/ 目录存在。")
            st.stop()

        st.markdown(f"**🏷️ 预测知识类别：** `{out['predicted_category']}`")

        local_ans = build_kb_answer(q, out)

        st.markdown("### 🤖 智能解答")
        st.markdown(local_ans)

        st.markdown("### 📖 命中的知识点")
        for it in out["results"]:
            flag = "★ 类内命中" if it["in_pred_category"] else "全局命中"
            with st.expander(f"【{it['category']}】{it['title']} （相似度 {it['score']} | {flag}）"):
                st.write(it["content"])
                st.caption(f"关键词：{it['keywords']}")


# ============================================================
# Tab 2：PDF 文档问答（RAG）
# ============================================================
with tab2:
    st.subheader("PDF 文档问答（RAG 检索增强生成）")
    import rag as rag_mod
    if not rag_mod.pdf_available():
        st.warning("未安装 pdfplumber，暂不能解析 PDF。可先 `pip install pdfplumber`；"
                   "或在下方文本框直接粘贴文本进行问答演示。")

    mode = st.radio("文档来源", ["上传 PDF", "粘贴文本"], horizontal=True, key="rag_mode")
    doc_text = None

    if mode == "上传 PDF":
        up = st.file_uploader("上传 PDF 文档", type=["pdf"], key="rag_pdf")
        if up is not None and rag_mod.pdf_available():
            tmp_path = os.path.join(DATA_DIR, "_uploaded.pdf")
            with open(tmp_path, "wb") as f:
                f.write(up.read())
            try:
                doc_text = rag_mod.extract_pdf_text(tmp_path)
                st.success(f"已解析 PDF，提取文本 {len(doc_text)} 字。")
            except Exception as e:
                st.error(f"解析失败：{e}")
    else:
        doc_text = st.text_area("粘贴文档文本", height=180,
                                placeholder="把要问答的资料粘贴到这里…", key="rag_text")

    rag_q = st.text_input("针对文档提问：", placeholder="例如：这篇文档讲了什么？", key="rag_q")
    if st.button("文档问答", key="rag_btn") and doc_text and rag_q.strip():
        try:
            engine = get_rag_engine(doc_text)
            st.caption(f"文档已切分为 {len(engine.chunks)} 个文本块")
            with st.spinner("文档检索中…"):
                hits = engine.retrieve(rag_q, top_k=3)
                ans = build_doc_answer(rag_q, hits)
            st.markdown("### 🤖 回答")
            st.markdown(ans)
            with st.expander("查看检索到的原文片段"):
                for i, (c, s) in enumerate(hits, 1):
                    st.markdown(f"**片段{i}（相似度 {s}）**")
                    st.write(c)
        except Exception as e:
            st.error(f"出错：{e}")


# ============================================================
# Tab 3：模拟考试
# ============================================================
with tab3:
    st.subheader("模拟考试")
    st.write("系统自动从知识库生成试卷。作答后自动评分，错题自动进入错题本。")

    c1, c2 = st.columns(2)
    with c1:
        n_q = st.number_input("题目数量", 1, 10, 5, key="ex_n")
    with c2:
        seed = st.number_input("随机种子（换一套题改这里）", 0, 9999, 42, key="ex_seed")

    if st.button("生成试卷", key="ex_gen"):
        with st.spinner("出题中…"):
            try:
                st.session_state["exam_qs"] = exam_mod.generate_exam(
                    int(n_q), seed=int(seed))
                if not st.session_state["exam_qs"]:
                    st.warning("出题结果为空，请检查 data/knowledge_base.csv 是否存在且有效。")
            except Exception as e:
                st.error(f"出题失败：{e}")
                st.session_state.pop("exam_qs", None)
        st.session_state.pop("exam_graded", None)

    if "exam_qs" in st.session_state and st.session_state["exam_qs"]:
        qs = st.session_state["exam_qs"]
        st.markdown("### 答题区")
        user_answers = []
        for i, qd in enumerate(qs):
            st.markdown(f"**第{i+1}题（{qd['type']} · {qd['category']}）**")
            st.write(qd["question"])
            ans = st.radio(f"选择第{i+1}题答案", qd["options"], key=f"ex_ans_{i}",
                           index=None)
            user_answers.append(ans if ans is not None else "")
            st.divider()

        if st.button("提交评分", key="ex_submit"):
            try:
                score, total, details = exam_mod.grade(qs, user_answers)
                st.session_state["exam_graded"] = (score, total, details)
            except Exception as e:
                st.error(f"评分失败：{e}")

    if "exam_graded" in st.session_state:
        score, total, details = st.session_state["exam_graded"]
        # 防止 total=0 除零
        pct_str = f"（{round(score / total * 100)} 分）" if total > 0 else ""
        st.markdown(f"## 📋 成绩：{score} / {total}{pct_str}")
        for i, d in enumerate(details, 1):
            mark = "✅" if d["is_right"] else "❌"
            st.markdown(f"{mark} **第{i}题** 你的答案：`{d['your_answer']}` | 正确答案：`{d['correct_answer']}`")
            with st.expander("查看解析"):
                st.write(d["explain"])
        if any(not d["is_right"] for d in details):
            st.info("错题已自动加入『错题/薄弱项』标签页，可前往查看薄弱项分析。")


# ============================================================
# Tab 4：错题 / 薄弱项分析
# ============================================================
with tab4:
    st.subheader("错题本与薄弱项分析")
    df_m = mistakes.load_mistakes()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"**累计错题：{len(df_m)} 道**")
        if st.button("清空错题本", key="clr"):
            mistakes.clear_mistakes()
            st.rerun()
    with col2:
        st.write(mistakes.advice_text())

    if len(df_m) > 0:
        st.markdown("### 薄弱项分布")
        try:
            img = mistakes.plot_weak_points()
            st.image(img, use_column_width=True)
        except Exception as e:
            st.warning(f"图表生成失败：{e}")
        st.markdown("### 错题明细")
        st.dataframe(df_m)
    else:
        st.caption("暂无错题。去『模拟考试』练几道题吧。")


# ============================================================
# Tab 5：模型评估看板
# ============================================================
with tab5:
    st.subheader("机器学习模型评估看板")
    report_path = os.path.join(MODELS_DIR, "eval_report.json")
    if not os.path.exists(report_path):
        st.warning("尚未训练模型。请先运行：`python src/train_classify.py`")
    else:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        st.markdown(f"**最优模型：** `{report.get('_best_model','-')}`（按测试集 F1-macro 选取）")

        # 多指标对比表
        rows = []
        for name in ["朴素贝叶斯", "决策树", "随机森林"]:
            if name in report:
                m = report[name]["metrics"]
                rows.append({"模型": name, **m})
        if rows:
            st.markdown("### 三种算法多指标对比")
            st.dataframe(pd.DataFrame(rows).set_index("模型"))

        st.markdown("### 混淆矩阵 / ROC 曲线")
        for name in ["朴素贝叶斯", "决策树", "随机森林"]:
            if name not in report:
                continue
            st.markdown(f"#### {name}")
            cc1, cc2 = st.columns(2)
            cm = os.path.join(MODELS_DIR, report[name]["confusion_png"])
            roc = os.path.join(MODELS_DIR, report[name]["roc_png"])
            if os.path.exists(cm):
                cc1.image(cm, caption="混淆矩阵", use_column_width=True)
            if os.path.exists(roc):
                cc2.image(roc, caption="ROC 曲线", use_column_width=True)


st.markdown("---")
st.caption("《机器学习基础及应用》期末大作业 · 方向3 教育服务类 · 进阶版 | 本地 Streamlit 智能体")