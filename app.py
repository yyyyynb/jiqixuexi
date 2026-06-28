# -*- coding: utf-8 -*-
"""
app.py —— 专升本机器学习知识点智能答疑助手（Streamlit 智能体主程序）

侧边栏导航：5 大功能按钮
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

# ---------------- 自定义样式 ----------------
st.markdown("""
<style>
/* 引入思源黑体 / 系统中文字体 */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;900&display=swap');

/* 全局字体 */
html, body, [class*="css"] {
    font-family: 'Noto Sans SC', 'Microsoft YaHei', '微软雅黑', sans-serif;
}

/* ============ 全局背景：淡蓝紫渐变 ============ */
.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 95% 100%, rgba(14, 165, 233, 0.08) 0%, transparent 40%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%) !important;
    background-attachment: fixed !important;
}
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}
.main {
    background: transparent !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}

/* ============ 输入框：边框 2px + 加深色 ============ */
[data-testid="stTextInput"] [data-baseweb="base-input"],
[data-testid="stNumberInput"] [data-baseweb="base-input"],
[data-testid="stTextArea"] [data-baseweb="base-input"],
[data-baseweb="base-input"] {
    border: 2px solid #334155 !important;
    border-radius: 8px !important;
    transition: border-color 0.15s, box-shadow 0.15s;
}
[data-testid="stTextInput"] [data-baseweb="base-input"]:focus-within,
[data-testid="stNumberInput"] [data-baseweb="base-input"]:focus-within,
[data-testid="stTextArea"] [data-baseweb="base-input"]:focus-within,
[data-baseweb="base-input"]:focus-within {
    border-color: #1e3a8a !important;
    box-shadow: 0 0 0 1px #1e3a8a !important;
}
[data-testid="stTextInput"] [data-baseweb="base-input"]:hover,
[data-testid="stNumberInput"] [data-baseweb="base-input"]:hover,
[data-testid="stTextArea"] [data-baseweb="base-input"]:hover {
    border-color: #1e3a8a !important;
}

/* ============ 数字框 +/- 按钮加深 ============ */
[data-testid="stNumberInput"] [data-baseweb="base-input"] button {
    background: linear-gradient(135deg, #475569 0%, #1e3a8a 100%) !important;
    color: #ffffff !important;
    border: 1px solid #1e3a8a !important;
    border-radius: 4px !important;
}
[data-testid="stNumberInput"] [data-baseweb="base-input"] button:hover {
    background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%) !important;
}
[data-testid="stNumberInput"] [data-baseweb="base-input"] button svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

/* 主标题居中 + 渐变色 */
h1 {
    text-align: center !important;
    font-weight: 900 !important;
    font-size: 2.4em !important;
    background: linear-gradient(120deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding: 0.3em 0 0.1em 0;
    letter-spacing: 2px;
}

/* 副标题（caption）居中 + 浅灰 */
.stApp [data-testid="stCaptionContainer"] {
    text-align: center !important;
    color: #64748b !important;
    font-size: 0.95em !important;
    margin-bottom: 1.5em !important;
}

/* ============ 侧边栏样式 ============ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%) !important;
    border-right: 1px solid #e2e8f0;
    max-width: 22rem !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 0.5rem !important;
}

/* 侧边栏标题 */
[data-testid="stSidebar"] h1 {
    font-size: 1.25em !important;
    text-align: left !important;
    background: linear-gradient(120deg, #1e3a8a 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding: 0.2em 0 !important;
    letter-spacing: 1px;
    margin-bottom: 0.2em !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    text-align: left !important;
    color: #64748b !important;
    font-size: 0.8em !important;
    margin-bottom: 1.2em !important;
}

/* 侧边栏导航按钮 */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    background: white !important;
    color: #334155 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 0.65rem 0.9rem !important;
    margin: 0.25rem 0 !important;
    font-weight: 500 !important;
    font-size: 0.95em !important;
    transition: all 0.2s ease;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #f1f5f9 !important;
    border-color: #3b82f6 !important;
    color: #1e3a8a !important;
    transform: translateX(2px);
}

/* 当前选中按钮（用 primary 渲染） */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    transform: none !important;
    color: white !important;
}

/* 侧边栏分割线 */
[data-testid="stSidebar"] hr {
    margin: 0.8rem 0 !important;
    border-color: #e2e8f0 !important;
}

/* 二级标题（subheader）加左侧色块 */
h2, h3 {
    border-left: 4px solid #3b82f6;
    padding-left: 0.6em !important;
    color: #1e293b !important;
}

/* 功能区块卡片化 */
.block-container .element-container:has(> [data-testid="stVerticalBlockBorderWrapper"]) {
    border-radius: 12px;
}

/* 通用 expander 美化 */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #fafbfc !important;
}

/* 按钮主色 */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
    transform: translateY(-1px);
}

/* 数据框圆角 */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}

/* Radio选项美化 */
.stRadio [role="radiogroup"] label {
    padding: 0.4em 0.8em !important;
    border-radius: 6px !important;
    transition: background 0.15s;
}
.stRadio [role="radiogroup"] label:hover {
    background: #f1f5f9;
}

/* 分割线 */
hr {
    border-color: #e2e8f0 !important;
}

/* ============ 侧边栏错题统计卡片 ============ */
.sidebar-stat {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.9rem 0.5rem;
    margin: 0.5rem 0;
    text-align: center;
    box-shadow: 0 2px 6px rgba(30, 58, 138, 0.06);
}
.sidebar-stat .num {
    font-size: 2.2em;
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(135deg, #1e3a8a 0%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-stat .label {
    color: #64748b;
    font-size: 0.85em;
    margin-top: 0.25rem;
    letter-spacing: 1px;
}

/* ============ 数字输入框（检索数量） ============ */
[data-testid="stNumberInput"] input {
    text-align: center !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    color: #1e3a8a !important;
}
[data-testid="stNumberInput"] button {
    color: #1e3a8a !important;
}

/* ============ 主区 + / - 按钮美化（用 button key 前缀定位） ============
   Streamlit 没有简单 class hook，给 +/-/提问按钮包一层 .qa-row 容器即可。
   但当前按钮已是 st.button(use_container_width=True)，用相邻选择器粗略美化。 */
[data-testid="stMainBlockContainer"] button[kind="secondary"] {
    background: white !important;
    color: #1e3a8a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    transition: all 0.15s;
}
[data-testid="stMainBlockContainer"] button[kind="secondary"]:hover {
    background: #eff6ff !important;
    border-color: #3b82f6 !important;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ---------------- 汉化 st.dataframe 菜单 ----------------
# 注：glide-data-grid 菜单文字为内置英文，这里用 JS 注入进行实时翻译
st.components.v1.html("""
<script>
(function() {
    const I18N = {
        'Sort ascending': '升序排序',
        'Sort descending': '降序排序',
        'Autosize': '自适应列宽',
        'Autosi': '自适应列宽',
        'Pin column': '固定列',
        'Hide column': '隐藏列',
        'Pin left': '固定到左侧',
        'Pin right': '固定到右侧',
        'Unpin': '取消固定',
        'No columns': '暂无列',
        'Add column': '添加列',
        'Move left': '左移',
        'Move right': '右移',
    };
    function i18n(root) {
        if (!root) return;
        try {
            const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
            let n;
            while ((n = walker.nextNode())) {
                const t = n.textContent.trim();
                if (I18N[t]) {
                    n.textContent = I18N[t];
                }
            }
        } catch (e) {}
    }
    i18n(document.body);
    setTimeout(() => i18n(document.body), 300);
    setTimeout(() => i18n(document.body), 1000);
    setTimeout(() => i18n(document.body), 2500);
    new MutationObserver(muts => {
        muts.forEach(m => {
            m.addedNodes.forEach(n => {
                if (n.nodeType === 1) i18n(n);
                if (n.nodeType === 11) i18n(n);
            });
        });
    }).observe(document.body, { childList: true, subtree: true });
})();
</script>
""", height=0)


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


# ---------------- 侧边栏导航 ----------------
PAGES = [
    ("qa",   "💬 智能答疑",        "基于朴素贝叶斯的两级检索"),
    ("rag",  "📄 PDF文档问答",     "RAG 检索增强生成"),
    ("exam", "📝 模拟考试",        "自动出题 / 评分 / 错题入库"),
    ("err",  "📚 错题/薄弱项",     "错题本 + 薄弱项分析"),
    ("eval", "📊 模型评估看板",    "三算法多指标对比"),
]

# 当前页面状态
if "current_page" not in st.session_state:
    st.session_state.current_page = "qa"

with st.sidebar:
    # 标题（精简，让出顶部空间给功能导航）
    st.markdown(
        '<div style="text-align:center; font-size:1.05em; font-weight:800; '
        'background:linear-gradient(120deg,#1e3a8a 0%,#3b82f6 100%); '
        '-webkit-background-clip:text; -webkit-text-fill-color:transparent; '
        'background-clip:text; padding:0.1em 0 0.2em 0;">'
        '🎓 智能答疑助手</div>',
        unsafe_allow_html=True,
    )

    # ============ 分隔线 ============
    st.markdown("---")
    # 切换功能 caption 
    st.markdown("**五大功能模块**（点击切换）")

    # 功能导航（"五大功能模块"部分）
    for key, label, hint in PAGES:
        is_active = st.session_state.current_page == key
        if st.button(
            f"{label}",
            key=f"nav_{key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.current_page = key
            st.rerun()
        if is_active:
            st.caption(f"　　↳ {hint}")

    # ============ 分隔线 ============
    st.markdown("---")

    # ============ 错题数量统计卡片 ============
    st.markdown("**📊 错题统计**")
    try:
        _df_m = mistakes.load_mistakes()
        _n = len(_df_m)
    except Exception:
        _n = 0
    st.markdown(
        f'<div class="sidebar-stat">'
        f'<div class="num">{_n}</div>'
        f'<div class="label">累计错题（道）</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------- 顶部状态（带可视化背景框装饰 - 浅色系） ----------------
st.markdown(
    """
    <div style="
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #dbeafe 100%);
        padding: 1.6rem 2rem;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 6px 22px rgba(30, 58, 138, 0.10);
        margin: 0.4rem 0 1.6rem 0;
        border: 1px solid rgba(59, 130, 246, 0.20);
    ">
        <div style="
            position: absolute;
            top: -55%; right: -8%;
            width: 240px; height: 240px;
            background: rgba(59, 130, 246, 0.10);
            border-radius: 50%;
            pointer-events: none;
        "></div>
        <div style="
            position: absolute;
            bottom: -60%; left: -6%;
            width: 180px; height: 180px;
            background: rgba(14, 165, 233, 0.08);
            border-radius: 50%;
            pointer-events: none;
        "></div>
        <div style="
            position: absolute;
            top: 30%; right: 18%;
            width: 56px; height: 56px;
            background: rgba(59, 130, 246, 0.12);
            border-radius: 50%;
            pointer-events: none;
        "></div>
        <h1 style="
            position: relative;
            z-index: 1;
            margin: 0;
            background: linear-gradient(120deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 1.95em;
            font-weight: 900;
            letter-spacing: 3px;
        ">🎓 机器学习专升本知识点智能答疑助手</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 页面 1：智能答疑
# ============================================================
if st.session_state.current_page == "qa":
    with st.container(border=True):
        st.subheader("💬 智能答疑")
    st.write("输入机器学习相关问题，系统会自动判别知识类别、检索最相关知识点，并基于本地知识库组织解答。")
    q = st.text_input("请输入你的问题：", placeholder="例如：什么是过拟合？怎么解决？", key="qa_q")

    # 检索数量：窄数字框 + 提问按钮在数字框下方独立一行
    if "qa_top_k" not in st.session_state:
        st.session_state.qa_top_k = 3

    c_k, _ = st.columns([0.5, 1])
    with c_k:
        st.number_input("检索数量（1~5）", 1, 5, key="qa_top_k")

    ask_clicked = st.button("提问", key="qa_btn", type="primary")

    if ask_clicked and q.strip():
        try:
            r = get_retriever()
            out = r.search(q, top_k=st.session_state.qa_top_k)
        except Exception as e:
            st.error(f"知识库加载失败：{e}\n\n请确认 data/knowledge_base.csv 和 models/ 目录存在。")
            st.stop()

        st.markdown(f"**🏷️ 预测知识类别：** `{out['predicted_category']}`")

        local_ans = build_kb_answer(q, out)

        with st.container(border=True):
            st.markdown("### 🤖 智能解答")
            st.markdown(local_ans)

        with st.container(border=True):
            st.markdown("### 📖 命中的知识点")
            for it in out["results"]:
                flag = "★ 类内命中" if it["in_pred_category"] else "全局命中"
                with st.expander(f"【{it['category']}】{it['title']} （相似度 {it['score']} | {flag}）"):
                    st.write(it["content"])
                    st.caption(f"关键词：{it['keywords']}")


# ============================================================
# 页面 2：PDF 文档问答（RAG）
# ============================================================
elif st.session_state.current_page == "rag":
    with st.container(border=True):
        st.subheader("📄 PDF 文档问答")
    import rag as rag_mod
    if not rag_mod.pdf_available():
        st.warning("未安装 pdfplumber，暂不能解析 PDF。可先 `pip install pdfplumber`；"
                   "或在下方文本框直接粘贴文本进行问答演示。")

    with st.container(border=True):
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
                with st.container(border=True):
                    st.markdown("### 🤖 回答")
                    st.markdown(ans)
                with st.container(border=True):
                    st.markdown("### 📄 检索到的原文片段")
                    for i, (c, s) in enumerate(hits, 1):
                        st.markdown(f"**片段{i}（相似度 {s}）**")
                        st.write(c)
            except Exception as e:
                st.error(f"出错：{e}")


# ============================================================
# 页面 3：模拟考试
# ============================================================
elif st.session_state.current_page == "exam":
    with st.container(border=True):
        st.subheader("📝 模拟考试")
    st.write("系统自动从知识库生成试卷。作答后自动评分，错题自动进入错题本。")

    with st.container(border=True):
        st.markdown("### 🎯 试卷配置")
        c1, c2 = st.columns(2)
        with c1:
            n_q = st.number_input("题目数量", 1, 10, 5, key="ex_n")
        with c2:
            seed = st.number_input("随机种子（换一套题改这里）", 0, 9999, 42, key="ex_seed")

        if st.button("生成试卷", key="ex_gen", type="primary"):
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
        with st.container(border=True):
            st.markdown("### 📝 答题区")
            user_answers = []
            for i, qd in enumerate(qs):
                st.markdown(f"**第{i+1}题（{qd['type']} · {qd['category']}）**")
                st.write(qd["question"])
                ans = st.radio(f"选择第{i+1}题答案", qd["options"], key=f"ex_ans_{i}",
                               index=None)
                user_answers.append(ans if ans is not None else "")
                st.divider()

            if st.button("提交评分", key="ex_submit", type="primary"):
                try:
                    score, total, details = exam_mod.grade(qs, user_answers)
                    st.session_state["exam_graded"] = (score, total, details)
                except Exception as e:
                    st.error(f"评分失败：{e}")

    if "exam_graded" in st.session_state:
        score, total, details = st.session_state["exam_graded"]
        with st.container(border=True):
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
# 页面 4：错题 / 薄弱项分析
# ============================================================
elif st.session_state.current_page == "err":
    with st.container(border=True):
        st.subheader("📚 错题本与薄弱项分析")
    df_m = mistakes.load_mistakes()

    with st.container(border=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"**累计错题：{len(df_m)} 道**")
            if st.button("清空错题本", key="clr"):
                mistakes.clear_mistakes()
                st.rerun()
        with col2:
            st.write(mistakes.advice_text())

    if len(df_m) > 0:
        with st.container(border=True):
            st.markdown("### 📊 薄弱项分布")
            try:
                img = mistakes.plot_weak_points()
                st.markdown('<div style="text-align:center; padding:0.3rem 0;">', unsafe_allow_html=True)
                st.image(img, width=750)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"图表生成失败：{e}")
        with st.container(border=True):
            st.markdown("### 📋 错题明细")
            # 列名改为中文显示
            df_show = df_m.rename(columns={
                "time": "时间",
                "category": "类别",
                "question": "题目",
                "correct_answer": "正确答案",
                "user_answer": "你的答案",
            })
            st.dataframe(df_show)
    else:
        st.caption("暂无错题。去『模拟考试』练几道题吧。")


# ============================================================
# 页面 5：模型评估看板
# ============================================================
elif st.session_state.current_page == "eval":
    with st.container(border=True):
        st.subheader("📊 机器学习模型评估看板")
    report_path = os.path.join(MODELS_DIR, "eval_report.json")
    if not os.path.exists(report_path):
        st.warning("尚未训练模型。请先运行：`python src/train_classify.py`")
    else:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        with st.container(border=True):
            st.markdown(f"### 🏆 最优模型：**{report.get('_best_model','-')}**（按测试集 F1-macro 选取）")
            # 多指标对比表
            rows = []
            for name in ["朴素贝叶斯", "决策树", "随机森林"]:
                if name in report:
                    m = report[name]["metrics"]
                    rows.append({"模型": name, **m})
            if rows:
                st.markdown("#### 三种算法多指标对比")
                st.dataframe(pd.DataFrame(rows).set_index("模型"))

        with st.container(border=True):
            st.markdown("### 📈 混淆矩阵 / ROC 曲线")
            for name in ["朴素贝叶斯", "决策树", "随机森林"]:
                if name not in report:
                    continue
                st.markdown(f"#### {name}")
                cc1, cc2 = st.columns(2)
                cm = os.path.join(MODELS_DIR, report[name]["confusion_png"])
                roc = os.path.join(MODELS_DIR, report[name]["roc_png"])
                if os.path.exists(cm):
                    cc1.image(cm, caption="混淆矩阵", use_container_width=True)
                    if os.path.exists(roc):
                        cc2.image(roc, caption="ROC 曲线", use_container_width=True)
