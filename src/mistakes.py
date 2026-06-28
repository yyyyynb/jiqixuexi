# -*- coding: utf-8 -*-
"""
mistakes.py —— 错题收集与薄弱项分析（进阶②）

功能：
  - 错题以 CSV 持久化（题目/正确答案/用户答案/所属类别/时间）。
  - 按类别统计错误数量与错误率，输出薄弱知识点排名。
  - 生成针对性复习建议，并绘制薄弱项柱状图。
"""
import os
import csv
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MISTAKES_PATH = os.path.join(DATA_DIR, "mistakes.csv")
FIELDS = ["time", "category", "question", "correct_answer", "user_answer"]


def _ensure_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(MISTAKES_PATH):
        with open(MISTAKES_PATH, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(FIELDS)


def add_mistake(category: str, question: str, correct_answer: str, user_answer: str):
    """记录一道错题（固定使用当前时间字符串，避免依赖随机/时区）。"""
    _ensure_file()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(MISTAKES_PATH, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([ts, category, question, correct_answer, user_answer])


def load_mistakes() -> pd.DataFrame:
    _ensure_file()
    try:
        return pd.read_csv(MISTAKES_PATH, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=FIELDS)


def weak_points(top_n: int = 6):
    """
    统计各类别错题数量，返回薄弱项排名 DataFrame（按错题数降序）。
    """
    df = load_mistakes()
    if len(df) == 0:
        return pd.DataFrame(columns=["category", "错题数"])
    stat = (df.groupby("category").size()
              .reset_index(name="错题数")
              .sort_values("错题数", ascending=False)
              .head(top_n)
              .reset_index(drop=True))
    return stat


def advice_text():
    """根据薄弱项生成文字复习建议。"""
    stat = weak_points()
    if len(stat) == 0:
        return "暂无错题记录，继续保持！可在『模拟考试』中练习以收集错题。"
    lines = ["薄弱项复习建议：", ""]
    for _, row in stat.iterrows():
        lines.append(f"- 【{row['category']}】累计错题 {row['错题数']} 道，"
                     f"建议重点复习该类别的核心知识点。")
    top = stat.iloc[0]["category"]
    lines.append("")
    lines.append(f"=> 当前最薄弱类别为「{top}」，建议优先针对性突破。")
    return "\n".join(lines)


def plot_weak_points(out_path=None):
    """绘制薄弱项柱状图，返回图片路径。"""
    stat = weak_points()
    if out_path is None:
        out_path = os.path.join(DATA_DIR, "weak_points.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(stat) == 0:
        ax.text(0.5, 0.5, "暂无错题记录", ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        ax.bar(stat["category"], stat["错题数"], color="#e07a5f")
        ax.set_ylabel("错题数")
        ax.set_title("各知识类别错题分布（薄弱项分析）")
        plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def clear_mistakes():
    """清空错题本。"""
    _ensure_file()
    with open(MISTAKES_PATH, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(FIELDS)


if __name__ == "__main__":
    # 演示：写入几条错题并打印薄弱项
    add_mistake("模型评估", "AUC=0.5代表什么", "随机猜测", "完美分类")
    add_mistake("神经网络", "ReLU的表达式", "max(0,x)", "1/(1+e^-x)")
    add_mistake("模型评估", "F1是什么的平均", "精确率召回率调和平均", "算术平均")
    print(weak_points().to_string(index=False))
    print()
    print(advice_text())
