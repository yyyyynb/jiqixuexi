# -*- coding: utf-8 -*-
"""
train_classify.py —— 知识点分类模型训练与评估（必做核心）

实现 3 种监督学习分类算法，对知识点按 6 大类别分类：
  1. 朴素贝叶斯（MultinomialNB）—— 方向要求的主算法
  2. 决策树（DecisionTreeClassifier）
  3. 随机森林（RandomForestClassifier）

流程：
  jieba 中文分词 -> TF-IDF 向量化 -> GridSearchCV 网格搜索调参
  -> 多指标评估（准确率/精确率/召回率/F1/AUC）
  -> 绘制混淆矩阵与 ROC 曲线 -> 保存最优模型与评估报告

运行：python src/train_classify.py
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # 无界面后端，便于保存图片
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from text_utils import tokenize

warnings.filterwarnings("ignore")

# 中文字体，防止图表中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "knowledge_base.csv")
MODELS_DIR = os.path.join(ROOT, "models")


def load_data():
    """加载知识库，拼接标题+内容+关键词作为分类文本。"""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    # 组合文本特征：标题与关键词更具区分度，重复以加权
    df["text"] = (df["title"] + " " + df["title"] + " "
                  + df["keywords"] + " " + df["content"])
    return df


def build_models():
    """定义 3 种算法的 Pipeline 与网格搜索参数。"""
    # 统一的 TF-IDF 向量器（使用自定义 jieba 分词器）
    def make_tfidf():
        return TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                               ngram_range=(1, 2), min_df=1)

    models = {
        "朴素贝叶斯": {
            "pipe": Pipeline([("tfidf", make_tfidf()), ("clf", MultinomialNB())]),
            "params": {"clf__alpha": [0.01, 0.1, 0.5, 1.0]},
        },
        "决策树": {
            "pipe": Pipeline([("tfidf", make_tfidf()), ("clf", DecisionTreeClassifier(random_state=42))]),
            "params": {"clf__max_depth": [10, 20, 30, None],
                       "clf__min_samples_split": [2, 3]},
        },
        "随机森林": {
            "pipe": Pipeline([("tfidf", make_tfidf()), ("clf", RandomForestClassifier(random_state=42))]),
            "params": {"clf__n_estimators": [100, 200],
                       "clf__max_depth": [20, None]},
        },
    }
    return models


def evaluate(name, best_model, X_test, y_test, classes):
    """计算多指标评估，并返回指标字典与测试集预测概率。"""
    y_pred = best_model.predict(X_test)
    metrics = {
        "准确率": round(accuracy_score(y_test, y_pred), 4),
        "精确率(macro)": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "召回率(macro)": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "F1(macro)": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
    }
    # AUC（多分类 one-vs-rest）
    try:
        y_score = best_model.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        auc = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")
        metrics["AUC(macro-ovr)"] = round(auc, 4)
    except Exception as e:
        metrics["AUC(macro-ovr)"] = None
        print(f"  [警告] {name} 计算 AUC 失败：{e}")
    return metrics, y_pred


def plot_confusion(name, y_test, y_pred, classes, out_path):
    """绘制并保存混淆矩阵。"""
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"{name} 混淆矩阵")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc(name, best_model, X_test, y_test, classes, out_path):
    """绘制多分类 ROC 曲线（one-vs-rest）。"""
    try:
        from sklearn.metrics import roc_curve, auc as auc_fn
        y_score = best_model.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_fn(fpr, tpr):.2f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("假正率 FPR")
        ax.set_ylabel("真正率 TPR")
        ax.set_title(f"{name} ROC 曲线（OvR）")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  [警告] {name} 绘制 ROC 失败：{e}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_data()
    classes = sorted(df["category"].unique().tolist())
    print(f"加载知识库：{len(df)} 条，{len(classes)} 个类别 -> {classes}\n")

    X = df["text"].values
    y = df["category"].values
    # 分层划分，保证每个类别在训练/测试集都有样本
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"训练集 {len(X_train)} 条 | 测试集 {len(X_test)} 条\n")

    models = build_models()
    report = {}          # 汇总各模型评估
    best_overall = None  # (name, f1, model)

    for name, cfg in models.items():
        print("=" * 55)
        print(f"训练模型：{name}")
        grid = GridSearchCV(cfg["pipe"], cfg["params"], cv=3,
                            scoring="f1_macro", n_jobs=1)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        print(f"  最优参数：{grid.best_params_}")

        metrics, y_pred = evaluate(name, best, X_test, y_test, classes)
        print("  测试集多指标评估：")
        for k, v in metrics.items():
            print(f"    {k:<16}: {v}")

        # 保存图表
        cm_path = os.path.join(MODELS_DIR, f"confusion_{name}.png")
        roc_path = os.path.join(MODELS_DIR, f"roc_{name}.png")
        plot_confusion(name, y_test, y_pred, classes, cm_path)
        plot_roc(name, best, X_test, y_test, classes, roc_path)

        report[name] = {
            "best_params": {k: str(v) for k, v in grid.best_params_.items()},
            "metrics": metrics,
            "confusion_png": os.path.basename(cm_path),
            "roc_png": os.path.basename(roc_path),
        }

        f1 = metrics["F1(macro)"]
        if best_overall is None or f1 > best_overall[1]:
            best_overall = (name, f1, best)

    # ---- 保存最优模型 ----
    best_name, best_f1, best_model = best_overall
    model_path = os.path.join(MODELS_DIR, "nb_classifier.pkl")
    joblib.dump({"name": best_name, "model": best_model, "classes": classes}, model_path)

    # 始终单独保存朴素贝叶斯（方向要求主算法），供检索两级使用
    nb_grid = GridSearchCV(models["朴素贝叶斯"]["pipe"],
                           models["朴素贝叶斯"]["params"], cv=3,
                           scoring="f1_macro", n_jobs=1)
    nb_grid.fit(X, y)  # 用全量数据训练最终上线的 NB
    joblib.dump({"name": "朴素贝叶斯", "model": nb_grid.best_estimator_, "classes": classes},
                os.path.join(MODELS_DIR, "nb_final.pkl"))

    report["_best_model"] = best_name
    report["_summary"] = "最优模型按测试集 F1(macro) 选取"
    with open(os.path.join(MODELS_DIR, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 55)
    print(f"最优模型：{best_name}（F1-macro={best_f1}），已保存到 models/nb_classifier.pkl")
    print(f"朴素贝叶斯最终模型已保存到 models/nb_final.pkl")
    print(f"评估报告：models/eval_report.json")
    print("混淆矩阵 / ROC 曲线图已保存到 models/ 目录")
    print("=" * 55)


if __name__ == "__main__":
    main()
