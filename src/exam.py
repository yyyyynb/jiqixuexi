# -*- coding: utf-8 -*-
"""
exam.py —— 模拟试卷自动生成与评分（进阶③）
"""
import os
import json
import random

import pandas as pd

import mistakes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "knowledge_base.csv")


def load_kb():
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


def _make_true_false(row, rng):
    """
    模板法生成判断题。
    若知识库只有一个分类（极端情况），始终生成"正确"陈述以避免崩溃。
    """
    kb = load_kb()
    first_sentence = str(row["content"]).split("。")[0] + "。"
    other_cats = kb[kb["category"] != row["category"]]

    if rng.random() < 0.5 or len(other_cats) == 0:
        # 生成正确陈述
        stmt = f"关于「{row['title']}」：{first_sentence}"
        answer = "正确"
    else:
        # 用另一分类的内容制造错误陈述
        other = other_cats.sample(1, random_state=rng.randint(0, 9999)).iloc[0]
        wrong_sentence = str(other["content"]).split("。")[0] + "。"
        stmt = f"关于「{row['title']}」：{wrong_sentence}"
        answer = "错误"

    return {
        "type": "判断题",
        "category": row["category"],
        "question": stmt,
        "options": ["正确", "错误"],
        "answer": answer,
        "explain": f"{row['title']}：{row['content']}",
    }


def generate_exam(n: int = 5, seed: int = 42):
    """
    生成一份含 n 道题的试卷。
    每道题单独 try-except 保护，一题出错跳过不影响其他题。
    """
    rng = random.Random(seed)
    kb = load_kb()
    picks = kb.sample(min(n, len(kb)), random_state=seed).reset_index(drop=True)

    questions = []
    for _, row in picks.iterrows():
        q = None
        try:
            q = _make_true_false(row, rng)
        except Exception:
            # 单题出错：降级生成一道简单判断题
            try:
                first_sentence = str(row["content"]).split("。")[0] + "。"
                q = {
                    "type": "判断题",
                    "category": row["category"],
                    "question": f"关于「{row['title']}」：{first_sentence}",
                    "options": ["正确", "错误"],
                    "answer": "正确",
                    "explain": f"{row['title']}：{row['content']}",
                }
            except Exception:
                continue  # 实在无法出题则跳过
        if q is not None:
            questions.append(q)
    return questions


def grade(questions, user_answers):
    """
    判分。返回 (score, total, details)。
    """
    total = len(questions)
    correct = 0
    details = []
    for q, ua in zip(questions, user_answers):
        ua_norm = str(ua).strip()
        std = str(q["answer"]).strip()
        if q["type"] == "单选题":
            is_right = ua_norm[:1].upper() == std[:1].upper() if ua_norm else False
        else:
            is_right = ua_norm == std
        if is_right:
            correct += 1
        else:
            try:
                mistakes.add_mistake(q["category"], q["question"], std, ua_norm or "(未作答)")
            except Exception:
                pass
        details.append({
            "question": q["question"],
            "type": q["type"],
            "category": q["category"],
            "your_answer": ua_norm or "(未作答)",
            "correct_answer": std,
            "is_right": is_right,
            "explain": q["explain"],
        })
    return correct, total, details


if __name__ == "__main__":
    qs = generate_exam(n=3)
    for i, q in enumerate(qs, 1):
        print(f"[{i}] ({q['type']}/{q['category']}) {q['question']}")
        print(f"    选项：{q['options']}  答案：{q['answer']}")
    score, total, details = grade(qs, ["正确"] * len(qs))
    print(f"\n得分：{score}/{total}")
