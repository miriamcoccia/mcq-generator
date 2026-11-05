"""
summarize_metrics.py: mean text metrics per condition, with and without exact-match questions
"""

import pandas as pd

METRICS = [
    "f1",
    "bleu_1",
    "bleu_4",
    "bertscore",
    "ppl",
    "divs",
    "n_words",
    "answer_match",
    "any_distractor_match",
]

df = pd.read_csv("data/outputs/metrics.csv")


def summarize(data: pd.DataFrame) -> str:
    return data.groupby("condition")[METRICS].mean().round(3).T.to_string()


print("All MCQs")
print(summarize(df))

print("\nExcluding questions with F1 >= 0.999")
print(summarize(df[df["f1"] < 0.999]))
print(df[df["f1"] < 0.999].groupby("condition").size().to_string())
