"""
check_copying.py: share of generated questions that match the original, per condition
"""

import pandas as pd

df = pd.read_csv("data/outputs/metrics.csv")

# Boolean column: True where the score is a perfect match
df["copied"] = df["f1"] >= 0.999

summary = df.groupby("condition")["copied"].agg(["sum", "mean", "count"])
print(summary)
