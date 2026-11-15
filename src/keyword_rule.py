"""Reference for Task 6: how far does a one-word rule get?

Predict 'depression-indicative' if the cleaned text contains 'depress', otherwise not.
No training at all. Scored on the same test splits as the models.
"""
import os
import numpy as np
import pandas as pd
from config import RESULTS_DIR
from data_loader import load_reddit, load_twitter, split
from evaluation import compute_metrics

rows = []
for name, df in [("Twitter", load_twitter(verbose=False)), ("Reddit", load_reddit(verbose=False))]:
    test = split(df)[2]
    pred = test["text"].str.contains("depress").astype(float).values
    m = compute_metrics(test["label"].values, pred)
    m["test_on"] = name
    rows.append(m)
out = pd.DataFrame(rows).set_index("test_on")
out.to_csv(os.path.join(RESULTS_DIR, "cross_dataset", "keyword_rule.csv"))
print(out.round(4).to_string())
