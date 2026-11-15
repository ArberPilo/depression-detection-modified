"""Check where the negative class of the Reddit dataset comes from.

The 'not depression' posts look like tweets, so we match them against the
Sentiment140 corpus (the file the original project uses for its random tweets).
Matching uses a loose key (lower case, letters only, trailing 's' removed,
first 6 words) because the Kaggle author lemmatised the text.
"""
import os
import re
import pandas as pd
from config import REDDIT_CSV, TWITTER_RANDOM_CSV, RESULTS_DIR
from evaluation import save_json


def key(s, n=6):
    words = re.sub(r"[^a-z ]", " ", str(s).lower()).split()
    words = [w[:-1] if w.endswith("s") and len(w) > 2 else w for w in words]
    words = [w for w in words if len(w) > 1]
    return " ".join(words[:n]) if len(words) >= 4 else None


red = pd.read_csv(REDDIT_CSV)
s140 = pd.read_csv(TWITTER_RANDOM_CSV, encoding="ISO-8859-1", usecols=range(4), on_bad_lines="skip")
s140.columns = ["ItemID", "Sentiment", "SentimentSource", "SentimentText"]   # the file starts with a byte-order mark
s140["key"] = s140["SentimentText"].map(key)
pos_by_key = s140.dropna(subset=["key"]).groupby("key")["ItemID"].min()
first12k = set(s140.loc[s140["ItemID"] <= 12000, "key"].dropna())

out = {}
for lab in [0, 1]:
    k = red.loc[red["is_depression"] == lab, "clean_text"].map(key).dropna()
    out[f"class_{lab}"] = {
        "posts_with_key": int(len(k)),
        "matched_in_sentiment140": int(k.isin(pos_by_key.index).sum()),
        "share_matched": round(float(k.isin(pos_by_key.index).mean()), 4),
        "matched_in_first_12000_rows": int(k.isin(first12k).sum()),
    }
save_json(out, os.path.join(RESULTS_DIR, "eda", "negative_class_source_check.json"))
print(out)
