"""Task 3: exploratory data analysis of the Reddit dataset."""
import os
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import REDDIT_CSV, RESULTS_DIR, MAX_SEQUENCE_LENGTH
from data_loader import load_reddit, load_twitter, split
from evaluation import save_json

OUT = os.path.join(RESULTS_DIR, "eda")
os.makedirs(OUT, exist_ok=True)

raw = pd.read_csv(REDDIT_CSV)
info = {
    "rows": len(raw),
    "columns": list(raw.columns),
    "class_counts": raw["is_depression"].value_counts().to_dict(),
    "missing_values": raw.isna().sum().to_dict(),
    "duplicate_texts": int(raw["clean_text"].duplicated().sum()),
    "posts_under_3_words": int((raw["clean_text"].str.split().str.len() < 3).sum()),
    # inconsistencies / leftovers from the original scraping+cleaning
    "posts_with_url_fragments": int(raw["clean_text"].str.contains(r"http|www").sum()),
    "posts_with_amp_html_leftover": int(raw["clean_text"].str.contains(r"\bamp\b").sum()),
    "posts_with_usernames_u_or_at": int(raw["clean_text"].str.contains(r"\bu/|@").sum()),
    "keyword_depress_share_by_class": raw.groupby("is_depression")["clean_text"]
        .apply(lambda s: round(s.str.contains("depress").mean(), 4)).to_dict(),
    "words_per_post_by_class": raw.assign(n=raw["clean_text"].str.split().str.len())
        .groupby("is_depression")["n"].describe().round(1).to_dict(orient="index"),
}

df = load_reddit()
n_words = df["text"].str.split().str.len()
info["after_cleaning_rows"] = len(df)
info["after_cleaning_class_counts"] = df["label"].value_counts().to_dict()
info["after_cleaning_words_median"] = float(n_words.median())
info["after_cleaning_words_p90"] = float(n_words.quantile(0.9))
info["share_posts_fitting_max_len"] = round(float((n_words <= MAX_SEQUENCE_LENGTH).mean()), 4)

train, val, test = split(df)
info["split_sizes"] = {k: len(v) for k, v in [("train", train), ("val", val), ("test", test)]}
info["split_positive_share"] = {k: round(v["label"].mean(), 4)
                                for k, v in [("train", train), ("val", val), ("test", test)]}

# the same keyword check on the original Twitter data (used in Task 6)
tw = load_twitter(verbose=False)
info["twitter_keyword_depress_share_by_class"] = tw.groupby("label")["text"] \
    .apply(lambda s: round(s.str.contains("depress").mean(), 4)).to_dict()
info["twitter_class_counts"] = tw["label"].value_counts().to_dict()
info["twitter_words_median"] = float(tw["text"].str.split().str.len().median())
save_json(info, os.path.join(OUT, "eda_summary.json"))

# figure 1: class balance
fig, ax = plt.subplots(figsize=(3.5, 3))
counts = raw["is_depression"].value_counts().sort_index()
ax.bar(["not depression (0)", "depression (1)"], counts.values, color=["#8aa1b1", "#c0504d"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, str(v), ha="center")
ax.set_ylabel("posts"); ax.set_title("Reddit dataset: class balance", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "class_balance.png"), dpi=150); plt.close(fig)

# figure 2: post length per class (after cleaning), shared bins so the classes are comparable
import numpy as np
fig, ax = plt.subplots(figsize=(5.5, 3.2))
bins = np.arange(0, 305, 5)
for lab, col in [(0, "#8aa1b1"), (1, "#c0504d")]:
    ax.hist(n_words[df["label"] == lab].clip(upper=300), bins=bins, alpha=0.65, color=col,
            label=["not depression (median %d words)" % n_words[df["label"] == 0].median(),
                   "depression (median %d words)" % n_words[df["label"] == 1].median()][lab])
ax.axvline(MAX_SEQUENCE_LENGTH, color="k", ls="--", lw=1, label=f"max length = {MAX_SEQUENCE_LENGTH}")
ax.set_xlabel("words per post after cleaning (last bar = 300 or more)"); ax.set_ylabel("posts")
ax.set_title("Negative posts are short, positive posts are long", fontsize=10)
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(OUT, "length_by_class.png"), dpi=150); plt.close(fig)

# figure 3: most frequent words per class
fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
for ax, lab in zip(axes, [0, 1]):
    c = Counter(" ".join(df.loc[df["label"] == lab, "text"]).split()).most_common(15)[::-1]
    ax.barh([w for w, _ in c], [n for _, n in c], color=["#8aa1b1", "#c0504d"][lab])
    ax.set_title(["not depression", "depression"][lab], fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "top_words.png"), dpi=150); plt.close(fig)

for k, v in info.items():
    print(f"{k}: {v}")
