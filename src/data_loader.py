"""Dataset loaders. Both return a DataFrame with columns: text, label (1 = depression-indicative)."""
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (REDDIT_CSV, TWITTER_DEP_CSV, TWITTER_RANDOM_CSV,
                    VAL_SIZE, TEST_SIZE, SEED)
from preprocess import clean_text


def load_reddit(path=REDDIT_CSV, verbose=True):
    """Our dataset: Depression: Reddit Dataset (Cleaned), Kaggle (infamouscoder), CC0."""
    df = pd.read_csv(path).rename(columns={"clean_text": "raw", "is_depression": "label"})
    n0 = len(df)
    df = df.dropna(subset=["raw"])
    df = df.drop_duplicates(subset=["raw"])            # 81 exact duplicates
    df["text"] = df["raw"].map(clean_text)
    df = df[df["text"].str.split().str.len() >= 3]     # drop posts that end up (almost) empty
    if verbose:
        print(f"Reddit: {n0} rows -> {len(df)} after removing duplicates/empty posts")
    return df[["text", "label"]].reset_index(drop=True)


def load_twitter(dep_rows=3200, random_rows=12000, verbose=True):
    """Original project's data, selected exactly like the original notebook
    (first 3200 scraped tweets, first 12000 Sentiment140 tweets, and the same
    filter: skip tweets that start with a URL or have <= 10 characters)."""
    dep = pd.read_csv(TWITTER_DEP_CSV, sep="|", header=None, usecols=range(9), nrows=dep_rows)[5]
    rnd = pd.read_csv(TWITTER_RANDOM_CSV, encoding="ISO-8859-1", usecols=range(4),
                      nrows=random_rows)["SentimentText"]

    def keep(t):
        t = str(t)
        return re.match(r"(\w+:\/\/\S+)", t) is None and len(t) > 10

    dep = [t for t in dep if keep(t)]
    rnd = [t for t in rnd if keep(t)]
    df = pd.DataFrame({"raw": dep + rnd, "label": [1] * len(dep) + [0] * len(rnd)})
    df["text"] = df["raw"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    if verbose:
        print(f"Twitter: {len(dep)} depressive + {len(rnd)} random tweets")
    return df[["text", "label"]].reset_index(drop=True)


def split(df, seed=SEED):
    """Stratified 70/15/15 split."""
    train, temp = train_test_split(df, test_size=VAL_SIZE + TEST_SIZE,
                                   stratify=df["label"], random_state=seed)
    val, test = train_test_split(temp, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
                                 stratify=temp["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
