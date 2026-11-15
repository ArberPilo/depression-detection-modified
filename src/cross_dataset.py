"""Task 6: cross-dataset evaluation with our modified model (Experiment 2).

Train on Twitter (original project data) -> test on Twitter and on Reddit.
Train on Reddit (our data)               -> test on Reddit and on Twitter.

    python src/cross_dataset.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tf_keras
from tf_keras.callbacks import EarlyStopping

from config import RESULTS_DIR, EPOCHS, BATCH_SIZE, PATIENCE
from data_loader import load_reddit, load_twitter, split
from embeddings import fit_tokenizer, to_sequences, embedding_matrix
from evaluation import compute_metrics, plot_confusion, save_json
from models import modified_model
from run_experiments import class_weights

OUT = os.path.join(RESULTS_DIR, "cross_dataset")
os.makedirs(OUT, exist_ok=True)
SEEDS = [1234, 1, 2]


def oov_rate(tok, texts):
    """Share of word occurrences in `texts` that the source tokenizer does not know."""
    known = {w for w, i in tok.word_index.items() if i < tok.num_words}
    words = " ".join(texts).split()
    return float(np.mean([w not in known for w in words]))


def keyword_breakdown(df, prob):
    """How the model behaves on posts with / without the word 'depress*'."""
    has_kw = df["text"].str.contains("depress").values
    pred = (prob >= 0.5).astype(int)
    y = df["label"].values
    out = {}
    for name, mask in [("with_keyword", has_kw), ("without_keyword", ~has_kw)]:
        pos, neg = mask & (y == 1), mask & (y == 0)
        out[name] = {
            "n_depression_posts": int(pos.sum()),
            "recall_on_depression_posts": float(pred[pos].mean()) if pos.any() else None,
            "n_other_posts": int(neg.sum()),
            "false_positive_rate_on_other_posts": float(pred[neg].mean()) if neg.any() else None,
        }
    return out


def examples(df, prob, kind, n=6):
    y, pred = df["label"].values, (prob >= 0.5).astype(int)
    mask = (y == 0) & (pred == 1) if kind == "fp" else (y == 1) & (pred == 0)
    idx = np.where(mask)[0]
    idx = idx[np.argsort(-np.abs(prob[idx] - 0.5))][:n]     # the most confident mistakes
    return [{"prob_depression": round(float(prob[i]), 3),
             "text_first_25_words": " ".join(df["text"].iloc[i].split()[:25])} for i in idx]


def main():
    data = {"Twitter": load_twitter(), "Reddit": load_reddit()}
    splits = {k: split(v) for k, v in data.items()}
    rows, analysis = [], {}

    for src in ["Twitter", "Reddit"]:
        train, val, _ = splits[src]
        tok = fit_tokenizer(train["text"])
        emb = embedding_matrix(tok)
        Xtr, Xva = to_sequences(tok, train["text"]), to_sequences(tok, val["text"])
        ytr, yva = train["label"].values, val["label"].values
        cw = class_weights(ytr)

        for k, seed in enumerate(SEEDS):
            tf_keras.utils.set_random_seed(seed)
            model = modified_model(emb)
            model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS,
                      batch_size=BATCH_SIZE, class_weight=cw, verbose=2,
                      callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                               restore_best_weights=True)])
            for tgt in ["Twitter", "Reddit"]:
                test = splits[tgt][2]
                prob = model.predict(to_sequences(tok, test["text"]), verbose=0).ravel()
                m = compute_metrics(test["label"].values, prob)
                m.update({"train_on": src, "test_on": tgt, "seed": seed,
                          "setting": "in-domain" if src == tgt else "cross-domain",
                          "mean_predicted_prob": float(prob.mean()),
                          "share_predicted_positive": float((prob >= 0.5).mean())})
                rows.append(m)
                print(src, "->", tgt, "seed", seed,
                      {k2: round(v, 4) for k2, v in m.items() if isinstance(v, float)})
                if k == 0:   # detailed analysis on the first seed
                    tag = f"{src}_to_{tgt}"
                    plot_confusion(m, f"train {src} -> test {tgt}",
                                   os.path.join(OUT, f"confusion_{tag}.png"))
                    analysis[tag] = {
                        "oov_rate_vs_source_vocab": oov_rate(tok, test["text"]),
                        "keyword_breakdown": keyword_breakdown(test, prob),
                        "false_positive_examples": examples(test, prob, "fp"),
                        "false_negative_examples": examples(test, prob, "fn"),
                    }
                    # recall by post length (domain shift in length)
                    n = test["text"].str.split().str.len().values
                    y, pred = test["label"].values, (prob >= 0.5).astype(int)
                    bins = [0, 10, 25, 50, 100, 10**6]
                    analysis[tag]["by_length"] = []
                    for lo, hi in zip(bins[:-1], bins[1:]):
                        msk = (n > lo) & (n <= hi)
                        if msk.sum():
                            analysis[tag]["by_length"].append({
                                "words": f"{lo + 1}-{hi if hi < 10**6 else 'max'}",
                                "n": int(msk.sum()), "share_depression": float(y[msk].mean()),
                                "accuracy": float((pred[msk] == y[msk]).mean()),
                                "share_predicted_depression": float(pred[msk].mean())})

    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(OUT, "cross_all_runs.csv"), index=False)
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "share_predicted_positive"]
    summary = runs.groupby(["train_on", "test_on"], sort=False)[cols].agg(["mean", "std"]).round(4)
    summary.to_csv(os.path.join(OUT, "cross_summary.csv"))
    analysis["dataset_stats"] = {
        k: {"n": len(v), "share_depression": float(v["label"].mean()),
            "median_words": float(v["text"].str.split().str.len().median()),
            "keyword_share_in_depression_class":
                float(v.loc[v["label"] == 1, "text"].str.contains("depress").mean())}
        for k, v in data.items()}
    save_json(analysis, os.path.join(OUT, "cross_analysis.json"))
    print(summary.to_string())

    # bar chart: F1 in-domain vs cross-domain
    means = runs.groupby(["train_on", "test_on"])["f1"].mean()
    stds = runs.groupby(["train_on", "test_on"])["f1"].std()
    labels = ["Twitter→Twitter", "Twitter→Reddit", "Reddit→Reddit", "Reddit→Twitter"]
    keys = [("Twitter", "Twitter"), ("Twitter", "Reddit"), ("Reddit", "Reddit"), ("Reddit", "Twitter")]
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.bar(labels, [means[k] for k in keys], yerr=[stds[k] for k in keys], capsize=4,
           color=["#4f81bd", "#c0504d", "#4f81bd", "#c0504d"])
    for i, k in enumerate(keys):
        ax.text(i, means[k] + 0.03, f"{means[k]:.2f}", ha="center")
    ax.set_ylim(0, 1.1); ax.set_ylabel("F1 (depression class)")
    ax.set_title("In-domain (blue) vs cross-domain (red)", fontsize=10)
    plt.xticks(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f1_in_vs_cross.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
