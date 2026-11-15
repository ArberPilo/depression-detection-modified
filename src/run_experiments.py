"""Task 5: train and evaluate the four models on our Reddit dataset.

    python src/run_experiments.py

Writes everything to results/experiments/.
"""
import os
import resource
import time

import numpy as np
import pandas as pd
import tf_keras
from tf_keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

from config import RESULTS_DIR, SEED, EPOCHS, BATCH_SIZE, PATIENCE
from data_loader import load_reddit, split
from embeddings import fit_tokenizer, to_sequences, embedding_matrix
from evaluation import compute_metrics, plot_confusion, plot_history, plot_roc, save_json
from models import MODELS

OUT = os.path.join(RESULTS_DIR, "experiments")
os.makedirs(OUT, exist_ok=True)


def peak_ram_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB -> MB


def class_weights(y):
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return {0: float(w[0]), 1: float(w[1])}


SEEDS = [1234, 1, 2, 3, 4]   # 5 runs per network; the split stays the same


def train_nn(name, build_fn, emb, Xtr, ytr, Xva, yva, Xte, yte, seed=SEED, save_plots=True):
    tf_keras.utils.set_random_seed(seed)
    model = build_fn(emb)
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS,
                     batch_size=BATCH_SIZE, class_weight=class_weights(ytr),
                     callbacks=[early], verbose=2)
    train_time = time.time() - t0
    t0 = time.time()
    prob = model.predict(Xte, verbose=0).ravel()
    infer_ms = (time.time() - t0) / len(Xte) * 1000
    m = compute_metrics(yte, prob)
    path = os.path.join(OUT, f"{name}.keras")
    model.save(path)
    m.update({
        "model": name,
        "epochs_run": len(hist.history["loss"]),
        "train_time_s": round(train_time, 1),
        "inference_ms_per_post": round(infer_ms, 3),
        "trainable_params": trainable,
        "total_params": int(model.count_params()),
        "model_file_mb": round(os.path.getsize(path) / 1e6, 1),
        "peak_ram_mb_process": round(peak_ram_mb()),
    })
    m["seed"] = seed
    if save_plots:
        plot_history(hist.history, name, os.path.join(OUT, f"{name}_loss.png"))
        plot_confusion(m, name, os.path.join(OUT, f"{name}_confusion.png"))
        save_json(hist.history, os.path.join(OUT, f"{name}_history.json"))
    os.remove(path)  # do not keep large model files in the repo
    return m, prob


def train_tfidf_lr(train, test):
    """Experiment 4: traditional ML baseline, TF-IDF (1-2 grams) + logistic regression."""
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform(train["text"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(Xtr, train["label"])
    train_time = time.time() - t0
    t0 = time.time()
    prob = clf.predict_proba(vec.transform(test["text"]))[:, 1]
    infer_ms = (time.time() - t0) / len(test) * 1000
    m = compute_metrics(test["label"].values, prob)
    m.update({"model": "Exp4_TFIDF_LogReg", "epochs_run": None,
              "train_time_s": round(train_time, 1), "inference_ms_per_post": round(infer_ms, 3),
              "trainable_params": int(Xtr.shape[1] + 1), "total_params": int(Xtr.shape[1] + 1),
              "model_file_mb": None, "peak_ram_mb_process": round(peak_ram_mb())})
    plot_confusion(m, "Exp4_TFIDF_LogReg", os.path.join(OUT, "Exp4_TFIDF_LogReg_confusion.png"))
    # the words with the largest weights, useful for the discussion
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    save_json({"top_depression_terms": names[order[-25:]][::-1].tolist(),
               "top_non_depression_terms": names[order[:25]].tolist()},
              os.path.join(OUT, "Exp4_top_terms.json"))
    return m, prob


def main():
    df = load_reddit()
    train, val, test = split(df)
    print(f"train {len(train)}  val {len(val)}  test {len(test)}")
    print("class weights:", class_weights(train["label"].values))

    tok = fit_tokenizer(train["text"])          # fitted on the training set only
    emb = embedding_matrix(tok)
    covered = int((np.abs(emb).sum(1) > 0).sum())
    print(f"vocabulary {len(tok.word_index)}, embedding rows {len(emb)}, with Word2Vec vector {covered}")
    Xtr, Xva, Xte = (to_sequences(tok, d["text"]) for d in (train, val, test))
    ytr, yva, yte = (d["label"].values for d in (train, val, test))

    rows, curves = [], {}
    for name, fn in MODELS.items():
        for i, seed in enumerate(SEEDS):
            print(f"\n===== {name}  (seed {seed}) =====")
            m, p = train_nn(name, fn, emb, Xtr, ytr, Xva, yva, Xte, yte,
                            seed=seed, save_plots=(i == 0))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
            rows.append(m)
            if i == 0:
                curves[name] = (yte, p)

    print("\n===== Exp4_TFIDF_LogReg =====")
    m, p = train_tfidf_lr(train, test)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    rows.append(m); curves["Exp4_TFIDF_LogReg"] = (yte, p)

    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(OUT, "all_runs.csv"), index=False)
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "train_time_s",
            "epochs_run", "inference_ms_per_post"]
    summary = runs.groupby("model", sort=False)[cols].agg(["mean", "std"]).round(4)
    summary.to_csv(os.path.join(OUT, "results_summary.csv"))
    plot_roc(curves, os.path.join(OUT, "roc_curves.png"))
    print("\n", summary.to_string())


if __name__ == "__main__":
    main()
