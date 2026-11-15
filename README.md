# Depression-indicative text detection: reproduction, modification and cross-dataset test

Course project for *Artificial Neural Networks and Machine Learning*.

This repository is a modified version of
**[peijoy/DetectDepressionInTwitterPosts](https://github.com/peijoy/DetectDepressionInTwitterPosts)**
by Pei-Jo Yang (MIT licence). The original README is kept as `README_original.md`
and the original notebook is kept unchanged as `DepressionDetectionInTwitter.ipynb`.

> **Important:** the labels in these datasets come from *where a post was collected*
> (a keyword search on Twitter, a depression subreddit on Reddit). They are research
> labels, not clinical diagnoses, and neither are the predictions of these models.

## What is in here

| Path | Content |
| --- | --- |
| `DepressionDetectionInTwitter.ipynb` | Original notebook (unchanged) |
| `original_reproduction.ipynb` | Task 2: the original notebook with the minimum fixes needed to run on TF 2.21 / gensim 4 (executed, with outputs) |
| `src/` | Tasks 3–6: modified pipeline (see table below) |
| `results/reproduction/` | Logs and plots of the reproduction |
| `results/eda/` | Exploratory analysis of the new dataset |
| `results/experiments/` | Task 5: metrics, confusion matrices, loss curves, ROC curves |
| `results/cross_dataset/` | Task 6: cross-dataset results and analysis |

| Module | Purpose |
| --- | --- |
| `src/config.py` | Paths and settings |
| `src/preprocess.py` | Shared text cleaning |
| `src/data_loader.py` | `load_reddit()`, `load_twitter()`, stratified 70/15/15 split |
| `src/embeddings.py` | Tokenizer + Word2Vec embedding matrix |
| `src/models.py` | Exp 1 original LSTM+CNN, Exp 2 modified BiLSTM+CNN, Exp 3 GRU |
| `src/evaluation.py` | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, plots |
| `src/eda.py` | Exploratory data analysis |
| `src/source_check.py` | Checks where the Reddit dataset's negative class comes from |
| `src/run_experiments.py` | Task 5: the four experiments |
| `src/cross_dataset.py` | Task 6: train on one dataset, test on the other |

## The five modifications (Task 4)

1. **New dataset loader** – `load_reddit()` for the Kaggle *Depression: Reddit Dataset (Cleaned)*, with de-duplication and a stratified split.
2. **Architecture** – `LSTM(300)` → `Bidirectional(LSTM(150))`.
3. **Optimiser** – Nadam → RMSprop (lr = 0.001).
4. **Regularisation** – L2 weight decay (1e-4) on Conv1D, LSTM and output layers (plus `restore_best_weights` in early stopping).
5. **Evaluation** – precision, recall, F1, ROC-AUC, confusion matrix, loss curves, training/inference time, model size, memory.

Additional: a second architecture (GRU) and a traditional baseline (TF-IDF + logistic regression).

## Setup

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['stopwords', 'punkt', 'punkt_tab']]"
```

Download the data (not committed to this repository, to respect the original licences):

| File | Put it at | Source |
| --- | --- | --- |
| Depression: Reddit Dataset (Cleaned) | `data/depression_dataset_reddit_cleaned.csv` | https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned |
| Sentiment140 "Sentiment Analysis Dataset" (1,578,614 tweets) | `data/Sentiment Analysis Dataset 2.csv` | Kaggle dataset `ywang311/twitter-sentiment` (linked from the original notebook) |
| Google News Word2Vec, 300d | `embeddings/GoogleNews-vectors-negative300.bin.gz` | https://github.com/RaRe-Technologies/gensim-data/releases/tag/word2vec-google-news-300 |

`depressive_tweets_processed.csv` is part of the original repository and stays in the root.

## How to run

```bash
jupyter nbconvert --to notebook --execute original_reproduction.ipynb --output original_reproduction.ipynb   # Task 2 (~4 min)
python src/eda.py                # Task 3 (~1 min)
python src/source_check.py       # Task 3, source of the negative class (~1 min)
python src/run_experiments.py    # Task 5, 3 networks x 5 seeds + TF-IDF (~20 min on CPU)
python src/cross_dataset.py      # Task 6 (~10 min on CPU)
```

The first script that loads Word2Vec takes a few minutes and saves a faster `.kv` copy next to it.
Everything runs on a CPU; no GPU is needed.

## Results

**Task 2 – reproduction.** 98.87 % test accuracy vs. 98.91 % reported (one tweet difference).
The original logistic-regression baseline (83.755 %) only predicts the majority class (2,382 / 2,844).

**Task 5 – our Reddit dataset** (1,099 test posts; networks: mean ± sd over 5 seeds)

| Experiment | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Original LSTM + CNN | 0.973 ± 0.002 | 0.987 ± 0.007 | 0.960 ± 0.009 | 0.973 ± 0.002 | 0.996 | 80 s |
| 2 | Modified BiLSTM + CNN (RMSprop, L2) | 0.969 ± 0.008 | 0.981 ± 0.010 | 0.957 ± 0.026 | 0.969 ± 0.009 | 0.996 | 67 s |
| 3 | GRU | 0.974 ± 0.001 | 0.985 ± 0.005 | 0.963 ± 0.005 | 0.974 ± 0.001 | 0.997 | 70 s |
| 4 | TF-IDF + logistic regression | 0.954 | 0.962 | 0.946 | 0.954 | 0.991 | 0.4 s |

**Task 6 – cross-dataset** (modified model, F1, mean ± sd over 3 seeds)

| Train → test | F1 | One-word rule ("depress") on the same test set |
| --- | --- | --- |
| Twitter → Twitter | 0.963 ± 0.001 | 0.955 |
| Reddit → Twitter | 0.922 ± 0.031 | 0.955 |
| Reddit → Reddit | 0.966 ± 0.011 | 0.597 |
| Twitter → Reddit | 0.866 ± 0.047 | 0.597 |

Note: 95 % of the Reddit dataset's negative texts are Sentiment140 tweets (`src/source_check.py`),
the same corpus the original project uses for its negatives, so the cross-dataset numbers are an upper bound.
