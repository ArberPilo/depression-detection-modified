"""Shared settings for the modified pipeline (Tasks 3-6)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# data
REDDIT_CSV = os.path.join(ROOT, "data", "depression_dataset_reddit_cleaned.csv")
TWITTER_DEP_CSV = os.path.join(ROOT, "depressive_tweets_processed.csv")
TWITTER_RANDOM_CSV = os.path.join(ROOT, "data", "Sentiment Analysis Dataset 2.csv")
EMBEDDING_FILE = os.path.join(ROOT, "embeddings", "GoogleNews-vectors-negative300.bin.gz")
RESULTS_DIR = os.path.join(ROOT, "results")

# same values as the original notebook where possible
MAX_SEQUENCE_LENGTH = 140
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
BATCH_SIZE = 40
EPOCHS = 10
PATIENCE = 3

# our split: 70 / 15 / 15, stratified
VAL_SIZE = 0.15
TEST_SIZE = 0.15
SEED = 1234
