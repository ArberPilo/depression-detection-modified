"""Word2Vec (GoogleNews, 300d) embedding matrix, as in the original notebook."""
import os
import numpy as np
from gensim.models import KeyedVectors
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences

from config import EMBEDDING_FILE, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM

_W2V = None


def word2vec():
    """Load once. The first time we also save a gensim .kv copy, which loads much faster."""
    global _W2V
    if _W2V is None:
        kv_path = EMBEDDING_FILE.replace(".bin.gz", ".kv")
        if os.path.exists(kv_path):
            _W2V = KeyedVectors.load(kv_path, mmap="r")
        else:
            print("Loading Word2Vec (first run takes a couple of minutes)...")
            _W2V = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
            _W2V.save(kv_path)
    return _W2V


def fit_tokenizer(texts):
    tok = Tokenizer(num_words=MAX_NB_WORDS)
    tok.fit_on_texts(texts)
    return tok


def to_sequences(tok, texts):
    return pad_sequences(tok.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)


def embedding_matrix(tok):
    w2v = word2vec()
    n = min(MAX_NB_WORDS, len(tok.word_index) + 1)
    m = np.zeros((n, EMBEDDING_DIM))
    for word, idx in tok.word_index.items():
        if idx < n and word in w2v.key_to_index:
            m[idx] = w2v.get_vector(word)
    return m
