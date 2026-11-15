"""The three neural networks compared in Task 5."""
from tf_keras.models import Sequential
from tf_keras.layers import (Embedding, Conv1D, MaxPooling1D, Dropout, LSTM, GRU,
                             Bidirectional, Dense)
from tf_keras.regularizers import l2
from tf_keras.optimizers import Nadam, RMSprop

from config import MAX_SEQUENCE_LENGTH, EMBEDDING_DIM


def _embedding(emb_matrix):
    # frozen Word2Vec layer, same as the original
    return Embedding(len(emb_matrix), EMBEDDING_DIM, weights=[emb_matrix],
                     input_length=MAX_SEQUENCE_LENGTH, trainable=False)


def original_model(emb_matrix):
    """Experiment 1: the original LSTM + CNN, unchanged."""
    model = Sequential([
        _embedding(emb_matrix),
        Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        LSTM(300),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer=Nadam(), metrics=["acc"])
    return model


def modified_model(emb_matrix):
    """Experiment 2: our modified network.
    Changes vs. the original:
      - architecture: LSTM(300) -> Bidirectional LSTM (150 units per direction, 300 in total)
      - optimiser: Nadam -> RMSprop
      - extra regularisation: L2 weight penalty on the Conv1D, LSTM and output layers
    """
    reg = l2(1e-4)
    model = Sequential([
        _embedding(emb_matrix),
        Conv1D(filters=32, kernel_size=3, padding="same", activation="relu",
               kernel_regularizer=reg),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Bidirectional(LSTM(150, kernel_regularizer=reg)),
        Dropout(0.2),
        Dense(1, activation="sigmoid", kernel_regularizer=reg),
    ])
    model.compile(loss="binary_crossentropy", optimizer=RMSprop(learning_rate=1e-3),
                  metrics=["acc"])
    return model


def gru_model(emb_matrix):
    """Experiment 3: alternative architecture, a plain GRU (no convolution)."""
    model = Sequential([
        _embedding(emb_matrix),
        GRU(128),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer=Nadam(), metrics=["acc"])
    return model


MODELS = {
    "Exp1_original_LSTM_CNN": original_model,
    "Exp2_modified_BiLSTM_CNN": modified_model,
    "Exp3_GRU": gru_model,
}
