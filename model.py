"""
model.py.

This script creates a class that trains the model
and makes predictions on new data.
"""

# Imports
import pandas as pd
import numpy as np
import tensorflow as tf
from .preprocess_data import PreprocessData
from tensorflow.keras.layers import Dense, Embedding, Dropout
from tensorflow.keras.layers import GRU, Bidirectional
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision


class Model:
    """Class for training model and make predictions."""

    def __init__(self):
        self.data = PreprocessData()

    def train_model(self, train_df):
        """
        Trains the RNN model using the training set.

        The GloVe word embeddings used were downloaded from
        https://nlp.stanford.edu/projects/glove/.
        """
        # Get preprocessed training set
        padded_x_train, y_train = self.data.preprocess_train(train_df)

        # Using pre-trained GloVe embeddings for embedding layer
        f = open("Data/glove.6B.100d.txt", encoding="utf8")

        # Creating dictionary with the words as keys
        # and vector representations as values
        embeddings_index = {}
        for row in f:
            arr = row.split()
            word = arr[0]
            coefs = np.asarray(arr[1:], dtype="float32")
            embeddings_index[word] = coefs
        f.close()

        # Creating embedding matrix
        word_index = self.data.tokenizer.word_index
        vocab_size = len(word_index) + 1
        emb_dim = 100

        embedding_matrix = np.zeros((vocab_size, emb_dim))

        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        # Set random seed
        tf.random.set_seed(37)

        # Set parameters
        max_length = self.data.max_length
        gru_num_cells = 128
        dense_num_cells = 128
        num_outputs = y_train.shape[1]

        # Creating model
        self.model = Sequential(
            [
                Embedding(
                    vocab_size,
                    emb_dim,
                    weights=[embedding_matrix],
                    input_length=max_length,
                    trainable=False
                ),
                Bidirectional(GRU(gru_num_cells, return_sequences=True)),
                Dropout(0.1),
                GlobalMaxPooling1D(),
                Dropout(0.1),
                Dense(dense_num_cells, activation="relu"),
                Dense(num_outputs, activation="sigmoid")
            ]
        )

        # Compiling model
        optimizer = Adam(lr=1e-2)
        self.model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=[Precision(0.3, 5)]
        )

        # Fit model to the data
        self.model.fit(
            padded_x_train,
            y_train,
            epochs=10,
            batch_size=512
        )

    def predict(self, test_df):
        """Make predictions on the test set."""
        # Get preprocessed test set
        padded_x_test = self.data.preprocess_test_data(test_df)

        # Make predictions
        y_preds = self.model.predict(padded_x_test)

        # Get top 5 predicted genres for each movie
        outputs = []
        classes = np.asarray(self.data.mlb.classes_)

        for pred in y_preds:
            sorted_indices = np.argsort(pred)[::-1]
            output = classes[sorted_indices][:5]
            output_text = " ".join(output)
            outputs.append(output_text)

        outputs = np.asarray(outputs)

        # Create DataFrame with movie ids and top 5 predicted genres
        predictions_df = pd.DataFrame(
            {"movie_id": self.data.movie_ids, "predicted_genres": outputs}
        )

        return predictions_df