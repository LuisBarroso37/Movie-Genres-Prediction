"""
preprocess_data.py.

This script creates a data preprocessing class.
"""

# Imports
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords")


class PreprocessData:
    """Class for data preprocessing."""

    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.tokenizer = Tokenizer()
        self.stopwords = stopwords.words("english")

    def preprocess_train(self, train_df):
        """
        Filter the synopses of the training set.

        This function removes stopwords from the synopses and
        turns the movie genres into multi-label vectors
        """
        # Remove rows with missing values in either 'genres' or 'synopsis'
        train_df.dropna(axis=0, subset=["synopsis", "genres"], inplace=True)

        # Load data and split it per columns
        train_df = train_df.sample(frac=1, random_state=37)  # Shuffle data
        synopses = train_df["synopsis"].apply(lambda x: x.lower())
        genres = train_df["genres"].apply(lambda x: x.split())

        # Remove stopwords from the synopses
        processed_synopses = synopses.apply(
            lambda x: " ".join(
                [word for word in x.split() if word not in self.stopwords]
            )
        )

        # Turn the genres into multi-label vectors
        self.mlb.fit(genres.tolist())
        y_train = self.mlb.transform(genres.tolist())

        # Tokenize synopses
        self.tokenizer.fit_on_texts(processed_synopses)
        self.max_length = 200 # Max length of sequences

        x_train = self.tokenizer.texts_to_sequences(processed_synopses)
        pad_x_train = pad_sequences(x_train, maxlen=200, padding="post", truncating="post")

        return pad_x_train, y_train

    def preprocess_test_data(self, test_df):
        """
        Filter the synopses of the test set.

        This function removes stopwords from the synopses.
        """
        # Remove rows with missing values in 'synopsis'
        test_df.dropna(axis=0, subset=["synopsis"], inplace=True)

        # Load data and split it per columns
        self.movie_ids = test_df["movie_id"]
        synopses = test_df["synopsis"].apply(lambda x: x.lower())

        # Remove stopwords from the synopses
        proc_synopses = synopses.apply(
            lambda x: " ".join(
                [word for word in x.split() if word not in self.stopwords]
            )
        )

        # Tokenize synopses
        x_test = self.tokenizer.texts_to_sequences(proc_synopses)
        padded_x_test = pad_sequences(x_test, maxlen=200, padding="post", truncating="post")

        return padded_x_test