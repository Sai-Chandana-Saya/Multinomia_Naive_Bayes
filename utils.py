import pickle
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from typing import Tuple
from scipy import sparse as sp
from scipy.sparse import csr_matrix


class Vectorizer:
    """
    A vectorizer class that converts text data into a sparse matrix
    """
    def __init__(self, max_vocab_len=50_000) -> None:
        """
        Initialize the vectorizer
        """
        self.vocab = None
        self.max_vocab_len = max_vocab_len
        self.top_10k_words = None
        self.tfidf_matrix = None


        # TODO: Add more class variables if needed


    def fit(self, X_train: np.ndarray) -> None:
        """
        Fit the vectorizer on the training data
        :param X_train: np.ndarray
            The training sentences
        :return: None
        """
        # TODO: count the occurrences of each word
        d = dict()
        for sentence in X_train:
            words = sentence.split()
            for word in words:
                if word in d:
                    d[word] = d[word]+1
                else:
                    d[word]=1
        sorted_by_word_counts = sorted(d.items(), key=lambda x: x[1], reverse=True)
        self.top_10k_words = sorted_by_word_counts[:self.max_vocab_len]
        self.vocab = {word: idx for idx, word in enumerate(dict(self.top_10k_words))}
        

    def transform(self, X: np.ndarray):
        """
        Transform the input sentences into a sparse matrix based on the
        vocabulary obtained after fitting the vectorizer
        ! Do NOT return a dense matrix, as it will be too large to fit in memory
        :param X: np.ndarray
            Input sentences (can be either train, val or test)
        :return: sp.csr_matrix
            The sparse matrix representation of the input sentences
        """
        assert self.vocab is not None, "Vectorizer not fitted yet"
        # TODO: convert the input sentences into vectors
        #Initialise the sparse matrix
        rows, cols, data = [], [], []  # For constructing the sparse matrix
        doc_counts = np.zeros(len(self.vocab))  # To store document frequencies (DF)

        # Step 1: Compute Term Frequency (TF) and Document Frequency (DF)
        for row_idx, sentence in tqdm(enumerate(X), total=len(X)):
            words = sentence.split()
            word_counts = {}  # To store word counts for the current document

        # Count word occurrences in the current document
            for word in words:
                if word in self.vocab:
                    word_index = self.vocab[word]
                    word_counts[word_index] = word_counts.get(word_index, 0) + 1

            # Normalize TF by the total number of words in the document
            total_words = sum(word_counts.values())
            for word_index, count in word_counts.items():
                rows.append(row_idx)  # Document index
                cols.append(word_index)  # Word index
                data.append(count / total_words)  # Normalized TF

            # Update document frequencies (DF)
            for word_index in set(word_counts.keys()):  # Use set to avoid duplicate counting
                doc_counts[word_index] += 1

        # Step 2: Compute Inverse Document Frequency (IDF)
        idf_values = np.log(len(X) / (doc_counts + 1))

        # Step 3: Compute TF-IDF and store in CSR format
        tf_matrix = csr_matrix((data, (rows, cols)), shape=(len(X), len(self.vocab)))
        tfidf_matrix = tf_matrix.multiply(idf_values)  # Multiply TF by IDF
        self.tfidf_matrix = tfidf_matrix.tocsr()  # Ensure CSR format

        return self.tfidf_matrix       


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_data(
        path: str,
        seed: int
) -> Tuple[np.ndarray, np.ndarray, np.array, np.ndarray]:
    """
    Load twitter sentiment data from csv file and split into train, val and
    test set. Relabel the targets to -1 (for negative) and +1 (for positive).

    :param path: str
        The path to the csv file
    :param seed: int
        The random state for reproducibility
    :return:
        Tuple of numpy arrays - (data, labels) x (train, val) respectively
    """
    # load data
    df = pd.read_csv(path, encoding='utf-8')

    # shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # split into train, val and test set
    train_size = int(0.8 * len(df))  # ~1M for training, remaining ~250k for val
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    x_train, y_train =\
        train_df['stemmed_content'].values, train_df['target'].values
    x_val, y_val = val_df['stemmed_content'].values, val_df['target'].values
    return x_train, y_train, x_val, y_val
