import numpy as np
from scipy import sparse as sp


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes model with efficient incremental updates."""

    def __init__(self, alpha=6) -> None:
        """Initialize the model"""
        self.alpha = alpha
        self.priors = None
        self.means = None
        self.unique_classes = None
        self.log_priors = None
        self.log_class_conditional_probabilities = None
        self.num_classes = None
        self.tf_idf_sums = None
        self.class_counts = None  # Track number of samples per class

    def fit(self, X: sp.csr_matrix, y: np.ndarray, update=False) -> None:
        """
        Fit the model to data, either from scratch or by updating existing statistics.

        :param X: sp.csr_matrix
            The training data
        :param y: np.ndarray
            The training labels
        :param update: bool
            Whether to update the existing model with new data
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        num_classes, vocab_size = len(unique_classes), X.shape[1]

        if not update:
            # Training from scratch 
            self.unique_classes = unique_classes
            self.class_counts = counts.astype(float)  # Initialize class-wise sample counts
            self.priors = self.class_counts / np.sum(self.class_counts)
            self.log_priors = np.log(self.priors)

            # Compute class-wise TF-IDF sum
            class_matrices = {c: X[y == c] for c in self.unique_classes}
            self.tf_idf_sums = np.array([cls_mat.sum(axis=0) for cls_mat in class_matrices.values()]).squeeze()
        else:
            # Ensure model has been trained before updating
            assert self.unique_classes is not None, "Model must be trained first before updating."

            # Create a mapping for existing classes
            class_index_map = {c: i for i, c in enumerate(self.unique_classes)}

            # Update class counts
            for c, count in zip(unique_classes, counts):
                if c in class_index_map:
                    self.class_counts[class_index_map[c]] += count
                else:
                    raise ValueError("Encountered a new class during update. Re-train from scratch.")

            # Update TF-IDF sums per class efficiently
            for i, c in enumerate(unique_classes):
                class_indices = np.where(y == c)[0]  # Get indices of class `c`
                if class_indices.size > 0:
                    self.tf_idf_sums[class_index_map[c], :] += np.asarray(X[class_indices, :].sum(axis=0)).flatten()  # Update class TF-IDF sums
        # Compute updated class-conditional probabilities
        total_tf_idf_sums = self.tf_idf_sums.sum(axis=1, keepdims=True)
        self.means = (self.tf_idf_sums + self.alpha) / (total_tf_idf_sums + self.alpha * vocab_size)
        self.log_class_conditional_probabilities = np.log(self.means, where=(self.means > 0))

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        """
        Compute class probability estimates for input data.

        :param X: sp.csr_matrix
            The input data (CSR matrix)
        :return: np.ndarray
            Probability estimates of shape (num_samples, num_classes)
        """
        num_samples = X.shape[0]
        num_classes = len(self.unique_classes)

        log_posteriors = np.zeros((num_samples, num_classes))
        log_posteriors += self.log_priors  # Add log priors

        # Compute log posteriors
        for i in range(num_samples):
            row = X[i,:]
            word_indices = row.nonzero()[1] 
            log_posteriors[i, :] += self.log_class_conditional_probabilities[:, word_indices].sum(axis=1)

        # Convert log probabilities to normal probabilities
        probs = np.exp(log_posteriors)
        probs /= np.sum(probs, axis=1, keepdims=True)  # Normalize probabilities
        return probs

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        """
        Predict labels for input data using an optimized approach.

        :param X: sp.csr_matrix
            The input data (CSR matrix)
        :return: np.ndarray
            The predicted labels
        """
        num_samples = X.shape[0]
        num_classes = len(self.unique_classes)

        assert X.shape[1] == self.means.shape[1]

        log_posteriors = np.zeros((num_samples, num_classes))
        log_posteriors += self.log_priors  # Add log priors

        # Compute log posteriors
        for i in range(num_samples):
            word_indices = X.indices[X.indptr[i]:X.indptr[i+1]]  # Nonzero indices in row `i`
            log_posteriors[i, :] += self.log_class_conditional_probabilities[:, word_indices].sum(axis=1)

        # Predict class with the highest log posterior
        return self.unique_classes[np.argmax(log_posteriors, axis=1)]