# MIT License
# Copyright (c) 2025 Milad Payandeh
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# https://opensource.org/licenses/MIT

import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix

class DDGCAE:
    def __init__(self, input_dim, hidden_dim, num_layers=4, corruption_prob=0.43, reg_lambda=1e-5, m_iter=4, num_clusters=8):
        """
        Initialize DDGCAE model parameters.

        Args:
            input_dim (int): Dimension of input feature vectors.
            hidden_dim (int): Dimension of hidden representations.
            num_layers (int): Number of stacked GCN layers.
            corruption_prob (float): Probability of corruption.
            reg_lambda (float): Regularization coefficient.
            m_iter (int): Number of marginalization iterations.
            num_clusters (int): Number of clusters for spectral clustering.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.p = corruption_prob
        self.lambda_ = reg_lambda
        self.m_iter = m_iter
        self.K = num_clusters
        self.weights = []

    def _normalize_adj(self, A):
        """Compute symmetric normalized adjacency matrix."""
        A_hat = A + np.eye(A.shape[0])
        D_hat = np.diag(1.0 / np.sqrt(np.sum(A_hat, axis=1)))
        return D_hat @ A_hat @ D_hat

    def _corrupt_input(self, X):
        """Randomly mask elements of X according to corruption probability."""
        mask = np.random.binomial(1, 1 - self.p, size=X.shape)
        return X * mask

    def _compute_weights_closed_form(self, X, A_hat):
        """
        Compute optimal weight matrix using closed-form marginalization.

        Args:
            X (np.ndarray): Original feature matrix.
            A_hat (np.ndarray): Normalized adjacency matrix.

        Returns:
            np.ndarray: Weight matrix W.
        """
        S_p = X.T @ A_hat @ X
        S_q = X.T @ A_hat.T @ A_hat @ X
        E_P = S_p * (1 - self.p)
        E_Q = S_q * (1 - self.p)**2 + self.lambda_ * np.eye(S_q.shape[0])
        W = np.linalg.solve(E_Q, E_P.T).T
        return W

    def fit_transform(self, X, A):
        """
        Train DDGCAE and obtain final node embeddings.

        Args:
            X (np.ndarray): Input feature matrix (n x d).
            A (np.ndarray): Adjacency matrix (n x n).

        Returns:
            np.ndarray: Node embeddings after final GCN layer.
        """
        A_hat = self._normalize_adj(A)
        Z = X.copy()

        for layer in range(self.num_layers):
            Z_c_list = [self._corrupt_input(Z) for _ in range(self.m_iter)]
            Z_c_avg = np.mean(Z_c_list, axis=0)
            W = self._compute_weights_closed_form(Z_c_avg, A_hat)
            Z = A_hat @ Z @ W
            self.weights.append(W)

        return Z

    def cluster(self, Z):
        """
        Apply spectral clustering on final latent representation.

        Args:
            Z (np.ndarray): Final node embedding matrix.

        Returns:
            np.ndarray: Cluster assignments for each node.
        """
        Z_linear = Z @ Z.T
        Z_sym = 0.5 * (np.abs(Z) + np.abs(Z_linear))
        clustering = SpectralClustering(n_clusters=self.K, affinity='precomputed', assign_labels='kmeans', random_state=0)
        return clustering.fit_predict(Z_sym)

    def get_weights(self):
        """Return list of learned weight matrices."""
        return self.weights
