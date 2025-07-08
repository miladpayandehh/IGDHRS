# MIT License
# 
# Copyright (c) 2025 Milad Payandeh
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

def estimate_rating_matrix(UC, CI):
    """
    Estimate the final predicted user-item rating matrix R'.

    Args:
        UC (np.ndarray): User-cluster matrix (n_users x n_clusters)
        CI (np.ndarray): Cluster-item matrix (n_clusters x n_items)

    Returns:
        np.ndarray: Predicted rating matrix R' (n_users x n_items)
    """
    # Multiply UC and CI to get predicted ratings for each user from their cluster
    R_hat = np.dot(UC, CI)
    return R_hat


def recommend_items(R_hat, original_R, top_k=10):
    """
    Generate top-k recommendations for each user.

    Args:
        R_hat (np.ndarray): Predicted rating matrix (n_users x n_items)
        original_R (np.ndarray): Original user-item rating matrix (n_users x n_items), with zeros for unrated
        top_k (int): Number of top items to recommend

    Returns:
        dict: Dictionary where keys are user indices and values are lists of recommended item indices
    """
    n_users, n_items = R_hat.shape
    recommendations = {}

    for u in range(n_users):
        # Only consider items the user hasn't rated yet
        unrated_items = np.where(original_R[u] == 0)[0]
        predicted_scores = R_hat[u, unrated_items]

        # Rank the unrated items based on predicted scores
        top_indices = np.argsort(-predicted_scores)[:top_k]
        recommended_items = unrated_items[top_indices]

        recommendations[u] = recommended_items.tolist()

    return recommendations
