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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def build_user_cluster_matrix(user_clusters, num_users, num_clusters):
    """
    Create the UC matrix showing each user's cluster assignment.

    Args:
        user_clusters (np.ndarray): Cluster index for each user.
        num_users (int): Total number of users.
        num_clusters (int): Total number of clusters.

    Returns:
        np.ndarray: Binary UC matrix of shape (num_users, num_clusters).
    """
    UC = np.zeros((num_users, num_clusters))
    for u, c in enumerate(user_clusters):
        UC[u, c] = 1
    return UC


def build_cluster_item_matrix(R, UC, num_clusters):
    """
    Create the CI matrix containing average ratings for each item in each cluster.

    Args:
        R (np.ndarray): User-item rating matrix (n_users x n_items).
        UC (np.ndarray): User-cluster matrix (n_users x num_clusters).
        num_clusters (int): Total number of clusters.

    Returns:
        np.ndarray: Cluster-item rating matrix (num_clusters x n_items).
    """
    num_items = R.shape[1]
    CI = np.zeros((num_clusters, num_items))

    for c in range(num_clusters):
        users_in_cluster = np.where(UC[:, c] == 1)[0]
        if len(users_in_cluster) == 0:
            continue

        cluster_ratings = R[users_in_cluster]
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_ratings = np.true_divide(cluster_ratings.sum(axis=0), (cluster_ratings != 0).sum(axis=0))
            avg_ratings[np.isnan(avg_ratings)] = 0.0
        CI[c] = avg_ratings

    return CI


def assign_new_user_to_cluster(user_aux_vector, cluster_centroids):
    """
    Assign a new user to the nearest cluster based on cosine similarity.

    Args:
        user_aux_vector (np.ndarray): 1D feature vector of the new user (after projection).
        cluster_centroids (np.ndarray): Cluster center vectors.

    Returns:
        int: Index of the most similar cluster.
    """
    user_vec = normalize(user_aux_vector.reshape(1, -1))
    centroids_norm = normalize(cluster_centroids)
    similarities = cosine_similarity(user_vec, centroids_norm)
    return np.argmax(similarities)


def rank_items_for_cluster(cluster_index, CI, top_k=10):
    """
    Return top-k items ranked for a given cluster based on cluster-item ratings.

    Args:
        cluster_index (int): Index of the target cluster.
        CI (np.ndarray): Cluster-item matrix.
        top_k (int): Number of top items to return.

    Returns:
        np.ndarray: Indices of top-k recommended items.
    """
    scores = CI[cluster_index]
    top_items = np.argsort(-scores)[:top_k]
    return top_items


def rank_new_items(new_item_features, existing_item_features, top_k=10):
    """
    Rank new items based on their similarity to existing items using content features.

    Args:
        new_item_features (np.ndarray): Features of new items (m x d).
        existing_item_features (np.ndarray): Features of existing items (n x d).
        top_k (int): Number of top similar existing items to find for each new item.

    Returns:
        list: List of top-k indices for each new item.
    """
    similarities = cosine_similarity(new_item_features, existing_item_features)
    rankings = [np.argsort(-row)[:top_k] for row in similarities]
    return rankings
