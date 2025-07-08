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

def assign_new_users_to_clusters(new_user_features, ddgcae_model, adj_norm, cluster_centroids):
    """
    Assign new users to clusters based on cosine similarity between their embeddings and cluster centroids.
    Args:
        new_user_features: numpy array (n_new_users x feature_dim)
        ddgcae_model: trained DDGCAE PyTorch model
        adj_norm: normalized adjacency matrix (torch tensor)
        cluster_centroids: numpy array (n_clusters x embedding_dim)
    Returns:
        assigned_clusters: list of cluster indices per user
    """
    ddgcae_model.eval()
    with torch.no_grad():
        features_tensor = torch.FloatTensor(new_user_features)
        embeddings, _ = ddgcae_model(features_tensor, adj_norm)
        embeddings_np = embeddings.cpu().numpy()
    # Cosine similarity between embeddings and cluster centroids
    sim = cosine_similarity(embeddings_np, cluster_centroids)
    assigned_clusters = np.argmax(sim, axis=1)
    return assigned_clusters

def estimate_item_ratings(cluster_labels, cluster_item_ratings, user_idx):
    """
    Estimate user ratings based on cluster averages.
    Args:
        cluster_labels: cluster assignment array (user_idx -> cluster)
        cluster_item_ratings: matrix of average ratings per cluster (clusters x items)
        user_idx: index of the target user
    Returns:
        predicted_ratings: numpy array of predicted ratings for all items for the user
    """
    cluster = cluster_labels[user_idx]
    predicted_ratings = cluster_item_ratings[cluster]
    return predicted_ratings
