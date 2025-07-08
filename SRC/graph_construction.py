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

def construct_similarity_graph(user_features, method='cosine', threshold=0.01):
    """
    Construct similarity graph based on user auxiliary features or latent embeddings.
    Args:
        user_features: numpy array (n_users x n_features)
        method: similarity metric ('cosine' supported)
        threshold: similarity threshold Ts for edge inclusion
    Returns:
        adjacency_matrix: sparse or dense numpy array (n_users x n_users)
    """
    if method == 'cosine':
        sim_matrix = cosine_similarity(user_features)
        # Apply threshold Ts to build adjacency matrix
        adjacency = (sim_matrix >= threshold).astype(float)
        np.fill_diagonal(adjacency, 1)  # self loops
        return adjacency
    else:
        raise ValueError('Unsupported similarity method')

def compute_graph_features(adj):
    """
    Compute graph structural features:
    LC, PR, AND, CC as in paper.
    Args:
        adj: adjacency matrix (numpy array)
    Returns:
        F_g: graph feature matrix (n_users x 4)
    """
    n = adj.shape[0]
    # Local clustering coefficient (LC)
    LC = np.zeros(n)
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            LC[i] = 0.0
        else:
            subgraph = adj[np.ix_(neighbors, neighbors)]
            possible_edges = k * (k - 1) / 2
            actual_edges = np.sum(subgraph) / 2  # since symmetric
            LC[i] = actual_edges / possible_edges
    # PageRank (PR)
    PR = pagerank(adj)
    # Average neighbor degree (AND)
    degrees = adj.sum(axis=1)
    AND = np.zeros(n)
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) > 0:
            AND[i] = np.mean(degrees[neighbors])
        else:
            AND[i] = 0
    # Clustering coefficient (CC) same as LC in undirected graphs
    CC = LC.copy()
    # Combine features into matrix
    F_g = np.vstack([LC, PR, AND, CC]).T
    return F_g

def pagerank(adj, alpha=0.85, max_iter=100, tol=1e-6):
    """
    Compute PageRank for nodes.
    Args:
        adj: adjacency matrix (numpy array)
        alpha: damping factor
        max_iter: max iterations
        tol: convergence threshold
    Returns:
        pr: numpy array of PageRank values
    """
    n = adj.shape[0]
    adj = adj.astype(float)
    row_sum = adj.sum(axis=1)
    row_sum[row_sum == 0] = 1
    M = adj / row_sum[:, None]
    pr = np.ones(n) / n
    for _ in range(max_iter):
        pr_new = (1 - alpha) / n + alpha * M.T.dot(pr)
        if np.linalg.norm(pr_new - pr, 1) < tol:
            break
        pr = pr_new
    return pr
