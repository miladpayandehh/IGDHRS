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
import torch
import yaml
from src.data_loader import load_movielens_data
from src.graph_construction import construct_similarity_graph, compute_graph_features
from src.ddgcae import DDGCAE
from src.clustering import spectral_clustering
from src.recommender import assign_new_users_to_clusters, estimate_item_ratings
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score

def normalize_adj(adj):
    """
    Symmetric normalization of adjacency matrix A_hat = D^-0.5 (A + I) D^-0.5
    """
    adj = adj + np.eye(adj.shape[0])
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt

def run_igdhrs():
    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load data
    R, user_aux_features, item_aux_features = load_movielens_data(
        config['data']['ml100k_path'], config['data']['ml1m_path'], dataset=config['data']['dataset']
    )
    n_users, n_items = R.shape

    # Construct similarity graph based on auxiliary user features
    adj = construct_similarity_graph(user_aux_features, threshold=config['model']['Ts'])

    # Compute graph features
    F_g = compute_graph_features(adj)

    # Combine user features: auxiliary + graph features
    F_t = np.hstack([user_aux_features, F_g])

    # Normalize adjacency
    adj_norm = normalize_adj(adj)
    adj_norm_tensor = torch.FloatTensor(adj_norm)

    # Convert combined features to torch tensor
    features_tensor = torch.FloatTensor(F_t)

    # Initialize model
    ddgcae_model = DDGCAE(
        input_dim=F_t.shape[1],
        hidden_dims=config['model']['hidden_dims'],
        corruption_prob=config['model']['corruption_prob'],
        reg_lambda=config['model']['lambda']
    )

    # Optimizer setup
    optimizer = torch.optim.Adam(ddgcae_model.parameters(), lr=config['training']['lr'])
    epochs = config['training']['epochs']

    # Training loop
    ddgcae_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings, reconstructed = ddgcae_model(features_tensor, adj_norm_tensor)
        loss = ddgcae_model.loss(features_tensor, reconstructed, [layer.weight for layer in ddgcae_model.gc_layers])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch} Loss: {loss.item():.4f}')

    # Extract final embeddings
    ddgcae_model.eval()
    with torch.no_grad():
        embeddings, _ = ddgcae_model(features_tensor, adj_norm_tensor)
    embeddings_np = embeddings.cpu().numpy()

    # Determine number of clusters K via Elbow method (simplified here with fixed K)
    K = config['model']['num_clusters']

    # Clustering
    cluster_labels = spectral_clustering(embeddings_np, K)

    # Compute cluster-item rating averages
    cluster_item_ratings = np.zeros((K, n_items))
    for k in range(K):
        users_in_cluster = np.where(cluster_labels == k)[0]
        if len(users_in_cluster) == 0:
            continue
        cluster_item_ratings[k] = np.mean(R[users_in_cluster], axis=0)

    # Estimate ratings for each user based on cluster averages
    predicted_ratings = np.zeros_like(R)
    for u in range(n_users):
        predicted_ratings[u] = cluster_item_ratings[cluster_labels[u]]

    # Evaluation
    # Here, evaluate with RMSE and MAE on test set or full matrix (simplified)
    rmse = np.sqrt(mean_squared_error(R.flatten(), predicted_ratings.flatten()))
    mae = mean_absolute_error(R.flatten(), predicted_ratings.flatten())
    print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}')

if __name__ == '__main__':
    run_igdhrs()
