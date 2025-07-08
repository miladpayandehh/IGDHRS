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
from utils import load_data, evaluate_recommendations
from graph_construction import construct_similarity_graph
from feature_engineering import extract_graph_features, encode_auxiliary_features
from ddgcae import train_ddgcae
from clustering import assign_clusters, compute_cluster_item_matrix
from recommender import estimate_rating_matrix, recommend_items

def main():
    # Step 1: Load raw data (user-item matrix and auxiliary user/item info)
    R, F_u, F_i = load_data()

    # Step 2: Construct similarity graph using LA-optimized threshold
    adj_matrix = construct_similarity_graph(R)

    # Step 3: Extract graph-based features
    F_g = extract_graph_features(adj_matrix)

    # Step 4: Encode auxiliary user features (e.g., age, gender, occupation)
    F_s = encode_auxiliary_features(F_u)

    # Step 5: Concatenate features (F_t = [F_g, F_s])
    F_t = np.concatenate([F_g, F_s], axis=1)

    # Step 6: Train DDGCAE model and obtain latent user representations
    Z, model_weights = train_ddgcae(F_t, adj_matrix)

    # Step 7: Cluster users based on learned representations
    UC, cluster_centroids = assign_clusters(Z)

    # Step 8: Compute cluster-item average rating matrix (CI)
    CI = compute_cluster_item_matrix(R, UC)

    # Step 9: Estimate final rating matrix R' using UC × CI
    R_hat = estimate_rating_matrix(UC, CI)

    # Step 10: Generate top-k recommendations for all users
    top_k = 10
    recommendations = recommend_items(R_hat, R, top_k=top_k)

    # Step 11: Evaluate the recommendation performance
    evaluate_recommendations(recommendations, R)

if __name__ == "__main__":
    main()

