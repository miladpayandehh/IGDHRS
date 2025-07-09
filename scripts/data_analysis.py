
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

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Load datasets ----------
def load_movielens_100k():
    ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    users = pd.read_csv('data/ml-100k/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    items = pd.read_csv('data/ml-100k/u.item', sep='|', encoding='latin-1', header=None)
    return ratings, users, items

def load_movielens_1m():
    ratings = pd.read_csv('data/ml-1m/ratings.dat', sep='::', engine='python',
                          names=['user_id', 'item_id', 'rating', 'timestamp'])
    users = pd.read_csv('data/ml-1m/users.dat', sep='::', engine='python',
                        names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])
    items = pd.read_csv('data/ml-1m/movies.dat', sep='::', engine='python',
                        names=['item_id', 'title', 'genres'])
    return ratings, users, items

# ---------- Descriptive stats ----------
def dataset_statistics(ratings, users, items, dataset_name):
    print(f"\n--- {dataset_name} ---")
    num_users = users['user_id'].nunique()
    num_items = items['item_id'].nunique()
    num_ratings = len(ratings)
    sparsity = 1 - num_ratings / (num_users * num_items)
    print(f"Users: {num_users}, Items: {num_items}, Ratings: {num_ratings}")
    print(f"Sparsity: {sparsity:.4f}")
    print(f"Min ratings per user: {ratings.groupby('user_id').size().min()}")
    print(f"Median ratings per user: {ratings.groupby('user_id').size().median()}")
    print(f"Avg ratings per user: {ratings.groupby('user_id').size().mean():.2f}")
    print(f"Avg ratings per item: {ratings.groupby('item_id').size().mean():.2f}")
    print(f"Ratings Variance: {ratings['rating'].var():.2f}")
    print(f"Ratings Std Dev: {ratings['rating'].std():.2f}")
    print(f"Male/Female ratio: {users['gender'].value_counts(normalize=True) * 100}")

# ---------- Correlation Heatmap ----------
def correlation_heatmap(features_df, title, output_path):
    corr = features_df.corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ---------- Example execution ----------
if __name__ == "__main__":
    # Load MovieLens 100K
    ratings_100k, users_100k, items_100k = load_movielens_100k()
    dataset_statistics(ratings_100k, users_100k, items_100k, "MovieLens 100K")

    # Load MovieLens 1M
    ratings_1m, users_1m, items_1m = load_movielens_1m()
    dataset_statistics(ratings_1m, users_1m, items_1m, "MovieLens 1M")

    # Auxiliary feature correlations (example subset for demo)
    aux_features_100k = users_100k[['age']]
    aux_features_100k['gender'] = users_100k['gender'].map({'M': 1, 'F': 0})
    aux_features_100k['occupation'] = pd.factorize(users_100k['occupation'])[0]
    correlation_heatmap(aux_features_100k, "Auxiliary Features - ML-100K", "outputs/corr_aux_100k.png")

    # Simulated graph-based features (replace with real features from feature_extraction)
    simulated_graph_feats = pd.DataFrame(np.random.rand(943, 6), columns=['PR', 'DC', 'CC', 'BC', 'LC', 'AND'])
    correlation_heatmap(simulated_graph_feats, "Graph Features - ML-100K", "outputs/corr_graph_100k.png")
