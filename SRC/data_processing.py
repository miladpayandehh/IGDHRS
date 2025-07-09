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
import matplotlib.pyplot as plt
import seaborn as sns

def load_movielens_data(dataset='100k'):
    """
    Load the MovieLens 100K or 1M dataset
    Assumption: CSV files contain user, item, and rating information
    """
    base_path = f'data/ml-{dataset}/'  # Dataset path (should be configured)
    ratings = pd.read_csv(base_path + 'ratings.csv')  # userId, movieId, rating, timestamp
    users = pd.read_csv(base_path + 'users.csv')      # userId, age, gender, occupation, zip_code (depending on dataset)
    items = pd.read_csv(base_path + 'movies.csv')     # movieId, title, genres
    
    return ratings, users, items

def descriptive_stats(ratings, users, items):
    stats = {}
    stats['Number of ratings'] = len(ratings)
    stats['Number of users'] = ratings['userId'].nunique()
    stats['Number of items (movies)'] = ratings['movieId'].nunique()
    
    # Minimum, mean, and median number of ratings per user
    user_rating_counts = ratings.groupby('userId').size()
    stats['Minimum ratings per user'] = user_rating_counts.min()
    stats['Median ratings per user'] = user_rating_counts.median()
    stats['Average ratings per user'] = user_rating_counts.mean()
    
    # Average ratings per item
    item_rating_counts = ratings.groupby('movieId').size()
    stats['Average ratings per item'] = item_rating_counts.mean()
    
    # Gender ratio
    if 'gender' in users.columns:
        gender_counts = users['gender'].value_counts(normalize=True)
        stats['Male / Female ratio'] = f"{gender_counts.get('M',0)*100:.1f}% / {gender_counts.get('F',0)*100:.1f}%"
    else:
        stats['Male / Female ratio'] = 'N/A'
    
    # Variance and standard deviation of ratings
    stats['Ratings variance'] = ratings['rating'].var()
    stats['Ratings standard deviation'] = ratings['rating'].std()
    
    # Rating scale range
    stats['Rating scale'] = f"{ratings['rating'].min()} to {ratings['rating'].max()} (integer)"
    
    return stats

def print_stats_table(stats_100k, stats_1m):
    import tabulate
    headers = ['Statistic', 'MovieLens 100K', 'MovieLens 1M']
    rows = []
    for key in stats_100k.keys():
        rows.append([key, stats_100k[key], stats_1m.get(key, 'N/A')])
    print(tabulate.tabulate(rows, headers=headers, tablefmt='github'))

def compute_and_plot_correlations(features_g_df, features_s_df, dataset_name, ts_value):
    """
    features_g_df: DataFrame of graph-based features (e.g., LC, PR, AND, CC)
    features_s_df: DataFrame of auxiliary user features (age, gender, occupation converted to numeric)
    """
    # Correlation matrices for two feature sets
    corr_g = features_g_df.corr()
    corr_s = features_s_df.corr()
    
    fig, axs = plt.subplots(1, 2, figsize=(14,6))
    
    sns.heatmap(corr_g, annot=True, fmt=".2f", cmap='coolwarm', ax=axs[0])
    axs[0].set_title(f"Graph-based features correlation ({dataset_name})")
    
    sns.heatmap(corr_s, annot=True, fmt=".2f", cmap='coolwarm', ax=axs[1])
    axs[1].set_title(f"Auxiliary user features correlation ({dataset_name})")
    
    plt.suptitle(f"Correlation analysis for {dataset_name} dataset with $T_s = {ts_value}$")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"correlation_{dataset_name}.png")
    plt.show()

# Example usage:

# Load datasets (file paths must be configured)
ratings_100k, users_100k, items_100k = load_movielens_data('100k')
ratings_1m, users_1m, items_1m = load_movielens_data('1m')

# Extract descriptive statistics
stats_100k = descriptive_stats(ratings_100k, users_100k, items_100k)
stats_1m = descriptive_stats(ratings_1m, users_1m, items_1m)

print_stats_table(stats_100k, stats_1m)

# Assume the graph-based and auxiliary user features DataFrames are as follows:
# These DataFrames should be extracted from the model or prior steps
# Example:
features_g_100k = pd.DataFrame({
    'LC': np.random.rand(len(users_100k)),
    'PR': np.random.rand(len(users_100k)),
    'AND': np.random.rand(len(users_100k)),
    'CC': np.random.rand(len(users_100k))
})
features_s_100k = users_100k[['age']].copy()
features_s_100k['gender'] = users_100k['gender'].map({'M': 1, 'F': 0})
# Assume occupation is numeric (or converted from categorical)
features_s_100k['occupation'] = pd.factorize(users_100k['occupation'])[0]

features_g_1m = pd.DataFrame({
    'LC': np.random.rand(len(users_1m)),
    'PR': np.random.rand(len(users_1m)),
    'AND': np.random.rand(len(users_1m)),
    'CC': np.random.rand(len(users_1m))
})
features_s_1m = users_1m[['age']].copy()
features_s_1m['gender'] = users_1m['gender'].map({'M': 1, 'F': 0})
features_s_1m['occupation'] = pd.factorize(users_1m['occupation'])[0]

# Plot and save correlation heatmaps
compute_and_plot_correlations(features_g_100k, features_s_100k, 'ML-100K', 0.018)
compute_and_plot_correlations(features_g_1m, features_s_1m, 'ML-1M', 0.011)
