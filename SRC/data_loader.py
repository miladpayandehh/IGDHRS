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
import scipy.sparse as sp
import yaml

# Load MovieLens datasets and auxiliary features, preprocess features and ratings

def load_movielens_data(path_ml100k, path_ml1m, dataset='ml100k'):
    """
    Load MovieLens data for specified dataset.
    Args:
        path_ml100k: path to ML-100K files
        path_ml1m: path to ML-1M files
        dataset: 'ml100k' or 'ml1m'
    Returns:
        R: user-item rating matrix (numpy array)
        user_features: auxiliary user features matrix after one-hot encoding
        item_features: auxiliary item features matrix (one-hot or processed)
    """
    if dataset.lower() == 'ml100k':
        # Implement loading ML-100K dataset from path_ml100k
        # user IDs, item IDs are mapped to 0-based indices
        # load ratings and build R matrix
        # load user attributes (gender, age, occupation), one-hot encode them
        # load item attributes (genre, release date bucket), one-hot encode or numerical normalize
        # Return R, user_features, item_features
        pass
    elif dataset.lower() == 'ml1m':
        # Similarly for ML-1M dataset from path_ml1m
        pass
    else:
        raise ValueError('Unsupported dataset')
    # Placeholder return
    return None, None, None

def one_hot_encode_features(features, categories_list):
    """
    One-hot encode categorical features.
    Args:
        features: 2D array with categorical values (n_samples x n_features)
        categories_list: list of arrays or lists specifying unique categories per feature
    Returns:
        encoded: 2D numpy array with concatenated one-hot vectors per feature
    """
    encoded_features = []
    for col_idx, categories in enumerate(categories_list):
        feature_col = features[:, col_idx]
        one_hot = np.zeros((len(feature_col), len(categories)), dtype=np.float32)
        for i, val in enumerate(feature_col):
            if val in categories:
                one_hot[i, categories.index(val)] = 1
        encoded_features.append(one_hot)
    return np.concatenate(encoded_features, axis=1)
