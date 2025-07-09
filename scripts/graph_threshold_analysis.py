
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
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score
from math import sqrt
import random

# --- Initial settings ---
DATASET = "ml-100k"  # or "ml-1m"
METRIC = "rmse"  # one of: rmse, mae, precision, recall
MAX_ITER = 500
TOP_K = 10

# --- Learning Automata parameters ---
n_actions = 100
actions = np.linspace(0.001, 0.1, n_actions)
P = np.ones(n_actions) / n_actions
alpha = 0.05  # reward
beta = 0.02   # penalty

# --- Load ratings data ---
def load_ratings():
    # Note: Replace this path with the actual dataset path
    path = f"datasets/{DATASET}/ratings.csv"
    df = pd.read_csv(path)
    return df.pivot(index='userId', columns='itemId', values='rating').fillna(0).values

# --- Compute similarity matrix ---
def build_similarity(rating_matrix, threshold):
    sim = cosine_similarity(rating_matrix)
    np.fill_diagonal(sim, 0)
    sim[sim < threshold] = 0
    return sim

# --- Predict ratings ---
def predict(sim_matrix, rating_matrix):
    weighted_sum = sim_matrix @ rating_matrix
    norm = np.abs(sim_matrix).sum(axis=1, keepdims=True)
    norm[norm == 0] = 1e-8
    return weighted_sum / norm

# --- Evaluation function ---
def evaluate(y_true, y_pred, metric):
    mask = y_true > 0
    if metric == "rmse":
        return sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    elif metric == "mae":
        return mean_absolute_error(y_true[mask], y_pred[mask])
    elif metric == "precision":
        return precision_score(y_true[mask] >= 4, y_pred[mask] >= 4, zero_division=0)
    elif metric == "recall":
        return recall_score(y_true[mask] >= 4, y_pred[mask] >= 4, zero_division=0)
    else:
        raise ValueError("Unknown metric")

# --- Learning Automata optimization loop ---
def optimize_threshold(rating_matrix, metric):
    best_score = float("inf") if metric in ["rmse", "mae"] else 0
    best_Ts = None
    history = []

    for _ in range(MAX_ITER):
        action_idx = np.random.choice(range(n_actions), p=P)
        Ts = actions[action_idx]

        sim = build_similarity(rating_matrix, Ts)
        pred = predict(sim, rating_matrix)
        score = evaluate(rating_matrix, pred, metric)
        history.append((Ts, score))

        # Positive or negative feedback
        improve = score < best_score if metric in ["rmse", "mae"] else score > best_score

        if improve:
            best_score = score
            best_Ts = Ts
            P[action_idx] += alpha * (1 - P[action_idx])
        else:
            P[action_idx] -= beta * P[action_idx]
        
        P[:] = P / P.sum()

    return best_Ts, best_score, history

# --- Main execution ---
if __name__ == "__main__":
    print(f"Dataset: {DATASET}, Feedback: {METRIC}")
    rating_matrix = load_ratings()
    Ts_opt, score_opt, logs = optimize_threshold(rating_matrix, METRIC)

    print(f"Optimal T_s = {Ts_opt:.4f}")
    print(f"Best {METRIC.upper()} = {score_opt:.4f}")
