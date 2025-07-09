
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import yaml

def load_predictions(prediction_path):
    return np.load(prediction_path)

def load_test_data(test_path):
    df = pd.read_csv(test_path)
    return df

def evaluate_rmse_mae(test_df, predicted_matrix):
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        u, i, r = int(row['user']), int(row['item']), float(row['rating'])
        if u < predicted_matrix.shape[0] and i < predicted_matrix.shape[1]:
            y_true.append(r)
            y_pred.append(predicted_matrix[u, i])
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

def precision_recall_at_k(test_df, predicted_matrix, k=10):
    user_rated_items = defaultdict(set)
    for _, row in test_df.iterrows():
        user_rated_items[int(row['user'])].add(int(row['item']))

    precisions, recalls = [], []

    for user in range(predicted_matrix.shape[0]):
        user_preds = predicted_matrix[user]
        top_k_items = np.argsort(user_preds)[::-1][:k]
        true_items = user_rated_items[user]

        if len(true_items) == 0:
            continue

        hits = len(set(top_k_items) & true_items)
        precisions.append(hits / k)
        recalls.append(hits / len(true_items))

    return np.mean(precisions), np.mean(recalls)

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    prediction_path = config["evaluation"]["predicted_ratings"]
    test_path = config["evaluation"]["test_data"]
    k = config["evaluation"].get("top_k", 10)

    predicted_matrix = load_predictions(prediction_path)
    test_df = load_test_data(test_path)

    rmse, mae = evaluate_rmse_mae(test_df, predicted_matrix)
    precision, recall = precision_recall_at_k(test_df, predicted_matrix, k=k)

    print("=== Evaluation Results ===")
    print(f"RMSE     : {rmse:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}   : {recall:.4f}")

    # Save to log file
    with open("outputs/evaluation_metrics.txt", "w") as f:
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"Precision@{k}: {precision:.4f}\n")
        f.write(f"Recall@{k}: {recall:.4f}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
