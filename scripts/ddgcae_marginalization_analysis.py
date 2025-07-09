
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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.ddgcae import train_ddgcae
from src.clustering import perform_clustering
from src.recommender import estimate_ratings_from_clusters
from utils.evaluation import precision_at_k, recall_at_k

# Load dataset
dataset_name = "ml-100k"
ratings_path = f"data/{dataset_name}/ratings.csv"

# Define evaluation metrics
def evaluate(y_true, y_pred, k=10):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    prec = precision_at_k(y_true, y_pred, k)
    rec = recall_at_k(y_true, y_pred, k)
    return rmse, mae, prec, rec

# Run both versions: with and without marginalization
results = {"With Marginalization": {}, "Without Marginalization": {}}

for label, marginalized in zip(results.keys(), [True, False]):
    print(f"\nRunning DDGCAE with marginalized = {marginalized}")

    # 1. Train DDGCAE
    embeddings = train_ddgcae(ratings_path, 
                              corruption_prob=0.43, 
                              num_layers=4, 
                              lambda_reg=1e-5, 
                              marginalized=marginalized, 
                              iterations=4)
    
    # 2. Clustering
    cluster_labels = perform_clustering(embeddings, num_clusters=8)

    # 3. Rating estimation
    y_true, y_pred = estimate_ratings_from_clusters(cluster_labels, ratings_path)
    
    # 4. Evaluation
    rmse, mae, precision, recall = evaluate(y_true, y_pred)
    results[label] = {
        "RMSE": rmse,
        "MAE": mae,
        "Precision": precision,
        "Recall": recall
    }

# Plot results
metrics = ["RMSE", "MAE", "Precision", "Recall"]
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
vals1 = [results["With Marginalization"][m] for m in metrics]
vals2 = [results["Without Marginalization"][m] for m in metrics]

ax.bar(x - width/2, vals1, width, label="With Marginalization", color='steelblue')
ax.bar(x + width/2, vals2, width, label="Without Marginalization", color='darkorange')

ax.set_ylabel("Score")
ax.set_title("Model performance with/without marginalization")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/Fig3.png")
plt.close()
