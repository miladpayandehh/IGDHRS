
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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from src.graph_construction import build_similarity_graph
from src.recommender import estimate_ratings_from_graph

# Define evaluation metric
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Sample thresholds to test
thresholds = np.linspace(0.005, 0.03, 15)

# Placeholder results
results = []

# Load your dataset
dataset_name = "ml-100k"
ratings_df = pd.read_csv(f"data/{dataset_name}/ratings.csv")

# Main loop over thresholds
for ts in thresholds:
    print(f"Evaluating for T_s = {ts:.4f} ...")
    
    # 1. Build graph with current threshold
    G = build_similarity_graph(ratings_df, threshold=ts)
    
    # 2. Estimate rating matrix (dummy function, replace as needed)
    R_true, R_pred = estimate_ratings_from_graph(G, ratings_df)
    
    # 3. Sparsity = (1 - #edges / max_possible_edges)
    n = len(G.nodes)
    max_edges = n * (n - 1) / 2
    sparsity = 1 - (len(G.edges) / max_edges)
    
    # 4. Evaluate RMSE
    score = rmse(R_true, R_pred)
    
    results.append({
        "T_s": ts,
        "Sparsity": sparsity,
        "RMSE": score
    })

# Convert to DataFrame and save
df_results = pd.DataFrame(results)
df_results.to_csv("outputs/threshold_analysis_results.csv", index=False)

# Plot the RMSE vs. T_s
plt.figure(figsize=(8, 5))
plt.plot(df_results["T_s"], df_results["RMSE"], marker='o', color='teal')
plt.xlabel("Similarity Threshold $T_s$")
plt.ylabel("RMSE")
plt.title("Impact of $T_s$ on RMSE (ML-100K)")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/Fig2.png")
plt.close()
