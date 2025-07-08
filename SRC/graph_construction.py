# MIT License
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
import networkx as nx
from tqdm import tqdm
import random

# -----------------------------------------
# Function: Compute pairwise user similarity with threshold
# -----------------------------------------
def compute_common_similarity_matrix(rating_matrix, threshold):
    """
    Constructs the similarity graph based on common rated items and similarity threshold (Ts).
    Two users are connected if the percentage of similar ratings on commonly rated items exceeds the threshold.
    """
    num_users = rating_matrix.shape[0]
    graph = nx.Graph()

    for u in range(num_users):
        graph.add_node(u)

    for u in tqdm(range(num_users), desc="Building Similarity Graph"):
        for v in range(u + 1, num_users):
            u_ratings = rating_matrix[u]
            v_ratings = rating_matrix[v]
            common_items = np.where((u_ratings > 0) & (v_ratings > 0))[0]

            if len(common_items) == 0:
                continue

            diff = np.abs(u_ratings[common_items] - v_ratings[common_items])
            similar_count = np.sum(diff <= 1)
            similarity_ratio = similar_count / len(common_items)

            if similarity_ratio >= threshold:
                graph.add_edge(u, v)

    return graph

# -----------------------------------------
# Function: Evaluate graph performance (Mock)
# -----------------------------------------
def evaluate_performance(graph, metric='rmse', prev_score=1.0):
    """
    Mock function for reward signal. In real cases, you should evaluate RMSE, MAE, Precision, or Recall.
    """
    # Simulate slight performance improvement or degradation randomly
    new_score = prev_score + np.random.uniform(-0.01, 0.01)

    # For RMSE and MAE: reward if performance improves (score decreases)
    if metric in ['rmse', 'mae']:
        return 1 if new_score < prev_score else 0, new_score

    # For Precision and Recall: reward if performance improves (score increases)
    elif metric in ['precision', 'recall']:
        return 1 if new_score > prev_score else 0, new_score

    return 0, new_score

# -----------------------------------------
# Main Function: Learning Automata for optimal threshold Ts
# -----------------------------------------
def learning_automaton_graph_construction(rating_matrix,
                                          metric='rmse',
                                          max_iterations=50,
                                          initial_Ts=0.01,
                                          Ts_min=0.001,
                                          Ts_max=0.03,
                                          a=0.1,
                                          b=0.05,
                                          mode='LRP'):
    """
    Constructs a user similarity graph while dynamically optimizing the similarity threshold (Ts)
    using a Learning Automaton (LA) with reward/penalty updates.
    """

    actions = ['increase', 'decrease', 'unchanged']
    r = len(actions)
    P = np.ones(r) / r  # initial equal probabilities
    Ts = initial_Ts
    prev_score = 1.0
    best_graph = None
    best_score = float('inf') if metric in ['rmse', 'mae'] else -float('inf')
    best_Ts = Ts

    for iteration in range(max_iterations):
        # Select action based on probability distribution
        action_index = np.random.choice(range(r), p=P)
        action = actions[action_index]

        # Apply action to adjust Ts
        if action == 'increase':
            Ts += 0.001
        elif action == 'decrease':
            Ts -= 0.001
        Ts = np.clip(Ts, Ts_min, Ts_max)

        # Build new graph based on current Ts
        graph = compute_common_similarity_matrix(rating_matrix, Ts)

        # Evaluate performance (mock, replace in real system)
        beta, new_score = evaluate_performance(graph, metric=metric, prev_score=prev_score)

        # Update probabilities using reward or penalty rule
        if beta == 1:  # Reward
            for i in range(r):
                if i == action_index:
                    P[i] = P[i] + a * (1 - P[i])
                else:
                    P[i] = (1 - a) * P[i]
        else:  # Penalty
            if mode == 'LRI':
                pass  # No update in LRI mode
            else:
                for i in range(r):
                    if i == action_index:
                        P[i] = (1 - b) * P[i]
                    else:
                        P[i] = b / (r - 1) + (1 - b) * P[i]

        # Normalize probabilities
        P = P / np.sum(P)
        prev_score = new_score

        # Store best graph and score
        if ((metric in ['rmse', 'mae'] and new_score < best_score) or
            (metric in ['precision', 'recall'] and new_score > best_score)):
            best_score = new_score
            best_graph = graph.copy()
            best_Ts = Ts

        print(f"Iter {iteration+1}: Action={action}, Ts={Ts:.4f}, Score={new_score:.4f}, Best_Ts={best_Ts:.4f}")

    return best_graph, best_Ts
