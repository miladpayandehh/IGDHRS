# IGDHRS: Intelligent Graph-based Deep Hybrid Recommender System

This repository contains the implementation of the IGDHRS model, a novel hybrid recommender system integrating graph-based user similarity, deep denoising graph convolutional autoencoder (DDGCAE), and automata-based adaptive thresholding for similarity graph construction. IGDHRS is designed to address sparsity and cold-start problems by combining auxiliary user/item features with graph structural information.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output](#output)
- [Model Details](#model-details)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

---

## Project Overview

IGDHRS proposes a recommendation framework that constructs a user similarity graph (SG) with an adaptive similarity threshold dynamically tuned by a Learning Automata (LA). This balances graph sparsity and connectivity to optimize recommendation accuracy.

Key contributions include:

- **Similarity Graph Construction:** Uses LA to adapt the minimum similarity threshold \( T_s \) based on feedback from system performance (RMSE, MAE, precision, recall).
- **Graph Feature Extraction:** Extracts six graph-based node features: PageRank, Degree Centrality, Closeness Centrality, Betweenness Centrality, Load Centrality, and Average Neighbor Degree.
- **Feature Fusion:** Combines auxiliary user features (one-hot encoded) with graph features to form comprehensive node feature matrix.
- **Deep Denoising Graph Convolutional Autoencoder (DDGCAE):** Learns robust node embeddings by reconstructing original features from corrupted inputs through spectral graph convolutions, with closed-form weight updates.
- **Spectral Clustering:** Performs clustering on the learned embeddings to group users with similar preferences.
- **Recommendation Generation:** Predicts item ratings for each user by averaging ratings within assigned clusters, enabling cold-start handling.

---

## Features

- Adaptive thresholding for user similarity graph via Learning Automata
- Six comprehensive graph-based node features for enhanced representation
- Deep GCN-based autoencoder for robust latent embeddings
- Clustering based on learned embeddings for effective collaborative filtering
- Handles cold-start problem via auxiliary feature integration
- Modular and configurable Python implementation

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/IGDHRS.git
cd IGDHRS

##Create a Python virtual environment (optional but recommended):

python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

## Install required packages:

pip install -r requirements.txt

Note: Key dependencies include numpy, scipy, scikit-learn, networkx, and PyYAML.

##Configuration

Adjust model and runtime parameters via config.yaml. Key configurable options include:

Similarity threshold initial value and range
Automata learning rates and type (LRP, LRεP, LRI)
DDGCAE corruption probability, layers, and regularization
Number of clusters for spectral clustering
Paths to input data files

Example snippet from config.yaml:
graph:
  similarity_threshold_init: 0.015
  threshold_range: [0.001, 0.03]
  automata_type: LRP
  reward_learning_rate: 0.1
  penalty_learning_rate: 0.1

ddgcae:
  corruption_probability: 0.3
  layers: 3
  regularization_lambda: 0.01

clustering:
  num_clusters: 8

##Usage

The project contains modular Python scripts:

src/graph_construction.py: Constructs the similarity graph using LA to adaptively update the similarity threshold.
src/feature_extraction.py: Extracts graph-based and auxiliary features.
src/ddgcae.py: Implements the deep denoising graph convolutional autoencoder for node embedding learning.
src/clustering.py: Performs spectral clustering on learned embeddings and assigns new users to clusters.
src/recommender.py: Generates the final rating prediction matrix and recommendation lists.
main.py: Coordinates the full pipeline.

To run the complete pipeline, simply execute: python main.py --config config.yaml

##Output

Learned user embeddings saved in outputs/embeddings.npy.
Cluster assignments saved in outputs/clusters.npy.
Predicted rating matrix saved in outputs/predicted_ratings.npy.
Logs and intermediate files saved in outputs/logs/.

