# IGDHRS: An intelligent recommender system based on deep denoising graph convolutional autoencoder and learning automata

## Overview

This repository implements the IGDHRS model as described in the associated research article. IGDHRS is a novel recommender system designed to tackle cold-start and sparsity problems by integrating graph-based features with auxiliary user/item data using a Deep Denoising Graph Convolutional Autoencoder (DDGCAE). Additionally, a learning automaton is used for dynamic similarity threshold optimization.

---

## Project Structure

- `src/data_loader.py`: Load and preprocess MovieLens datasets (100K and 1M), including auxiliary features.
- `src/graph_construction.py`: Construct similarity graph based on user features, compute graph structural features (Local Clustering Coefficient, PageRank, Average Neighbor Degree, Clustering Coefficient).
- `src/ddgcae.py`: Implementation of the DDGCAE model for learning robust latent user embeddings.
- `src/clustering.py`: Spectral clustering of learned embeddings to assign users to clusters.
- `src/recommender.py`: Assign new users to clusters, estimate item ratings for users based on cluster averages.
- `src/main.py`: Orchestrates data loading, graph construction, model training, clustering, rating estimation, and evaluation.
- `config.yaml`: Configuration file for dataset paths, model parameters, and training settings.
- `requirements.txt`: Required Python packages.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/miladpayandehh/IGDHRS.git
cd IGDHRS
