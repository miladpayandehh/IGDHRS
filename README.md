# IGDHRS: An Intelligent Graph-based Deep Hybrid Recommender System

This repository contains the official implementation of **IGDHRS**, a novel hybrid recommender system that integrates graph-based user similarity, a Deep Denoising Graph Convolutional Autoencoder (DDGCAE), and an automata-driven adaptive thresholding mechanism. IGDHRS effectively addresses key challenges in recommendation systems, such as **data sparsity** and the **cold-start problem**, by combining auxiliary user/item metadata with graph-derived structural features.

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Outputs](#outputs)
- [Model Components](#model-components)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## 📖 Overview

IGDHRS constructs a **user similarity graph (SG)** using a dynamic threshold \( T_s \), which is automatically optimized via **Learning Automata (LA)** based on feedback from recommendation performance (e.g., RMSE, MAE, precision, recall). The system extracts **graph-based features**, merges them with **auxiliary user/item metadata**, and applies a **DDGCAE** to learn robust representations. Clustering and collaborative filtering are then applied to generate accurate recommendations.

## 🚀 Key Features

- Adaptive similarity threshold tuning via Learning Automata (LA)
- Six comprehensive graph features: PageRank, Degree Centrality, Closeness Centrality, Betweenness Centrality, Load Centrality, and Average Neighbor Degree
- Fusion of structural and auxiliary (demographic/content) features
- Deep GCN-based autoencoder with corruption-aware denoising
- Spectral clustering on graph embeddings
- Handles both **cold-start** and **sparse data** scenarios
- Fully modular, Pythonic, and configurable via `config.yaml`

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/IGDHRS.git
cd IGDHRS
```

2. (Optional) Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

> Note: Dependencies include: `numpy`, `scikit-learn`, `scipy`, `networkx`, `PyYAML`.

## ⚙️ Configuration

All parameters are defined in the `config.yaml` file. Key options include:

```yaml
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
```

## ▶️ Usage

The system is organized into modular components:

| File | Description |
|------|-------------|
| `src/graph_construction.py` | Constructs the user similarity graph using adaptive LA mechanism |
| `src/feature_extraction.py` | Extracts graph-based and auxiliary features |
| `src/ddgcae.py` | Learns latent embeddings using the DDGCAE model |
| `src/clustering.py` | Performs spectral clustering and new user assignment |
| `src/recommender.py` | Generates predicted ratings and recommendation lists |
| `main.py` | Orchestrates the full training and recommendation pipeline |

To run the complete system:

```bash
python main.py --config config.yaml
```

## 📂 Outputs

The system produces the following output files (saved in `/outputs/` directory):

- `embeddings.npy`: Final latent user embeddings
- `clusters.npy`: Cluster assignment of users
- `predicted_ratings.npy`: Full user–item prediction matrix
- `logs/`: Evaluation logs and metadata

## 🧠 Model Components

### Similarity Graph Construction

- Threshold \( T_s \) is adjusted using Learning Automata (LA)
- Actions: `Increase`, `Decrease`, or `Unchanged`
- Feedback: Based on RMSE, MAE, precision, recall
- Variants: LRP, LRI, LRεP schemes supported

### Feature Extraction

- Structural features (x6): PR, DC, CC, BC, LC, AND
- Auxiliary features: One-hot encoded demographics/items
- Combined into a unified feature matrix \( F_t \)

### DDGCAE: Deep Denoising Graph Convolutional Autoencoder

- Inputs: corrupted node features \( \tilde{X} \), graph adjacency matrix \( A \)
- Multi-layer GCN with spectral convolutions
- Closed-form weight updates; no backpropagation
- Robust embedding learning via corruption marginalization

### Clustering & Recommendation

- Spectral clustering on \( Z_2 = 0.5(|Z| + |ZZ^T|) \)
- New users: assigned to nearest cluster using cosine similarity
- Predictions: item scores estimated via cluster-wise rating averages

## 📊 Evaluation

- Supports: RMSE, MAE, Precision@k, Recall@k
- Per-epoch evaluation logging
- Early stopping supported (optional)

## 📚 Citation

If you use this repository in your research, please cite the corresponding paper:

> _[Insert BibTeX or citation information here once published]_

## 🪪 License

This project is released under the [MIT License](LICENSE). You are free to use, modify, and distribute this work with attribution.

## 📬 Contact

For questions, suggestions, or collaboration:

**Milad Payandeh**  
📧 [milad71payandeh@gmail.com](mailto:milad71payandeh@gmail.com)  
🌐 [https://miladpayandeh.com](https://miladpayandeh.com)

---

Thank you for using **IGDHRS**! If you find this repository helpful, please consider starring ⭐ the project.
