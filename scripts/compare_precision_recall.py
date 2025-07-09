
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

import matplotlib.pyplot as plt
import numpy as np

# Models
models = ["GSHF", "UPCSim", "GDPE", "PP/NP", "GHRS", "IGDHRS"]

# Precision values
precision_100k = [0.6592, 0.6416, 0.7130, 0.6466, 0.7710, 0.7703]
precision_1m    = [0.6173, 0.6048, 0.6910, 0.6097, 0.7920, 0.7911]

# Recall values
recall_100k = [0.7081, 0.6866, 0.7560, 0.7010, 0.7993, 0.8001]
recall_1m    = [0.6614, 0.6280, 0.7380, 0.6522, 0.8381, 0.8398]

x = np.arange(len(models))
width = 0.18

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Precision
axs[0].bar(x - width, precision_100k, width, label="ML-100K")
axs[0].bar(x,         precision_1m,   width, label="ML-1M")
axs[0].set_ylabel("Precision")
axs[0].set_title("Precision Comparison Across Recommender Systems")
axs[0].legend()
axs[0].grid(True, linestyle="--", alpha=0.5)

# Recall
axs[1].bar(x - width, recall_100k, width, label="ML-100K")
axs[1].bar(x,         recall_1m,   width, label="ML-1M")
axs[1].set_ylabel("Recall")
axs[1].set_title("Recall Comparison Across Recommender Systems")
axs[1].legend()
axs[1].grid(True, linestyle="--", alpha=0.5)

# Labels
plt.xticks(x, models)
plt.xlabel("Recommender System")
plt.tight_layout()
plt.savefig("outputs/Fig4.png")
plt.close()
