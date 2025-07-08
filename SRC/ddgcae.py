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
# SOFTWARE

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Single graph convolutional layer based on Kipf and Welling (2017).
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, input, adj_norm):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj_norm, support)
        return output

class DDGCAE(nn.Module):
    """
    Deep Denoising Graph Convolutional Autoencoder with marginalization
    """
    def __init__(self, input_dim, hidden_dims=[64, 32, 16, 8], corruption_prob=0.43, reg_lambda=1e-5):
        super(DDGCAE, self).__init__()
        self.corruption_prob = corruption_prob
        self.reg_lambda = reg_lambda
        # Build stacked GCN layers
        dims = [input_dim] + hidden_dims
        self.gc_layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.gc_layers.append(GraphConvolution(dims[i], dims[i+1]))
        # Output layer - reconstruction
        self.reconstruct_layer = nn.Linear(hidden_dims[-1], input_dim)
    
    def corrupt_features(self, x):
        """
        Randomly mask features with corruption probability
        """
        noise = torch.rand_like(x)
        mask = (noise > self.corruption_prob).float()
        return x * mask
    
    def forward(self, x, adj_norm):
        """
        Forward pass: input features x and normalized adjacency matrix adj_norm
        """
        x_corrupted = self.corrupt_features(x)
        out = x_corrupted
        for i in range(len(self.gc_layers)):
            out = self.gc_layers[i](out, adj_norm)
            if i != len(self.gc_layers) - 1:
                out = F.relu(out)
            else:
                out = torch.sigmoid(out)
        # Reconstruction
        reconstructed = self.reconstruct_layer(out)
        reconstructed = torch.sigmoid(reconstructed)
        return out, reconstructed
    
    def loss(self, x, reconstructed, weights):
        """
        Reconstruction loss with Frobenius norm regularization
        """
        mse_loss = F.mse_loss(reconstructed, x)
        reg_loss = 0
        for w in weights:
            reg_loss += torch.norm(w, p='fro')**2
        return mse_loss + self.reg_lambda * reg_loss
