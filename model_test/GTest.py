import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, SortPooling
import argparse
from tqdm import tqdm

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP    
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)
    
class GTest(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pool='sum'):
        super().__init__()
        assert pool in ['sum', 'avg', 'max'], "Not supported pooling method."
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 2
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
#         for layer in range(num_layers - 1): # excluding the input layer
#             if layer == 0:
#                 mlp = MLP(input_dim, hidden_dim, hidden_dim)
#             else:
#                 mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
#             self.ginlayers.append(GINConv(mlp, learn_eps=False)) # set to True if learning epsilon
#             self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.8)
        if pool == 'sum'
            self.pool = SumPooling() # change to mean readout (AvgPooling) on social network datasets
        elif pool == 'avg':
            self.pool = AvgPooling()
        else:
            self.pool = MaxPooling()
        
    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer