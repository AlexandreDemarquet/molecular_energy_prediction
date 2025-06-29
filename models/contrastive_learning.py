import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models import SchNet
from torch_geometric.nn.pool import global_add_pool
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import ConcatDataset
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_add_pool
from torch_cluster import radius_graph
import torch_cluster
import os
import csv

from utils import ATOM_TYPES

class GraphEncoder(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, cutoff=5.0):
        super().__init__()
        # Atom type embedding
        self.atom_embed = nn.Embedding(len(ATOM_TYPES), hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.act = nn.ReLU()
        self.cutoff = cutoff

    def forward(self, data):
        # data.z: [num_nodes], data.pos: [num_nodes,3], data.batch: [num_nodes]
        x = self.atom_embed(data.z)
        # build edge index based on distance cutoff
        edge_index = radius_graph(data.pos, r=self.cutoff, batch=data.batch, loop=False)
        # GraphSAGE layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
        # global pooling to get [batch_size, hidden_dim]
        h_graph = global_add_pool(x, data.batch)
        return h_graph
    
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)



class FeatureGraphEncoder(GraphEncoder):
    def extract_features(self, data):
        # data.z, data.pos, data.batch
        x = self.atom_embed(data.z)
        pos, batch = data.pos, data.batch
        features = []

        # couche “0” : embedding initial mis en pool
        h0 = global_add_pool(x, batch)
        features.append(h0)

        # chaque convolution + activation + pool
        for conv in self.convs:
            # reconstruire edge_index à chaque fois
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=False)
            x = conv(x, edge_index)
            x = self.act(x)
            h = global_add_pool(x, batch)
            features.append(h)

        # renvoie liste de tenseurs [batch_size, hidden_dim]
        return features
