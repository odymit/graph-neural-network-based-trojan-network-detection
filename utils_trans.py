import dgl
from dgl.data import DGLDataset
import os
from dgl import save_graphs, load_graphs
import torch
from torch import nn
import numpy as np
# from model_lib.mnist_cnn_model import Model


def cnn2graph(model):
    '''transform a cnn model to dgl graph'''

