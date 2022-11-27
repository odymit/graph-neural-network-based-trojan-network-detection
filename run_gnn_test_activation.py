import dgl
import torch
from utils_gnn import cnn2graph_activation
# from model_lib import mnist_cnn_model as father_model
from utils_basic import load_spec_model
from utils_gnn import padding
from dgl.data import DGLDataset
import os
from tqdm import tqdm
from dgl import save_graphs, load_graphs
import torch
import json
from torch import nn
import numpy as np
# from model_lib.mnist_cnn_model import Model
from random import randint
from utils_basic import load_spec_model
from sklearn.metrics import roc_auc_score
from torchinfo import summary
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, SortPooling
from utils_gnn import MLP
from utils_gnn import unpadding, padding




# x = '/home/dorian/repos/Meta-Nerual-Trojan-Detection/shadow_model_ckpt/mnist/models5/shadow_jumbo_9.model'
x = './shadow_model_ckpt/mnist/models5/shadow_jumbo_0.model'
# load model 
# Model = load_spec_model(father_model, '5')
from model_lib.mnist_cnn_model import Model6 as Model
model = Model(gpu=True)
params = torch.load(x)
model.load_state_dict(params)
del params

# load model detail 
model_detail = {}
model_detail_path = "./intermediate_data/model_detail.json"
import json
with open(model_detail_path, 'r') as f:
    model_detail = json.load(f)
# print(model_detail)
g = cnn2graph_activation(model, model_detail['mnist']['5'])
dgl.save_graphs('./intermediate_data/grapj_test.bin', g)
del model_detail

import torch
import torchvision
import torchvision.transforms as transforms
# from utils_gnn import SGNACT
GPU = True
if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
transform = transforms.Compose([
            transforms.ToTensor(),
        ])
BATCH_SIZE = 1
# MNIST image dataset 
trainset = torchvision.datasets.MNIST(root='./raw_data/', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE)

# get a image
image = None
label = None
for i, (x_in, y_in) in enumerate(dataloader):
    image = x_in
    label = y_in
    break
del trainset, dataloader

def conv(in_channels, out_channels, data, weight, bias, kernel_size, stride, padding):
    row, col = weight.size()
    # get actual conv kernel weight and bias
    w = unpadding(weight, kernel_size, kernel_size)
    w = w.unsqueeze(0).unsqueeze(0)
    b = unpadding(bias, 1, 1)[0]
    # get conv operator 
    operator = torch.nn.Conv2d(in_channels, out_channels, 1, kernel_size=kernel_size, 
                    stride=stride, padding=padding)
    # set conv operator weight and bias
    operator.weight.data = w
    operator.bias.data = b
    # conduct conv operation
    x = operator(data)
    return x

def maxpool(kernel_size, stride, padding):
    pass  


def reduce_func(nodes):
    print("this is reduce function")
    # input_feat = nodes.data['x'][input_mask]
    # print(input_feat)
    print("reduce function ends")
    return {'ft': nodes.data['x']}


def initiate_node_feature(graph):
    ft = None 

    graph.ndata['ft'] = ft

def cnn_cal(graph, image):
    initiate_node_feature(graph, image)
    graph.update_all(message_func=message_func, reduce_func=reduce_func)
    ft = graph.ndata['ft'][0]
    return ft

import json
# get cnn results
pred, params = model(x_in)
print(params.keys())
with open("./intermediate_data/params.json", "w") as f:
    json.dump(params, f)
# get gnn transmission results
res = cnn_cal(g)
