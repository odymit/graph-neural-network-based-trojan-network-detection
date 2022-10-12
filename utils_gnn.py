import dgl
from dgl.data import DGLDataset
import os
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
    
class SGN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pooling='sum'):
        super().__init__()
        assert pooling in ['sum', 'avg', 'max'], "Not supported pooling method."
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
        if pooling == 'sum':
            self.pool = SumPooling() # change to mean readout (AvgPooling) on social network datasets
        elif pooling == 'avg':
            self.pool = AvgPooling()
        else:
            self.pool = MaxPooling()
#         self.topK = topK
#         self.pool = SortPooling(topK)
#         self.pool = AvgPooling()
        
    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        # img_class = cnn_foward()
        # gnn_class = gnn_foward()
        # return img_class, gnn_class 
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
    
def split_fold10(labels, fold_idx=0):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, valid_idx = idx_list[fold_idx]
    return train_idx, valid_idx

def evaluate(dataloader, device, model):
    model.eval()
    total = 0
    total_correct = 0
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    for batch, (batched_graph, labels) in enumerate(tqdm(dataloader)):
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        feat = batched_graph.ndata.pop('x')
        total += len(labels)
        logits = model(batched_graph, feat.view(len(feat), -1))
        _, predicted = torch.max(logits, 1)
        total_correct += (predicted == labels).sum().item()
        m = predicted + labels
        total_tp += (m >= 2).sum().item()
        total_tn += (m == 0).sum().item()
        total_fp += (predicted > labels).sum().item()
        total_fn += (predicted < labels).sum().item()
        # print(m, predicted, labels, total_tp, total_tn, total_fp, total_fn)
    acc = 1.0 * total_correct / total
    # pre = 1.0 * total_tp / (total_tp + total_fp)
    # rec = 1.0 * total_tp / (total_tp + total_fn)
    # f1 = 2.0 * pre * rec / (rec + rec)
    # print(acc, pre, rec, f1)
    return acc, total_tp, total_tn, total_fp, total_fn


def all_u_to_v(src, dst):
    ret = []
    for v in dst:
        ret += [[u, v] for u in src]
    return ret

def cnn2graph(model, model_info):
    # convert cnn model to a dgl graph
    # model: model weight data
    # model_info: model struct info
    
    layers = []
    all_node_feats = []
    all_edges = []
    cnt = 0
    with torch.no_grad():
        for i in range(len(model_info)):
            cur_layer_info = model_info[i]
            cur_layer_node = []
            
            if 'conv' in cur_layer_info['name']:
                # construct cur layer nodes
                for weight, bias in zip(model.get_submodule(cur_layer_info['name']).weight, 
                                        model.get_submodule(cur_layer_info['name']).bias):
                    cur_layer_node.append(cnt)
                    cnt += 1
                    # featue resize?
                    w = weight[0] + bias
                    all_node_feats.append(padding(w, 512, 513))
            else:
                # construct dense layer node
                cur_layer_node.append(cnt)
                cnt += 1
                # feature resize?
                w = model.get_submodule(cur_layer_info['name']).weight.t() + model.get_submodule(cur_layer_info['name']).bias
                all_node_feats.append(padding(w, 512, 513))
            layers.append(cur_layer_node)
            
    # get all edges
    for idx in range(len(layers)):
        if idx < len(layers) - 1:
            edges = all_u_to_v(layers[idx], layers[idx+1])
            all_edges += edges
    
    all_edges = torch.tensor(all_edges).t()
    u, v = all_edges[0], all_edges[1]
    g = dgl.graph((u,v)).to('cuda')
    g.ndata['x'] = torch.stack(all_node_feats)
    return g


def padding(src: torch.Tensor, row: int, col: int):
    r, c = src.size()
    
    if r > row or c > col:
        return None
    
    if (row - r) % 2 == 0:
        rl = int((row - r) / 2)
    else:
        rl = int((row - r) / 2 + 1)
    
    if row == r:
        rr = 0
    else:
        rr = int((row - r) / 2)

    if (col - c) % 2 == 0:
        cl = int((col - c) / 2)
    else:
        cl = int((col - c) / 2 + 1)

    if col == c:
        cr = 0
    else:
        cr = int((col - c) / 2)
   
    result = F.pad(input=src, pad=(cl, cr, rl, rr), mode='constant', value=0)
    return result

def load_dataset(shadow_path, is_specific, troj_type, TRAIN_NUM, VAL_NUM, TEST_NUM):
    '''get the data path list of train/val/test dataset'''
    def seed():
        if not is_specific:
            '''load hetero structure model in random'''
            return str(randint(0,5))
        else:
            '''load same structure in specific path'''
            return is_specific

    train_dataset = []
    for i in range(TRAIN_NUM):
        idx = seed()
        x = shadow_path + '/shadow_jumbo_%d.model'%i + idx
        train_dataset.append((x,1,idx))
        x = shadow_path + '/shadow_benign_%d.model'%i + idx
        train_dataset.append((x,0,idx))

    val_dataset = []
    for i in range(TRAIN_NUM, TRAIN_NUM+VAL_NUM):
        idx = seed()
        x = shadow_path + '/shadow_jumbo_%d.model'%i + idx
        val_dataset.append((x,1,idx))
        x = shadow_path + '/shadow_benign_%d.model'%i + idx
        val_dataset.append((x,0,idx))

    test_dataset = []
    for i in range(TEST_NUM):
        idx = seed()
        x = shadow_path + '_hetero' + '/target_troj%s_%d.model'%(troj_type, i) + idx
        test_dataset.append((x,1,idx))
        x = shadow_path + '_hetero' + '/target_benign_%d.model'%i + idx
        test_dataset.append((x,0,idx))
    return train_dataset, val_dataset, test_dataset





def epoch_meta_train(meta_model, father_model, optimizer, dataset, input_size, threshold=0.0):

    meta_model.train()

    cum_loss = 0.0
    preds = []
    labs = []
    perm = np.random.permutation(len(dataset))
    for i in perm:
        x, y, z = dataset[i]
        basic_model = load_spec_model(father_model, z)
        basic_model.train()
        basic_model.load_state_dict(torch.load(x))
        # if is_discrete:
        #     out = basic_model.emb_forward(meta_model.inp)
        # else:
        #     out = basic_model.forward(meta_model.inp)
        graph = cnn2graph(basic_model, input_size)
        score = meta_model.forward(graph)
        l = meta_model.loss(score, y)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.asscalar(np.median(preds))
    acc = ( (preds>threshold) == labs ).mean()

    return cum_loss / len(dataset), auc, acc