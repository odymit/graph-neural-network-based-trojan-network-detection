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
from model_lib.homo_struct_backdoor_dataset import HomoStrucBackdoorDataset


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

def train(epochs, train_loader, val_loader, test_loader, device, model):
    # loss function, optimizer and scheduler
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    info = {'train_acc':[], 'val_acc':[], 'test_acc':[], 
            'train_tp':[], 'val_tp':[], 'test_tp':[],
            'train_tn':[], 'val_tn':[], 'test_tn':[],
            'train_fp':[], 'val_fp':[], 'test_fp':[],
            'train_fn':[], 'val_fn':[], 'test_fn':[],
            'loss':[]}
    # training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch, (batched_graph, labels) in enumerate(tqdm(train_loader)):
            batched_graph = batched_graph.to(device)
            #print(batch, labels, type(labels))
            labels = labels.to(device)
            # print(labels)
            feat = batched_graph.ndata.pop('x')
            # print(feat.view(50,-1).shape)
            logits = model(batched_graph, feat.view(len(feat), -1))
            # print(logits)
            # print(logits.shape, labels.shape)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        # acc, pre, rec, f1
        train_acc, tp, tn, fp, fn  = evaluate(train_loader, device, model)
        info['train_acc'].append(train_acc)
        info['train_tp'].append(tp)
        info['train_tn'].append(tn)
        info['train_fp'].append(fp)
        info['train_fn'].append(fn)
        val_acc, tp, tn, fp, fn = evaluate(val_loader, device, model)
        info['val_acc'].append(val_acc)
        info['val_tp'].append(tp)
        info['val_tn'].append(tn)
        info['val_fp'].append(fp)
        info['val_fn'].append(fn)
        test_acc, tp, tn, fp, fn = evaluate(test_loader, device, model)
        info['test_acc'].append(test_acc)
        info['test_tp'].append(tp)
        info['test_tn'].append(tn)
        info['test_fp'].append(fp)
        info['test_fn'].append(fn)
        info['loss'].append(total_loss/(batch+1))

        print("Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Validation Acc. {:.4f}| Test Acc. {:.4f} "
              . format(epoch, total_loss / (batch + 1), train_acc, val_acc, test_acc))
        
    from datetime import datetime
    import json
    now = datetime.now()
    date = now.strftime("%Y-%m-%d-%H:%M:%S")
    with open('./intermediate_data/train-%s.json' % date, 'w') as f:
        json.dump(info, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default="MUTAG",
#                         choices=['MUTAG', 'PTC', 'NCI1', 'PROTEINS'],
#                         help='name of dataset (default: MUTAG)')
    parser.add_argument('--pooling', type=str, default='sum', choices=['sum', 'avg', 'max'], help='pooling method, default:sum')
    args = parser.parse_args()
    print(f'Training with DGL built-in GINConv module with a fixed epsilon = 0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load and split dataset
    # dataset = GINDataset(args.dataset, self_loop=True, degree_as_nlabel=False) # add self_loop and disable one-hot encoding for input features
    dataset = HomoStrucBackdoorDataset()
    val_dataset = HomoStrucBackdoorDataset(mode='valid')
    test_dataset = HomoStrucBackdoorDataset(mode='test')

    # train_idx, val_idx = split_fold10(labels)
    # print(train_idx, val_idx)
    
    # create dataloader
    train_loader = GraphDataLoader(dataset, batch_size=4, pin_memory=torch.cuda.is_available())
    val_loader = GraphDataLoader(val_dataset, batch_size=4, pin_memory=torch.cuda.is_available())
    test_loader = GraphDataLoader(test_dataset, batch_size=4, pin_memory=torch.cuda.is_available())
    
    # create GIN model
    in_size = 512 * 513
    #gin_dataset = GINDataset('MUTAG', self_loop=True, degree_as_nlabel=False) # add self_loop and disable one-hot encoding for input features
    # print(gin_dataset.dim_nfeats)
    out_size = 2
    model = SGN(in_size, 16, out_size, pooling=args.pooling).to(device)

    # model training/validating
    print('Training Procedure...')
    train(100, train_loader, val_loader, test_loader, device, model)
    