from curses import raw
from multiprocessing.dummy import active_children
from re import L
from tkinter import W
from urllib.request import AbstractBasicAuthHandler
import numpy as np
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
# from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, SortPooling
import argparse
from tqdm import tqdm
from utils_gnn import SGN, split_fold10, evaluate, load_spec_model
from model_lib.homo_struct_backdoor_dataset import HomoStrucBackdoorDataset
from model_lib.hetero_struct_backdoor_dataset import HeteroStrucBackdoorDataset


def train(epochs, train_loader, val_loader, test_loader, device, model):
    # loss function, optimizer and scheduler
    # loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    info = {'train_acc':[], 'val_acc':[], 'test_acc':[], 
            'train_tp':[], 'val_tp':[], 'test_tp':[],
            'train_tn':[], 'val_tn':[], 'test_tn':[],
            'train_fp':[], 'val_fp':[], 'test_fp':[],
            'train_fn':[], 'val_fn':[], 'test_fn':[],
            'fpr': None, 'tpr': None, 'threshold': None,
            'loss':[]}
    # training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        actual_labels = None
        pred_scores = None 
        for batch, (batched_graph, labels) in enumerate(tqdm(train_loader)):
            batched_graph = batched_graph.to(device)
            #print(batch, labels, type(labels))
            labels = labels.unsqueeze(1).float().to(device)
            # print("labels: ", labels)
            if actual_labels is not None:
                actual_labels = torch.cat([actual_labels, labels], dim=0)
            else:
                actual_labels = labels
            # print(labels)
            feat = batched_graph.ndata.pop('x')
            # print(feat.view(50,-1).shape)
            logits = model(batched_graph, feat.view(len(feat), -1))
            # print("logits: ", logits)
            if pred_scores is not None:
                inner_logits = torch.cat([logits.t()], dim=0)[0]
                # print("pred_scores & inner_logits:", pred_scores, inner_logits[0])
                pred_scores = torch.cat([pred_scores, inner_logits], dim=0)
            else:
                inner_logits = torch.cat([logits.t()], dim=0)[0]
                pred_scores = inner_logits

            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        model.eval()
        # print(actual_labels)
        # print(pred_scores)
        threshold = None
        with torch.no_grad():
            fpr, tpr, threshold = roc_curve(actual_labels.cpu(), pred_scores.cpu(), pos_label=1)
            info['threshold'] = threshold
            threshold = threshold[(tpr - fpr).argmax()]
            info['fpr'] = fpr
            info['tpr'] = tpr
            

        # print("threshold: ", threshold)
        # acc, pre, rec, f1
        train_acc, tp, tn, fp, fn  = evaluate(train_loader, device, model, threshold=threshold)
        # train_acc, tp, tn, fp, fn  = evaluate(train_loader, device, model)
        info['train_acc'].append(train_acc)
        info['train_tp'].append(tp)
        info['train_tn'].append(tn)
        info['train_fp'].append(fp)
        info['train_fn'].append(fn)
        # val_acc, tp, tn, fp, fn = evaluate(val_loader, device, model)
        # info['val_acc'].append(val_acc)
        # info['val_tp'].append(tp)
        # info['val_tn'].append(tn)
        # info['val_fp'].append(fp)
        # info['val_fn'].append(fn)
        # test_acc, tp, tn, fp, fn = evaluate(test_loader, device, model)
        # info['test_acc'].append(test_acc)
        # info['test_tp'].append(tp)
        # info['test_tn'].append(tn)
        # info['test_fp'].append(fp)
        # info['test_fn'].append(fn)
        # info['loss'].append(total_loss/(batch+1))
        val_acc = 0
        test_acc = 0

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
    parser.add_argument('--model', type=str, default='SGN', choices=['SGN'], help='choose a graph classification model')
    parser.add_argument('--struct', type=str, default='homo', choices=['homo', 'hetero'])

    args = parser.parse_args()
    print(f'Training with DGL built-in GINConv module with a fixed epsilon = 0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load and split dataset
    # dataset = GINDataset(args.dataset, self_loop=True, degree_as_nlabel=False) # add self_loop and disable one-hot encoding for input features
    if args.struct == 'homo':
        Dataset = HomoStrucBackdoorDataset
    elif args.struct == 'hetero':
        Dataset = HeteroStrucBackdoorDataset
    # dataset = Dataset(raw_dir='./shadow_model_ckpt/mnist/models/')
    dataset = Dataset()
    val_dataset = Dataset(raw_dir='./shadow_model_ckpt/mnist/models/', mode='valid')
    test_dataset = Dataset(raw_dir='./shadow_model_ckpt/mnist/models/', mode='test')
    print("train_dataset: %d" % len(dataset))
    print("val_dataset: %d" % len(val_dataset))
    print("test_dataset: %d" % len(test_dataset))

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
    out_size = 1
    model = SGN(in_size, 16, out_size, pooling=args.pooling).to(device)

    # model training/validating
    print('Training Procedure...')
    train(3, train_loader, val_loader, test_loader, device, model)
    