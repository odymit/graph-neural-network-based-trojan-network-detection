import dgl
from dgl.data import DGLDataset
import os
from dgl import save_graphs, load_graphs
import torch
from torch import nn
import numpy as np
# from model_lib.mnist_cnn_model import Model
from random import randint
from utils_basic import load_spec_model
from sklearn.metrics import roc_auc_score
from torchinfo import summary

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



def cnn2graph(model, input_size):
    '''transform a cnn model to dgl graph'''
    batch_size = (1,)
    summary_str = str(summary(model, batch_size+input_size))
    print(summary_str)

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