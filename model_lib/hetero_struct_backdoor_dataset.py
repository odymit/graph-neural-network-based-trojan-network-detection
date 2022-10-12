from fileinput import filename
from dgl.data.dgl_dataset import DGLBuiltinDataset
from numpy import dtype
from sklearn import datasets
import model_lib.mnist_cnn_model as father_model
import os 
import re
import torch
import json
from torch import nn
import dgl
import random
from utils_gnn import cnn2graph

class HeteroStrucBackdoorDataset(DGLBuiltinDataset):
    def __init__(self, mode='train', raw_dir='/home/ubuntu/date/hdd4/shadow_model_ckpt/mnist/models',
                 force_reload=False, verbose=False, transform=None, seed=1024):
        mode = mode.lower()
        assert mode in ['train', 'valid', 'test'], "Mode not valid."
        self.mode = mode    
        self.x = []
        self.y = []
        _url = None
        random.seed(seed)
        self.ntypes = 6
        self.model_nums = 2048
        
        super(HeteroStrucBackdoorDataset, self).__init__(name='HeteroBackdoorDT',
                                           raw_dir=raw_dir,
                                           force_reload=force_reload,
                                           verbose=verbose,
                                           url=_url,
                                           transform=transform)
        self.load()
        
    
    def randtype(self):
        randint = int(random.random() * self.ntypes)
        while(randint not in [0, 1, 5]):
            randint = int(random.random() * self.ntypes)
        return str(randint)

    def process(self):
        pass
    
    def has_cache(self):
        pass
    
    def load(self):
        '''load dataset info'''
        for idx in range(self.model_nums):
            filename = "shadow_benign_%d.model" % idx
            self.add_x_y(self.randtype(), filename)
            filename = "shadow_jumbo_%d.model" % idx
            self.add_x_y(self.randtype(), filename)
            
    def add_x_y(self, model_type, filename):            
        for filename in os.listdir(self.raw_dir):
            if '.model' not in filename:
                # not a model
                continue
            idx_pattern = '[0-9]+'
            idx = re.findall(idx_pattern, filename)
            if self.mode == 'train':
                if int(idx[0]) < 2048 and 'target' not in filename:
                    # is a training model
                    self.x.append([model_type, filename])
                else:
                    continue
                # print(filename)
            elif self.mode == 'valid':
                if int(idx[0]) >= 2048 and 'target' not in filename:
                    self.x.append([model_type, filename])
                else:
                    continue
            else:
                # self.mode == 'test'
                if 'target' in filename:# and 'B' not in filename
                    self.x.append([model_type, filename])
                else:
                    continue
            # add co
            if 'benign' in filename:
                self.y.append(0)
            else:
                self.y.append(1)
        
    def __getitem__(self, idx):
        assert idx < len(self.x), "Out of index when get item."
        # load data, process and return
        g, y = self.load_g(idx)
        return g, y
        
    def __len__(self):
        return len(self.x)
    
    def get_x_y(self):
        return self.x, self.y
    
    def iter_y(self):
        for y in self.y:
            yield y
            
    def is_correct_labeled(self):
        _, x = zip(self.x)
        y = self.y
        cnt = 0
        error = 0
        for i,j in zip(x,y):
            if 'benign' in i and j == 0:
                cnt += 1
            elif 'benign' not in i and j == 1:
                cnt += 1
            else:
                error += 1
        if cnt != len(x) or error > 0:
            return False
        else:
            return True
    def load_spec_model(self, module, model_index):
        model = getattr(module, 'Model'+model_index)
        return model
    
        
    def load_g(self, idx):
        x = os.path.join(self.raw_dir+self.x[idx][0], self.x[idx][1])
        y = self.y[idx]

        # print(x)
        CUDA_LAUNCH_BLOCKING=1
        # load model 
        Model = self.load_spec_model(father_model, self.x[idx][0])
        model = Model().cuda()
        params = torch.load(x)
        model.load_state_dict(params)
        
        # load model detail 
        model_detail = {}
        model_detail_path = "./intermediate_data/model_detail.json"
        with open(model_detail_path, 'r') as f:
            model_detail = json.load(f)

        # get dataset type 
        dataset_type = None
        if 'mnist' in x:
            dataset_type = 'mnist'
        elif 'cifar10' in x:
            dataset_type = 'cifar10'
        
        # get model type id
        id = self.x[idx][0]

        # load graph as dgl graph
        g = cnn2graph(model, model_detail[dataset_type][id])
        return g, y