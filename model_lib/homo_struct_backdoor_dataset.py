from dgl.data.dgl_dataset import DGLBuiltinDataset
from model_lib.mnist_cnn_model import Model0 as Model
import model_lib.mnist_cnn_model as father_model

import os 
import re
import torch
from torch import nn
import dgl
from utils_gnn import cnn2graph
import json

class HomoStrucBackdoorDataset(DGLBuiltinDataset):
    def __init__(self, mode='train', raw_dir='/home/ubuntu/date/hdd4/shadow_model_ckpt/mnist/models/',
                 force_reload=False, verbose=False, transform=None):
        mode = mode.lower()
        assert mode in ['train', 'valid', 'test'], "Mode not valid."
        self.mode = mode    
        self.x = []
        self.y = []
        _url = None
        
        super(HomoStrucBackdoorDataset, self).__init__(name='HomoBackdoorDT',
                                           raw_dir=raw_dir,
                                           force_reload=force_reload,
                                           verbose=verbose,
                                           url=_url,
                                           transform=transform)
        self.load()
        
        
    def process(self):
        pass
    
    def has_cache(self):
        pass
    
    def load(self):
        '''load dataset info'''
        
        for filename in os.listdir(self.raw_dir):
            if '.model' not in filename:
                # not a model
                continue
            idx_pattern = '[0-9]+'
            idx = re.findall(idx_pattern, filename)
            if self.mode == 'train':
                if int(idx[0]) < 2048 and 'target' not in filename:
                    # is a training model
                    self.x.append(filename)
                else:
                    continue
                # print(filename)
            elif self.mode == 'valid':
                if int(idx[0]) >= 2048 and 'target' not in filename:
                    self.x.append(filename)
                else:
                    continue
            else:
                # self.mode == 'test'
                if 'target' in filename:# and 'B' not in filename
                    self.x.append(filename)
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

    def load_spec_model(self, module, model_index):
        model = getattr(module, 'Model'+model_index)
        return model
            
    def is_correct_labeled(self):
        x = self.x
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
        
    def load_g(self, idx):
        x = os.path.join(self.raw_dir, self.x[idx])
        y = self.y[idx]
    #         print(label)
        CUDA_LAUNCH_BLOCKING=1
        Model = self.load_spec_model(father_model, '0')
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
        

        # load graph as dgl graph
        g = cnn2graph(model, model_detail['mnist']['0'])
        return g, y
 