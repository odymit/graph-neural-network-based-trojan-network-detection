from dgl.data.dgl_dataset import DGLBuiltinDataset
from model_lib.mnist_cnn_model import Model0 as Model
import os 
import re
import torch
from torch import nn
import dgl

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
        basic_model = Model().cuda()
        t = torch.load(x)
        t = basic_model.load_state_dict(t)
        
        g = None
        with torch.no_grad():
            # nodes_feat 512 * 513
            nodes_feat = []
            cnt = 0
            # get conv1 nodes 
            conv1 = {}
            for weight in basic_model.conv1.weight:
                pad = nn.ZeroPad2d(padding=(254,254,253,254))
                feat = pad(weight[0])
                conv1[cnt] = feat
                nodes_feat.append(feat)
                cnt += 1

            # get conv2 nodes
            conv2 = {}
            for weight in basic_model.conv2.weight:
                pad = nn.ZeroPad2d(padding=(254,254,253,254))
                feat = pad(weight[0])
                conv2[cnt] = feat
                nodes_feat.append(feat)
                cnt += 1

            # get conv1 -> conv2 edges
            conv1_2 = []
            for src in conv1.keys():
                for dst in conv2.keys():
                    conv1_2.append([src, dst])


            # get fc node
            fc_index = cnt
            cnt += 1
            fc_node = torch.concat([basic_model.fc.weight, basic_model.fc.bias.reshape(512, 1)], 1)
            nodes_feat.append(fc_node)
            # print(fc_node.shape)

            # get conv2 -> fc edges
            conv2_fc = []
            for src in conv2.keys():
                conv2_fc.append([src, fc_index])

            # get output node
            out_index = cnt
            cnt += 1 
            out = torch.concat([basic_model.output.weight, basic_model.output.bias.reshape(10, 1)], 1)
            pad = nn.ZeroPad2d(padding=(0,0,251,251))
            out_node = pad(out)
            nodes_feat.append(out_node)

            # print(out_node.shape)

            # get fc -> output edge
            fc_out_edge = [[fc_index, out_index]]

            # get all nodes
            nodes_feat = torch.stack(nodes_feat)
            # print(nodes_feat.shape)
            # get all edges
            all_edges = torch.tensor(conv1_2 + conv2_fc + fc_out_edge).t().tolist()
            u, v = all_edges[0], all_edges[1]


            g = dgl.graph((u,v)).to('cuda')
            g.ndata['x'] = nodes_feat
        return g, y
 