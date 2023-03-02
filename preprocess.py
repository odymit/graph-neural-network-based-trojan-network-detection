# Load and preprocess data
import torch
from tqdm import tqdm
from time import sleep
TRAIN_NUM = 2048
VAL_NUM = 256
TEST_NUM = 256
shadow_path = '/home/ubuntu/date/hdd4/shadow_model_ckpt/%s/models' % 'mnist'
train_dataset = []
for i in range(TRAIN_NUM):
    x = shadow_path + '/shadow_jumbo_%d.model' % i
    train_dataset.append((x, 1))
    x = shadow_path + '/shadow_benign_%d.model' % i
    train_dataset.append((x, 0))

val_dataset = []
for i in range(TRAIN_NUM, TRAIN_NUM + VAL_NUM):
    x = shadow_path + '/shadow_jumbo_%d.model' % i
    val_dataset.append((x, 1))
    x = shadow_path + '/shadow_benign_%d.model' % i
    val_dataset.append((x, 0))

test_dataset = []
for i in range(TEST_NUM):
    x = shadow_path + '/target_troj%s_%d.model' % ('M', i)
    test_dataset.append((x, 1))
    x = shadow_path + '/target_benign_%d.model' % i
    test_dataset.append((x, 0))

# print(train_dataset, val_dataset, test_dataset)
import numpy as np
from torchsummary import summary
from model_lib.mnist_cnn_model import Model0 as Model
from torch import nn
from cogdl.data import Graph

m = None


def load_dataset(dataset):
    perm = np.random.permutation(len(dataset))
    graphs = []

    for i in tqdm(perm):
        sleep(0.2)
        x, y = dataset[i]
        label = torch.IntTensor([y])
        #         print(label)
        basic_model = Model()
        t = torch.load(x)
        t = basic_model.load_state_dict(t)

        with torch.no_grad():
            # nodes_feat 512 * 513
            nodes_feat = []
            cnt = 0
            # get conv1 nodes
            conv1 = {}
            for weight in basic_model.conv1.weight:
                pad = nn.ZeroPad2d(padding=(254, 254, 253, 254))
                feat = pad(weight[0])
                conv1[cnt] = feat
                nodes_feat.append(torch.reshape(feat, (1, 262656))[0])
                cnt += 1
            # print(conv1[0].shape)
            # get conv2 nodes
            conv2 = {}
            for weight in basic_model.conv2.weight:
                pad = nn.ZeroPad2d(padding=(254, 254, 253, 254))
                feat = pad(weight[0])
                conv2[cnt] = feat
                nodes_feat.append(torch.reshape(feat, (1, 262656))[0])
                cnt += 1
            # print(conv2[16].shape)
            # get conv1 -> conv2 edges
            conv1_2 = []
            for src in conv1.keys():
                for dst in conv2.keys():
                    conv1_2.append([src, dst])

            # get fc node
            fc_index = cnt
            cnt += 1
            fc_node = torch.concat([basic_model.fc.weight, basic_model.fc.bias.reshape(512, 1)], 1)
            nodes_feat.append(torch.reshape(fc_node, (1, 262656))[0])
            # print(fc_node.shape)

            # get conv2 -> fc edges
            conv2_fc = []
            for src in conv2.keys():
                conv2_fc.append([src, fc_index])

            # get output node
            out_index = cnt
            cnt += 1
            out = torch.concat([basic_model.output.weight, basic_model.output.bias.reshape(10, 1)], 1)
            pad = nn.ZeroPad2d(padding=(0, 0, 251, 251))
            out_node = pad(out)
            nodes_feat.append(torch.reshape(out_node, (1, 262656))[0])

            # print(out_node.shape)

            # get fc -> output edge
            fc_out_edge = [[fc_index, out_index]]

            # get all nodes
            nodes_feat = torch.stack(nodes_feat).to_sparse()
            # print(nodes_feat.shape)
            # get all edges
            all_edges = torch.tensor(conv1_2 + conv2_fc + fc_out_edge).t().to_sparse()
            # print(all_edges)
            # g = Graph(edge_index=all_edges, x=nodes_feat, y=label)
            # dst_filename = '/home/ubuntu/date/hdd4'+ x[1::] + '.pth'
            dst_filename = '/'+ x[1::] + '.pth'
            # print(dst_filename.replace('models', 'sparse_graph_info_data'))
            # input("wait to check path")
            torch.save({'edges': all_edges, 'x': nodes_feat, 'y': label},
                       dst_filename.replace('models', 'sparsed_graph_info_data'))

            # print(g.save_sparse_csr())
            #             print("nodes:", g.num_nodes)
            #             print("edges:", g.num_edges)
            # graphs.append(g)
    return graphs


res = load_dataset(train_dataset)
res = load_dataset(val_dataset)
res = load_dataset(test_dataset)
