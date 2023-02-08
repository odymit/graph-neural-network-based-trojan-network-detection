import dgl
import torch
import numpy as np
from utils_gnn import padding, unpadding, equals, prepare_data
from torch.nn.parameter import Parameter


def activation_passing(img, g):
    '''
    input a imgae and graph, output the activation graph of the image
    '''
    # init ft feature on first conv layer and get conv subg
    with torch.no_grad():
        initiate_node_feature(g, img)
    
    def neural_conv_nodes(nodes):
        return nodes.data['layer_type'] == 0
    nodes_idx = g.filter_nodes(neural_conv_nodes)
    subg = dgl.node_subgraph(g, nodes_idx, relabel_nodes=True)
    # message passing on conv layers
    with torch.no_grad():
        # 当最后一个节点尚未传播到的时候，继续执行传播操作
        while equals(subg.ndata['ft_size'][-1], zeros=True):
            subg.update_all(my_message, my_reduce)
            for idx in range(len(subg.ndata['ft'])):
                if equals(subg.ndata['ft'][idx], zeros=True):
                    subg.ndata['ft'][idx] = subg.ndata['h'][idx]
                if equals(subg.ndata['ft_size'][idx], zeros=True):
                    subg.ndata['ft_size'][idx] = subg.ndata['g'][idx]
    # add message passing ndata on conv layers to original graph 
    g.ndata['ft'][:len(subg.ndata['ft'])] = subg.ndata['ft']
    g.ndata['ft_size'][:len(subg.ndata['ft_size'])] = subg.ndata['ft_size']
    # get fc subg 
    def neural_fc_nodes(nodes):
        layer_mask = nodes.data['layer_type'] == 1
        layer_idxs = nodes.data['layer_idx'][layer_mask]
        n = layer_idxs[0] 
        return nodes.data['layer_idx'] >= n
    fc_nodes = g.filter_nodes(neural_fc_nodes)
    subfcg = dgl.node_subgraph(g, fc_nodes, relabel_nodes=True)
    # init ft on first fc layer
    n, fc_concat_result, r, c = get_fc_first_ft(g)
    g.ndata['ft'][n] = encode_fc_feat(fc_concat_result)
    g.ndata['ft_size'][n] = torch.tensor([r, c], dtype=torch.float32).to("cuda")
    # message passing on fc layers 
    with torch.no_grad():
        while equals(subfcg.ndata['ft_size'][-1], zeros=True):
            subfcg.update_all(my_fc_message, my_fc_reduce)
            for idx in range(len(subfcg.ndata['ft'])):
                if equals(subfcg.ndata['ft'][idx], zeros=True):
                    subfcg.ndata['ft'][idx] = subfcg.ndata['h'][idx]
                if equals(subfcg.ndata['ft_size'][idx], zeros=True):
                    subfcg.ndata['ft_size'][idx] = subfcg.ndata['g'][idx]
    # add message passing ndata on fc layers to original graph
    g.ndata['ft'][-len(subfcg.ndata['ft']):] = subfcg.ndata['ft']
    g.ndata['ft_size'][-len(subfcg.ndata['ft_size']):] = subfcg.ndata['ft_size']
    return g


def initiate_node_feature(graph, image):
    ft = None 
    mask = graph.ndata['layer_idx'] == 0
    out_channels = int(sum(mask))
    ft = torch.zeros((len(graph.nodes()), 1, 1, 28, 28))
    convd_size = torch.zeros((len(graph.nodes()), 2))
    for i in range(out_channels):
        kernel_size, stride, padding = graph.ndata['kernel_params'][i]
        # do conv
        res_ft = init_conv(image, None, graph.ndata['kernel_weight'][i], graph.ndata['bias'][i],
                kernel_size, stride, padding)
        # do relu
        relu_opt = torch.nn.functional.relu
        res_ft = relu_opt(res_ft)

        # do max pooling
        pooling = graph.ndata['pooling_params'][i]
        if pooling.all() != 0:
            # do max_pooling
            kernel_size, stride, pad, dilation, ceil_mode = pooling
            max_pooling_operator = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, 
                padding=pad, dilation=dilation, ceil_mode=ceil_mode)
            
            res_ft = max_pooling_operator(res_ft)
        _, _, r, c = res_ft.size()
        ft[i] = res_ft
        convd_size[i] = torch.tensor([int(r), int(c)])
    graph.ndata['ft'] = ft.to("cuda")
    graph.ndata['ft_size'] = convd_size.to("cuda")

def init_conv(data, data_size, weight, bias, kernel_size, stride, padding):
    # global cnt
    row, col = weight.size()
    # print(weight.size())
    # get actual conv kernel weight and bias
    w = unpadding(weight, 1, kernel_size[0]*kernel_size[1])
    w = w.reshape(1, 1, kernel_size[0], kernel_size[1])
    ws = w
    # np.savetxt("./intermediate_data/init/weight-%d.csv" % cnt, ws[0][0].cpu().numpy(), delimiter=',')
    # cnt += 1
    b = unpadding(bias, 1, 1)[0]
    # np.savetxt("./intermediate_data/init/bias-%d.csv" % cnt, b.cpu().numpy(), delimiter=',')
    # get conv operator 
    # print("kernel_size, stride, padding:")
    # print(kernel_size, stride, padding)
    operator = torch.nn.Conv2d(1, 1, kernel_size=kernel_size, 
                    stride=stride, padding=padding)
    # set conv operator weight and bias
    operator.weight.data = w
    operator.bias.data = b
    # conduct conv operation
    # print("conv input size:", data.size())
    x = operator(data.to("cuda"))

    return x


def prepare_conv_data(nkernel_weight, 
                      nkernel_size, 
                      nkernel_bias, 
                      nkernel_params, 
                      channels):
    # prepare conv kernel weight
    _, width, height = nkernel_size
    kernel_weight = nkernel_weight
    kernel_weight = unpadding(kernel_weight, channels, width * height)
    kernel_weight = kernel_weight.reshape(1, channels, width, height)
    # prepare conv bias
    bias = nkernel_bias
    kernel_bias = unpadding(bias, 1, 1)[0]

    # prepare conv operator
    kernel_size, stride, pad = nkernel_params

    return kernel_weight, kernel_bias, kernel_size, stride, pad
    
def do_operation(data, conv_opt, pooling_params):
    # do conv
    torch.save(data, "./intermediate_data/reshaped_data.pt")
    conv_ft = conv_opt(data.to("cuda")).to("cuda")
    _, _, width, height = conv_ft.size()
    # np.savetxt("./intermediate_data/process/conv_processed_ft.csv", conv_ft.reshape(width, height).cpu().numpy(), delimiter=',')
    # do relu
    relu_ft = torch.nn.functional.relu(conv_ft)
    np.savetxt("./intermediate_data/process/relu_processed_ft.csv", relu_ft.reshape(width, height).cpu().numpy(), delimiter=',')
    ret_ft = relu_ft
    # do pooling
    if pooling_params.any() != 0:
        kernel_size, stride, pad, dilation, ceil_mode = pooling_params
        kernel_size, stride, pad, dilation, ceil_mode = int(kernel_size), int(stride), int(pad), int(dilation), bool(ceil_mode)
        max_pooling_operator = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, 
            padding=pad, dilation=dilation, ceil_mode=ceil_mode)
        
        ret_ft = max_pooling_operator(relu_ft)
        _, _, width, height = ret_ft.size()
        # np.savetxt("./intermediate_data/ret_ft.csv", ret_ft.reshape(width, height).cpu().numpy(), delimiter=',')

    return ret_ft



def my_reduce(nodes):
    # received data and their size
    mfeat = nodes.mailbox['m']
    msize = nodes.mailbox['n']
    
    mf_size = mfeat.size()
    n_nodes = mf_size[0]
    n_channels = mf_size[1]

    # prepare ret_ft & ret_size, input size (1, 1, 28, 28)
    ret_ft = torch.zeros((n_nodes, 1, 1, 28, 28)).to("cuda")
    ret_size = torch.zeros((n_nodes, 2)).to("cuda")

    # for each node, do conv for received data
    for out_idx in range(n_nodes):

        # prepare received data and conv weight
        received_data = mfeat[out_idx]
        received_size = msize[out_idx]
        np.savetxt("./intermediate_data/process/rec-size-%d.csv" % out_idx, received_size.cpu().numpy(), delimiter=',')
        torch.save(received_data, "./intermediate_data/process/receivec_data.pt")

        # skip if all zeros
        if equals(received_data, zeros=True):
            continue
        # (n, 1, 1, 28, 28) -> (1, n, width, height)
        data = prepare_data(received_data, received_size)

        # prepare conv operator data
        kernel_weight, kernel_bias, kernel_size, stride, pad = prepare_conv_data(nodes.data['kernel_weight'][out_idx], 
                                                                                nodes.data['kernel_size'][out_idx],
                                                                                nodes.data['bias'][out_idx], 
                                                                                nodes.data['kernel_params'][out_idx], 
                                                                                n_channels)
        
        # prepare conv operator
        conv_opt = torch.nn.Conv2d(n_channels, 1, kernel_size=kernel_size, stride=stride, padding=pad)
        conv_opt.weight.data = kernel_weight
        conv_opt.bias.data = kernel_bias
        torch.save(kernel_weight, "./intermediate_data/process/kernel_weight.pt")
        np.savetxt("./intermediate_data/process/kernel-bias-%d.csv" % (out_idx),  kernel_bias.cpu().numpy(), delimiter=',')
        # print("kernel weight size:", kernel_weight.size(), kernel_size)
        # print("kernel_bias size:", kernel_size.size())

        # do operation
        ft = do_operation(data, conv_opt, nodes.data['pooling_params'][out_idx])

        # save ft size
        _, _, width, height = ft.size()
        np.savetxt("./intermediate_data/process/ret_ft.csv", ft.reshape(width, height).cpu().numpy(), delimiter=',')
        pad_ft = padding(ft.reshape(width, height), 28, 28)
        np.savetxt("./intermediate_data/process/pad_ft.csv", pad_ft.cpu().numpy(), delimiter=',')
        reshaped_ft = pad_ft.reshape(1, 1, 28, 28)
        # update return data
        ret_ft[out_idx] = reshaped_ft
        ret_size[out_idx] = torch.tensor([width, height]).to("cuda")
        # a_break = input("go next?")
        
    # return size is [n, 1, 1, 28, 28], reduced from [n, m, 1, 1, 28, 28]
    return {'h': ret_ft, 'g': ret_size}

def my_message(edges):
    return {'m': edges.src['ft'], 'n': edges.src['ft_size']}

def get_fc_first_ft(g):
    def neural_nodes_before_fc(nodes):
        layer_mask = nodes.data['layer_type'] == 1
        layer_idxs = nodes.data['layer_idx'][layer_mask]
        n = layer_idxs[0] 
        return nodes.data['layer_idx'] == n - 1 
    pre_nodes_idx = g.filter_nodes(neural_nodes_before_fc)
    # print(pre_nodes_idx)
    # get feats and reshape to (1, channels, width, height)
    prefg = dgl.node_subgraph(g, pre_nodes_idx, relabel_nodes=True)
    pre_feats = prefg.ndata['ft']
    # (8, 1, 1, 28, 28)
    # print(pre_feats.shape)
    n_pre_nodes = len(pre_nodes_idx)
    shape = prefg.ndata['ft_size'][0]
    w, h = shape
    w, h = int(w), int(h)
    del prefg
    # reshape to (1, 8, 7, 7)
    feats = torch.zeros([n_pre_nodes, w, h])
    for idx in range(n_pre_nodes):
        cur_data = pre_feats[idx]
        cur_data = cur_data.reshape(28, 28)
        res_data = unpadding(cur_data, w, h)
        # print("res_data size:", res_data.size())
        feats[idx] = res_data
    data = feats
    # print("data size: ", data.shape)

    # concat weight
    def neural_concat_nodes(nodes):
        layer_mask = nodes.data['layer_type'] == 1
        layer_idxs = nodes.data['layer_idx'][layer_mask]
        n = layer_idxs[0] 
        return nodes.data['layer_idx'] == n
    n = g.filter_nodes(neural_concat_nodes)
    n = int(n)
    # print(n, g.ndata['kernel_size'][n])
    in_dim = g.ndata['kernel_size'][n][1]
    out_dim = g.ndata['kernel_size'][n][2]
    in_dim, out_dim = int(in_dim), int(out_dim)
    # print(n, " in_dim:", in_dim, " out_dim:", out_dim)
    fc_concat_weight = g.ndata['kernel_weight'][n]
    # reshape fc weight to (in_dim, out_dim)
    fc_concat_weight = unpadding(fc_concat_weight, in_dim, out_dim).t()
    # bias
    fc_concat_bias = g.ndata['bias'][n]
    bias_size = g.ndata['bias_size'][n]
    # print(bias_size)
    r, c = bias_size
    r, c = int(r), int(c)
    # reshape
    # print(fc_concat_bias.shape)
    fc_concat_bias = unpadding(fc_concat_bias, r, c)
    # init fc layer
    # print("Linear params: ", in_dim, out_dim)
    concat_opt = torch.nn.Linear(in_dim, out_dim)
    # print(concat_opt.weight.data.shape, fc_concat_weight.shape)
    concat_opt.weight = Parameter(fc_concat_weight)
    concat_opt.bias.data = Parameter(fc_concat_bias.reshape(512))

    # do concat
    # print("data size: ", data.shape, data.view(1, in_dim).shape)
    fc_concat_result = torch.nn.functional.relu(concat_opt(data.view(1, in_dim).to("cuda")))
    shape = fc_concat_result.size()
    r, c = shape
    r, c = int(r), int(c)
    return n, fc_concat_result, r, c


def encode_fc_feat(data):
    # data shape = (1, n)
    data = padding(data, 1, 23 * 23)
    ret = torch.zeros([23, 1, 23]).to("cuda")
    start = 0
    end = 23
    idx = 0
    while end <= 23 * 23:
        cur_data = data[0][start:end]
        ret[idx] = cur_data.view(1, 23)
        idx += 1
        start += 23 
        end += 23
    # reshape (23, 1, 23) to (1, 1, 28, 28)
    ret = padding(ret.reshape(23, 23), 28, 28)
    return ret.reshape(1, 1, 28, 28)

def  decode_fc_feat(data, n):
    # data shape = (1, 1, 28, 28)
    ret = unpadding(data.reshape(28, 28), 23, 23) # (23, 23)
    ret = torch.cat([ret[i] for i in range(len(ret))]) # (23*23)
    ret = ret.reshape(1, 23*23)
    ret = unpadding(ret, 1, n)
    return ret


def do_linear_opt(nodes, idx, data, size):
    # prepare weight
    in_dim = nodes.data['kernel_size'][idx][1]
    out_dim = nodes.data['kernel_size'][idx][2]
    in_dim, out_dim = int(in_dim), int(out_dim)
    weight = nodes.data['kernel_weight'][idx]
    # (out_dim, in_dim) for calculation optimization
    weight = unpadding(weight, in_dim, out_dim).t()
    # prepare bias
    bias_size = nodes.data['bias_size'][idx]
    r, c = bias_size
    r, c = int(r), int(c)
    bias = nodes.data['bias'][idx] # (1, 512) -> (r, c)
    bias = unpadding(bias, r, c)
    # prepare Linear layer
    fc_opt = torch.nn.Linear(in_dim, out_dim)
    fc_opt.weight = Parameter(weight)
    fc_opt.bias = Parameter(bias.reshape(c))
    # prepare data (1, 1, 28, 28) -> (1, n)
    _, n = size[0]
    data = decode_fc_feat(data, n)

    # do Linear operation
    ret = torch.nn.functional.relu(fc_opt(data.view(1, in_dim).to("cuda")))
    return ret


def my_fc_message(edges):
    return {'mf': edges.src['ft'], 'ms': edges.src['ft_size']}
def my_fc_reduce(nodes):
    mfeat = nodes.mailbox['mf']
    msize = nodes.mailbox['ms']
    
    n_nodes = len(mfeat)
    ret_ft = torch.zeros((n_nodes, 1, 1, 28, 28)).to("cuda")
    ret_size = torch.zeros((n_nodes, 2)).to("cuda") 

    for out_idx in range(n_nodes):
        received_data = mfeat[out_idx]
        received_size = msize[out_idx]

        # skip if all zeros, that is not transformed
        if equals(received_size, zeros=True):
            continue
        # do Linear opt 
        ret = do_linear_opt(nodes, out_idx, received_data, received_size)
        np.savetxt("./intermediate_data/process/ret_fc_ft-%d.csv" % out_idx, ret.cpu().numpy(), delimiter=',')

        shape = ret.size()
        r, c = shape
        r, c = int(r), int(c)

        # set ret feat
        ret_ft[out_idx] = encode_fc_feat(ret)
        ret_size[out_idx] = torch.tensor([r, c], dtype=torch.float32).to("cuda")
    return {'h': ret_ft, 'g':ret_size}