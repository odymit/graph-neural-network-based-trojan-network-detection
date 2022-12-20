import torch
from utils_gnn import prepare_data

def test_prepare_data():
    # equals 
    data = torch.zeros((8, 1, 1, 28, 28))
    size = torch.tensor([[28, 28] for i in range(8)])
    ret = prepare_data(data, size)
    _, n, width, height = ret.size()
    assert n == 8 and width == 28 and height == 28, "case 1: (1,1,28,28) equals."
    # not equals
    size = torch.tensor([[14, 14] for i in range(8)])
    ret = prepare_data(data, size)
    _, n, width, height = ret.size()
    assert n == 8 and width == 14 and height == 14, "case 2: (1,1,14,14) equals."