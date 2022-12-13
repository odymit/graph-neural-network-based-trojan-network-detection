import torch
from utils_gnn import equals
def test_act_equals():
    # case 1, ones equal
    x = torch.ones((2,2,2))
    y = torch.ones((2,2,2))
    ret = equals(x, y)
    assert ret == True, "x != y in case 1"
    # case 2, ones not equal
    x = torch.ones((2,2,2))
    y = torch.rand((2,2,2))
    ret = equals(x, y)
    assert ret == False, "x == y in case 2"
    # case 3, zeros
    x = torch.zeros((2,2,2)).to("cuda")
    ret = equals(x, zeros=True)
    assert ret == True, "x != y in case 3"