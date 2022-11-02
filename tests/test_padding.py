from utils_gnn import padding, unpadding
import torch 

def test_padding_normal():
    # create tensor
    ones = torch.ones((2,2))
    # row > original_row, col > original_col
    big = padding(ones, 4, 5)
    row, col = big.size()
    assert row == 4
    assert col == 5
    big = padding(ones, 5, 4)
    row, col = big.size()
    assert row == 5
    assert col ==4
    big = padding(ones, 3, 3)
    row, col = big.size()
    assert row == 3
    assert col == 3
    # row == original_row, col == original_col 
    same = padding(ones, 2, 2)
    row, col = same.size()
    assert row == 2
    assert col == 2
    # ones test
    ones = torch.ones((1,1))
    one_pad = padding(ones, 1, 512)
    row, col = one_pad.size()
    assert row == 1
    assert col == 512

def test_unpadding_normal():
    pad_list = [[3,3], [4,5], [5,4], [512, 512]]
    ori_list = [[3,3], [2,3], [3,2]]
    # normal case
    for size in ori_list:
        row_o = size[0]
        col_o = size[0]
        norm = torch.ones((row_o, col_o))
        for padd_size in pad_list:
            r = padd_size[0]
            c = padd_size[1]
            pad = padding(norm, r, c)
            unpad = unpadding(pad, row_o, col_o)
            torch.save(unpad, './unpad.txt')
            row, col = unpad.size()
            assert row == row_o
            assert col == col_o

    # one-dim case
    ones = torch.ones((1,1))
    # pad_list = [[3,3], [4,5], [5,4], [512, 512]]
    pad = padding(ones, 1, 512)
    unpad = unpadding(pad, 1, 1)
    row, col = unpad.size()
    assert row == 1
    assert col == 1
    # full case
    ones = torch.ones((1,512))
    pad = padding(ones, 1, 512)
    unpad = unpadding(pad, 1, 512)
    row, col = unpad.size()
    assert row == 1
    assert col == 512


