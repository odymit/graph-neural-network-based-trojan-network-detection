from utils_gnn import padding
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




