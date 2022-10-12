from ..hetero_struct_backdoor_dataset import HeteroStrucBackdoorDataset
from ..homo_struct_backdoor_dataset import HomoStrucBackdoorDataset

from dgl.dataloading import GraphDataLoader
import torch
import dgl
import os
from dgl.heterograph import DGLHeteroGraph

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'


def test_hetero():
    dataset = HeteroStrucBackdoorDataset()
    dataloader = GraphDataLoader(dataset, batch_size=4, pin_memory=torch.cuda.is_available())
    for batch, (batched_graph, labels) in enumerate(dataloader):
        assert batch == 0
        assert isinstance(batched_graph, DGLHeteroGraph)
        break


def test_homo():
    dataset = HomoStrucBackdoorDataset()
    dataloader = GraphDataLoader(dataset, batch_size=4, pin_memory=torch.cuda.is_available())
    for batch, (batched_graph, labels) in enumerate(dataloader):
        assert batch == 0
        assert isinstance(batched_graph, DGLHeteroGraph)
        break
