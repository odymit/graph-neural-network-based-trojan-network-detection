from cogdl.data import Graph
from cogdl.datasets import GraphDataset
import torch
from tqdm import tqdm
import os
import sys
import psutil

class MyGraphDataset(GraphDataset):
    def __init__(self, dir=None, path="/home/ubuntu/date/hdd4/shadow_model_ckpt/mnist/processed_data/train-data.pt"):
        self.path = path
        # preprocessed data directory
        self.dir = dir
        super(MyGraphDataset, self).__init__(path, metric="accuracy")

    def process(self):
        # Load and preprocess data
        graphs = []
        if not os.path.exists(self.dir):
            raise FileExistsError

        for file in tqdm(os.listdir(self.dir)):
            if 'train-data.pt' in file:
                continue
            real_path = os.path.join(self.dir, file)
            data = torch.load(real_path)
            # print(data)
            graphs.append(Graph(edge_index=data['edges'], x=data['x'], y=data['y']))
            usage = psutil.virtual_memory().percent
            if usage >= 95.0 or len(graphs) > 1500:
                print("almost read the max memory limit, exiting reading dataset...")
                break

        return graphs
    

    
from cogdl import experiment


if __name__ == "__main__":
    dataset = MyGraphDataset(dir="/home/ubuntu/date/hdd4/shadow_model_ckpt/mnist/processed_data/")
    print("dataset process and load successfully!")
    experiment(model=sys.argv[1], dataset=dataset, num_workers=0, batch_size=2, epochs=200, devices=[int(sys.argv[2])])
