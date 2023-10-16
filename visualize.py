import torch
import matplotlib.pyplot as plt
import numpy as np

from config import Config
import utils.utils as utils
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import dense_to_sparse, to_torch_coo_tensor, to_networkx

from dataloader import get_adj_dataloader


def main():

    dataloader = get_adj_dataloader(Config.DATA_PATH, Config.BATCH_SIZE, Config.MODEL_DEPTH)
    for i, data in enumerate(dataloader):
        adj = data[0][0]
        if adj.shape[-1] != 128:
            continue
        adj = adj.numpy()
        plt.imshow(adj, cmap='gray')
        print(np.min(adj), np.max(adj))
        plt.show()
        break

    '''

    num_nodes = 128
    for i in range(9):
        adj = torch.load(f"output/graphs/graph_{num_nodes}_{i}.pt", map_location=torch.device('cpu'))
        
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        adj = adj.squeeze().squeeze().numpy()
        adj = utils.normalize(adj)
        plt.imshow(adj, cmap='gray')
        plt.show()
    '''

    adj = torch.randn(128,128)
    adj = torch.where(adj < 1.5, 0.0, 1.0)

    plt.imshow(adj.squeeze().numpy(), cmap='gray')
    plt.show()




if __name__ == "__main__":
    main()
