import os
from os.path import exists

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import random_split
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.data.data import BaseData
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm


class CPGDataset(GeometricDataset):
    """
    CPGDataset is a PyTorch Geometric Dataset subclass for handling pre-processed Code Property Graphs (CPGs).
    """

    def __init__(self, path_to_data_folder, node_features=178, classes=2):
        """
        Constructor for CPGDataset class.

        Initializes the dataset by specifying the path to the data folder, the number of node features and the number of classes.

        Args:
            path_to_data_folder (str): Path to the directory that contains the .pt files.
            node_features (int, optional): The number of node features in each graph. Defaults to 178.
            classes (int, optional): The number of unique classes in the dataset. Defaults to 2.
        """
        self.folder_path = path_to_data_folder
        self.files = self.get_file_list()
        self.node_features = node_features
        self.classes = classes
        super(CPGDataset, self).__init__('.')

    @property
    def raw_file_names(self):
        """
        Property that returns the list of .pt files.

        Returns:
            list: List of .pt file paths.
        """
        return self.files

    @property
    def processed_file_names(self):
        """
        Property that returns the list of processed .pt files.

        Returns:
            list: List of processed .pt file paths.
        """
        return self.files

    def download(self):
        """
        Method to download the dataset.

        As the dataset is assumed to be locally available, this method does not perform any operation.
        """
        pass

    def process(self):
        """
        Method to process the dataset.

        As the dataset is assumed to be pre-processed, this method does not perform any operation.
        """
        pass

    def len(self):
        """
        Method to get the number of .pt files in the directory.

        Returns:
            int: Number of .pt files in the directory.
        """
        return len(self.files)

    def get(self, idx):
        """
        Method to get a pre-processed CPG given an index.

        Args:
            idx (int): Index of the .pt file.

        Returns:
            BaseData: Deserialized BaseData object from the .pt file at the specified index.
        """
        file_path = self.files[idx]
        data = torch.load(file_path)
        return data

    @property
    def num_node_features(self):
        """
        Method to get the number of node features in the graphs.

        Returns:
            int: Number of node features.
        """
        return self.node_features

    @property
    def num_classes(self):
        """
        Property to get the number of unique classes in the dataset.

        Returns:
            int: Number of unique classes.
        """
        return self.classes

    def get_file_list(self):
        """
        Method to get a list of all .pt files in the directory specified by folder_path.

        Returns:
            list: List of paths to .pt files in the directory.
        """
        file_list = []
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.pt'):
                file_list.append(os.path.join(self.folder_path, file_name))
        return file_list

    def train_test_split(self, train_ratio=0.8):
        """
        Method to perform a train-test split on the dataset.

        Args:
            train_ratio (float, optional): The ratio of the training set size to the total dataset size. Defaults to 0.8.

        Returns:
            tuple: A tuple containing two Dataset objects (train_dataset, test_dataset).
        """
        train_size = int(self.len() * train_ratio)
        test_size = self.len() - train_size
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        return train_dataset, test_dataset


class AdjacencyCPGDataset(TorchDataset):
    def __init__(self, raw_dir, model_depth):
        super().__init__()

        self.raw_dir = raw_dir
        self.processed_dir = 'processed'
        self.model_depth = model_depth

        if not exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            self.process()

        self.processed_file_names = self.process_file_names()

    def process_file_names(self):
        """
        Property that returns a list of processed .pt files sorted by the get_dim method.
        Returns:
            list: Sorted list of paths to processed .pt files.
        """
        return [f for f in os.listdir(self.processed_dir) if f.endswith('.pt')]

    def process(self):
        """
        Method to process the dataset, converting adjacency information into tensors.
        """
        dataset = CPGDataset(self.raw_dir)

        for file_path in tqdm(dataset.files):
            data = torch.load(file_path)
            adj = to_dense_adj(data.edge_index)
            if adj.shape[1] > 100:
                adj = self.adjust_dimensions(adj)
                adj = torch.clamp(2 * adj - 1, -1, 1)

                torch.save(adj, os.path.join(self.processed_dir, os.path.basename(file_path) + '.pt'))

    def adjust_dimensions(self, adjacency_matrix):
        """
        Adjusts the dimensions of the adjacency matrix to match 2 to the power of the model_depth which is necessary for the Unet model.
        Args:
            adjacency_matrix (torch.Tensor): The adjacency matrix to be adjusted.
        Returns:
            torch.Tensor: The adjusted adjacency matrix.
        """
        divisor = 2 ** self.model_depth
        num_nodes = adjacency_matrix.shape[1]
        pad_nodes = divisor - (num_nodes % divisor) if num_nodes % divisor != 0 else 0

        return F.pad(adjacency_matrix, (0, pad_nodes, 0, pad_nodes))

    def get_dim(self, file_path):
        """
        Gets the size of the tensor at the specified index in the dataset.
        Args:
            index (int): Index of the tensor to get size of.
        Returns:
            torch.Size: The size of the tensor at the specified index.
        """
        file_path = os.path.join(self.processed_dir, file_path)
        adjacency_matrix = torch.load(file_path)
        return adjacency_matrix.shape[1]

    def __len__(self):
        """
        Gets the length of the dataset.
        Returns:
            int: The length of the dataset.
        """
        return len(self.processed_file_names)

    def __getitem__(self, index):
        """
        Gets the item at the specified index in the dataset.
        Args:
            index (int): Index of the item to get.
        Returns:
            torch.Tensor: The item (processed adjacency matrix) at the specified index.
        """
        file_path = os.path.join(self.processed_dir, self.processed_file_names[index])
        adjacency_matrix = torch.load(file_path)

        return adjacency_matrix
