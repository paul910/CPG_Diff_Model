import os

import torch
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from tqdm import tqdm


class CPGDataset(Dataset):
    """
    CPGDataset is a PyTorch Geometric Dataset subclass for handling pre-processed Code Property Graphs (CPGs).
    """

    def __init__(self, path_to_data_folder, node_features=178, classes=2, transform=None, pre_transform=None):
        """
        Constructor for CPGDataset class.

        Initializes the dataset by specifying the path to the data folder, the number of node features, the number of classes,
        and any necessary transform/pre-transform functions.

        Args:
            path_to_data_folder (str): Path to the directory that contains the .pt files.
            node_features (int, optional): The number of node features in each graph. Defaults to 178.
            classes (int, optional): The number of unique classes in the dataset. Defaults to 2.
            transform (callable, optional): A function/transform that takes in an BaseData object and returns a transformed version.
                The transform is applied every time a data object is accessed. Defaults to None.
            pre_transform (callable, optional): A function/transform that takes in an BaseData object and returns a transformed version.
                The pre_transform is applied only once in the process method. Defaults to None.
        """
        self.folder_path = path_to_data_folder
        self.files = self.get_file_list()
        self.node_features = node_features
        self.classes = classes
        super(CPGDataset, self).__init__('.', transform, pre_transform)

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
        print('Loading CPGs...')
        for file_name in tqdm(os.listdir(self.folder_path)):
            if file_name.endswith('.pt'):
                file_list.append(os.path.join(self.folder_path, file_name))
        return file_list
