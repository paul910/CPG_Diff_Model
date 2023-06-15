from dataset import CPGDataset
from model import DiffusionModel
from torch_geometric.loader import DataLoader


def main():

    path_to_data_folder = 'data/reveal'

    dataset = CPGDataset(path_to_data_folder)
    train_dataset, val_dataset = dataset.train_test_split()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    main()