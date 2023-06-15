import torch


class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 8
    MODEL_DEPTH = 4
    LEARNING_RATE = 1e-4
    T = 1000
    EPOCHS = 1000
    VISUALIZE = False
    DATA_PATH = 'data/reveal'
    LOAD_MODEL = False
    MODEL_PATH = 'model/model.pth'
