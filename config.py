import torch


class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 4
    MODEL_DEPTH = 2
    MODEL_START_CHANNELS = 64
    TIME_EMB_DIM = 32
    LEARNING_RATE = 1e-4
    T = 1000
    EPOCHS = 1000
    VISUALIZE = True
    DATA_PATH = 'data/reveal'
    LOAD_MODEL = True
    MODEL_PATH = 'model/model.pth'
