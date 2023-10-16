import torch


class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 1
    MODEL_DEPTH = 4
    MODEL_START_CHANNELS = 64
    TIME_EMB_DIM = 32
    LEARNING_RATE = 1e-4
    T = 1000
    EPOCHS = 1000
    VISUALIZE = True
    LOAD_MODEL = True
    DATA_PATH = 'data/reveal'
    MODEL_PATH = 'model/model.pth'
