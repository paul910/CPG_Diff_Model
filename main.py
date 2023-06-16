from config import Config
from dataloader import get_adj_dataloader
from model.model_manager import ModelManager
from utils.utils import DiffusionUtils, show_tensor_image
import matplotlib.pyplot as plt


def main():
    config = Config()
    diffusion_utils = DiffusionUtils(config)
    dataloader = get_adj_dataloader(config.DATA_PATH, config.BATCH_SIZE, config.MODEL_DEPTH)

    model_manager = ModelManager(config, diffusion_utils, dataloader)
    model_manager.train_model()


if __name__ == "__main__":
    main()
