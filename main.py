from config import Config
from model.model_manager import ModelManager
from utils.utils import DiffusionUtils


def main():
    config = Config()
    diffusion_utils = DiffusionUtils(config)
    model_manager = ModelManager(config, diffusion_utils)
    model_manager.train_model()


if __name__ == "__main__":
    main()
