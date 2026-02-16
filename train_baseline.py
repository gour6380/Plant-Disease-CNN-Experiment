from config import Config
from train import train_from_config
from utils import select_device, setup_config_as_per_device

if __name__ == "__main__":
    Config.device = select_device(Config.device)
    Config= setup_config_as_per_device(Config)
    Config.model_name = "baseline"
    train_from_config(Config)
