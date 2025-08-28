import torch
import argparse
from logger import CustomLogger
from utils import load_config
from data_split import DataSplit

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config_file)

    logger = CustomLogger(__name__, log_directory=config.split_dir, log_file=f'{config.name}.log')
    logger.log_message(f"\n\n*********** Configuration loaded: {config}")

    torch.manual_seed(config.random_state)
    torch.cuda.manual_seed_all(config.random_state)

    DataSplit.split(logger, config, label='label')