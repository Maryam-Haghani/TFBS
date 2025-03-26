import torch
from torch.nn.functional import softmax
import torch.nn.functional as F
from Bio import SeqIO
import numpy as np
import pandas as pd
import os
import sys
from itertools import product
import argparse

from standard_fine_tune import FineTune
from embedding import Embedding
from dna_dataset import DNADataset
from data_split import DataSplit
from utils import dict_to_namespace, load_config
from hyena_dna import HyenaDNAModel

from utils import extract_single_value, extract_value_as_list

from logger import CustomLogger


# python 02-train.py --config_file "../configs/standard_config.yml"
# python 02-train.py --config_file "../configs/standard_cross-species_config.yml"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file.")
    return parser.parse_args()

def read_dataset(dataset_paths):
    dfs = []

    dataset_paths = extract_value_as_list(dataset_paths)
    print(dataset_paths)

    # Iterate through each file path and read the CSV into a DataFrame
    for path in dataset_paths:
        df = pd.read_csv(path)
        dfs.append(df)

    # Concatenate all DataFrames into one
    df = pd.concat(dfs, ignore_index=True)

    df['sequence'] = df['sequence'].str.upper()
    # randomly rearrange the rows of df (shuffle rows)
    random_state = 1972934
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df

if __name__ == "__main__":

    args = parse_arguments()

    # Load the configuration
    config = load_config(args.config_file)
    print(f"Configuration loaded: {config}")

    device = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    config.device = device
    print("Using device:", config.device)

    # Extract sections from the config
    paths = config.paths
    dataset_split_config = config.dataset_split

    df = read_dataset(config.paths.dataset_path)

    model_dir = os.path.join(config.paths.model_dir, config.model.pretrained_model_name, config.dataset_split.split_type)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    loss_dir = os.path.join(config.paths.loss_dir, config.model.pretrained_model_name, config.dataset_split.split_type)
    os.makedirs(loss_dir, exist_ok=True)

    test_result_dir = os.path.join(config.paths.test_result_dir, config.model.pretrained_model_name, config.dataset_split.split_type)
    os.makedirs(test_result_dir, exist_ok=True)

    logger = CustomLogger(__name__, log_directory=config.paths.log_dir, log_file = config.log_file).get_logger()

    data_split = DataSplit(logger, df, config.paths.dataset_split_path, config.dataset_split.split_type,
                           config.dataset_split.train_size, config.dataset_split.random_state)
    df_train, df_val, df_test = data_split.split(config.dataset_split.id_column, config.dataset_split.train_ids,
                                                 config.dataset_split.val_ids, config.dataset_split.test_ids)
    
    tokenizer = HyenaDNAModel.get_tokenizer(config.model.model_max_length)

    ds_train = DNADataset(df_train, tokenizer, config.model.model_max_length, config.model.use_padding)
    ds_val = DNADataset(df_val, tokenizer, config.model.model_max_length, config.model.use_padding)
    ds_test = DNADataset(df_test, tokenizer, config.model.model_max_length, config.model.use_padding)


    if config.model.use_saved_model:  # saved_finetuned_model_name should be present
        config.training.model_params.batch_size = extract_single_value(config.training.model_params.batch_size)

        ft = FineTune(logger, config.model.pretrained_model_name, config.device, model_dir, config.training)
        model = ft.load(config.model.saved_finetuned_model_name)

        test_accuracy = ft.test(ds_test, test_result_dir)

    else:
        grid_combinations = list(product(config.training.model_params.batch_size,
                                         config.training.model_params.learning_rate,
                                         config.training.model_params.weight_decay))
        
        logger.info(f'There are {len(grid_combinations)} combination of parameters...')

        results = []
        # finetune based on each combination
        for batch_size, learning_rate, weight_decay in grid_combinations:
            logger.info(f"Training with batch_size={batch_size}, learning_rate={learning_rate}, weight_decay={weight_decay}")

            # Update model_params for this combination
            config.training.model_params.batch_size = batch_size
            config.training.model_params.learning_rate = learning_rate
            config.training.model_params.weight_decay = weight_decay

            ft = FineTune(logger, config.model.pretrained_model_name, config.device, model_dir, config.training)
            model = ft.finetune(ds_train, ds_val, loss_dir)

            test_accuracy = ft.test(ds_test, test_result_dir)

            results.append({
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'test_accuracy': test_accuracy
                    })

        df_results = pd.DataFrame(results)
        results_csv_path = f'../outputs/hyper_params.csv'
        df_results.to_csv(results_csv_path, index=False)
        logger.info(f"Grid search results saved to {results_csv_path}")
