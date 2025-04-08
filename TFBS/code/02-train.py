import torch
import pandas as pd
import os
from itertools import product
import argparse

from standard_fine_tune import FineTune
from dna_dataset import DNADataset
from data_split import DataSplit
from utils import load_config, get_file_name, serialize_dict, serialize_array
from visualization import plot_loss, plot_auroc_auprc
from hyena_dna import HyenaDNAModel

from utils import extract_single_value, extract_value_as_list

from logger import CustomLogger


# python 02-train.py --config_file "../configs/standard_config.yml"
# python 02-train.py  --config_file "../configs/standard_cross-species_config.yml"
# python 02-train.py  --config_file "../configs/standard_cross-dataset-Ronan_Josey-config.yml"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file.")
    return parser.parse_args()


def get_unique(df):
    print(f'number of original rows: {len(df)}')
    subset = ['chromosomeId', 'sequence']
    redundant_df = df[df.duplicated(subset=subset, keep=False)]
    print(f'number of redundant rows for {subset}: {len(redundant_df)}')

    redundant_df = df[df.duplicated(subset=subset)]
    print(f'number of removing redundant rows for {subset}: {len(redundant_df)}')

    unique_df = df.loc[~df.duplicated(subset=subset, keep='first')]
    print(f'number of unique rows for {subset}: {len(unique_df)}')
    return unique_df

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

    df = get_unique(read_dataset(config.paths.dataset_path))

    model_dir = os.path.join(config.paths.model_dir, config.model.pretrained_model_name, config.dataset_split.split_type)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    plot_dir = os.path.join(config.paths.plot_dir, config.model.pretrained_model_name, config.dataset_split.split_type)
    os.makedirs(plot_dir, exist_ok=True)

    test_result_dir = os.path.join(config.paths.test_result_dir, config.model.pretrained_model_name, config.dataset_split.split_type)
    os.makedirs(test_result_dir, exist_ok=True)

    logger = CustomLogger(__name__, log_directory=config.paths.log_dir, log_file = f'log_{get_file_name(args.config_file)}')

    data_split = DataSplit(logger, df, config.paths.dataset_split_path, config.dataset_split.split_type,
                           config.dataset_split.train_size, config.dataset_split.random_state)
    df_train, df_val, df_test = data_split.split(config.dataset_split.id_column, config.dataset_split.train_ids,
                                                 config.dataset_split.val_ids, config.dataset_split.test_ids)

    tokenizer = HyenaDNAModel.get_tokenizer(config.model.model_max_length)

    ds_train = DNADataset(df_train, tokenizer, config.model.model_max_length, config.model.use_padding)
    ds_val = DNADataset(df_val, tokenizer, config.model.model_max_length, config.model.use_padding)
    ds_test = DNADataset(df_test, tokenizer, config.model.model_max_length, config.model.use_padding)

    ft = FineTune(logger, config.device, model_dir, config.training)

    if config.model.use_saved_model:  # saved_finetuned_model_name should be present
        config.training.model_params.batch_size = extract_single_value(config.training.model_params.batch_size)

        model = ft.load(config.model.saved_finetuned_model_name)
        test_accuracy = ft.test(ds_test, test_result_dir)

    else:

        grid_combinations = list(product(config.training.model_params.batch_size,
                                         config.training.model_params.learning_rate,
                                         config.training.model_params.weight_decay,
                                         config.training.freeze_layers,
                                         ))

        logger.log_message(f'There are {len(grid_combinations)} combination of parameters...')

        results = []
        param_combinations = []
        train_loss_per_param = []
        val_loss_per_param = []
        auroc_per_param = []
        auprc_per_param = []
        
        # finetune based on each combination
        for batch_size, learning_rate, weight_decay, freeze_layer in grid_combinations:
            logger.log_message("\n********************************************************************")
            logger.log_message(f"Training with batch_size={batch_size}, learning_rate={learning_rate},"
                        f" weight_decay={weight_decay}, freeze_layer={freeze_layer}")

            # Update model_params for this combination
            config.training.model_params.batch_size = batch_size
            config.training.model_params.learning_rate = learning_rate
            config.training.model_params.weight_decay = weight_decay
            config.training.freeze_layers = freeze_layer

            param_combinations.append(
                f'{serialize_dict(config.training.model_params)}_freeze_layer-{serialize_array(config.training.freeze_layers)}')

            # Reload the pretrained model fresh each time for the current combination
            ft.model = HyenaDNAModel(logger, pretrained_model_name=config.model.pretrained_model_name,
                                     use_head=True, device=config.device).load_pretrained_model()

            model, trainable_params, train_losses, val_losses, auroc_per_epoch, auprc_per_epoch = (
                ft.finetune(ds_train, ds_val))
            train_loss_per_param.append(train_losses)
            val_loss_per_param.append(val_losses)
            auroc_per_param.append(auroc_per_epoch)
            auprc_per_param.append(auprc_per_epoch)
            
            test_accuracy, auroc, auprc = ft.test(ds_test, test_result_dir)
            results.append({
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'freeze_layer': freeze_layer,
                        'trainable_params': trainable_params,
                        'accuracy': test_accuracy,
                        'auroc': auroc,
                        'auprc': auprc
                    })
            
        config_name = get_file_name(args.config_file)

        # plot validation metrics for each epoch
        plot_loss(param_combinations, train_loss_per_param, val_loss_per_param, plot_dir, f'loss_{config_name}.pdf')
        plot_auroc_auprc('AUROC', param_combinations, auroc_per_param, plot_dir, config_name)
        plot_auroc_auprc('AUPRC', param_combinations, auprc_per_param, plot_dir, config_name)

        # get metrics for test set
        df_results = pd.DataFrame(results)
        results_csv_path = f'../outputs/hyper_params_{get_file_name(args.config_file)}.csv'
        df_results.to_csv(results_csv_path, index=False)
        logger.log_message(f"Grid search results saved to {results_csv_path}", use_time=True)
