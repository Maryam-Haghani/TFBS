import torch
import pandas as pd
import os
from itertools import product
import argparse
import pytz
from datetime import datetime
import wandb

from standard_fine_tune import FineTune
from dna_dataset import DNADataset
from data_split import DataSplit
from utils import load_config, serialize_dict, serialize_array, extract_single_value
from hyena_dna import HyenaDNAModel
from logger import CustomLogger


# python 02-train.py --config_file "../configs/standard_config.yml"

def _init_wandb(wandb_params, model, project_name, run_name):
    try:
        wandb.login(key=wandb_params.token)

        eastern = pytz.timezone(wandb_params.timezone)
        wandb.init(project=project_name, entity=wandb_params.entity_name,
                   name=f"{run_name}-{datetime.now(eastern).strftime(wandb_params.timezone_format)}")
        wandb.watch(model, log="all")
    except Exception as e:
        logger.log_message(f"Error initializing Wandb: {e}")
        raise

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file.")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    config = load_config(args.config_file)
    output_dir = os.path.join(config.output_dir, config.model.pretrained_model_name,
                              'standard', config.name, config.dataset_split.partition_mode)
    os.makedirs(output_dir, exist_ok=True)

    logger = CustomLogger(__name__, log_directory=output_dir, log_file = f'log')

    logger.log_message(f"Configuration loaded: {config}")

    config.device = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    logger.log_message("Using device:", config.device)

    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    test_result_dir = os.path.join(output_dir, "test_results")
    os.makedirs(test_result_dir, exist_ok=True)

    tokenizer = HyenaDNAModel.get_tokenizer(config.model.model_max_length)

    ft = FineTune(logger, config.device, model_dir, config.training)

    dfs_train, dfs_val, df_test = DataSplit.split(logger, config.dataset_path,
                                                  os.path.join(config.dataset_split.dir, config.name),
                                                  config.dataset_split, label='label')

    ds_test = DNADataset(df_test, tokenizer, config.model.model_max_length, config.model.use_padding)

    input("Exit!")


    if config.model.use_saved_model:  # saved_finetuned_model_name should be present
        config.training.model_params.batch_size = extract_single_value(config.training.model_params.batch_size)

        model = ft.load(config.model.saved_finetuned_model_name)
        test_accuracy, test_auroc, test_auprc = ft.test(ds_test, config.model.saved_finetuned_model_name, test_result_dir)

    else:
        grid_combinations = list(product(config.training.model_params.batch_size,
                                         config.training.model_params.learning_rate,
                                         config.training.model_params.weight_decay,
                                         config.training.freeze_layers,
                                         ))

        logger.log_message(f'There are {len(grid_combinations)} combination of parameters...')

        results = []
        
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

            # for each fold
            for fold in range(1, config.dataset_split.fold + 1):
                logger.log_message(f'**** Fold {fold}')

                model_name = (f'fold-{fold}_{serialize_dict(config.training.model_params)}'
                                   f'_freeze_layer-{serialize_array(freeze_layer)}')

                # Reload the pretrained model fresh each time for the current combination
                ft.model = HyenaDNAModel(logger, pretrained_model_name=config.model.pretrained_model_name,
                                         use_head=True, device=config.device).load_pretrained_model()

                project_name = f"{config.model.pretrained_model_name}_{config.name}_{config.dataset_split.partition_mode}"

                if config.wandb.enabled:  # visualization with wandb
                    _init_wandb(config.wandb, ft.model, project_name, model_name)

                ds_train = DNADataset(dfs_train[fold-1], tokenizer, config.model.model_max_length, config.model.use_padding)
                ds_val = DNADataset(dfs_val[fold-1], tokenizer, config.model.model_max_length, config.model.use_padding)

                trainable_params, best_epoch, last_val_acc, last_val_auroc, last_val_auprc\
                    = ft.finetune(ds_train, ds_val, model_name, wandb, )
                test_accuracy, test_auroc, test_auprc = ft.test(ds_test, model_name, test_result_dir)

                results.append({
                            'fold': fold,
                            'batch_size': batch_size,
                            'learning_rate': learning_rate,
                            'weight_decay': weight_decay,
                            'freeze_layer': freeze_layer,
                            'trainable_params': trainable_params,
                            'best_epoch' : best_epoch,
                            'last_val_acc' : round(last_val_acc, 2),
                            'last_val_auroc': round(last_val_auroc, 2),
                            'last_val_auprc': round(last_val_auprc, 2),
                            'test_accuracy': round(test_accuracy, 2),
                            'test_auroc': round(test_auroc, 2),
                            'test_auprc': round(test_auprc, 2)
                        })

                if config.wandb.enabled:
                    wandb.finish()

        # get metrics for test set
        df_results = pd.DataFrame(results)
        results_csv_file = os.path.join(output_dir, 'hyper_params.csv')
        df_results.to_csv(results_csv_file, index=False)
        logger.log_message(f"Grid search results saved to {results_csv_file}", use_time=True)
