import torch
import pandas as pd
import os
from itertools import product
import argparse
import pytz
from datetime import datetime
import wandb

from logger import CustomLogger
from utils import load_config, serialize_dict, serialize_array, extract_single_value

from data_split import DataSplit

from datasets.deepbind_dataset import DeepBindDataset
from datasets.hyenadna_dataset import HyenaDNA_Dataset
from datasets.bert_tfbs_dataset import BERT_TFBS_dataset

from train_test import Train_Test

from models.hyena_dna import HyenaDNAModel
from baselines.DeepBind import DeepBind
from baselines.BERT_TFBS.bert_tfbs import BERT_TFBS

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
    output_dir = os.path.join(config.output_dir, config.model.model_name,
                              'standard', config.name, config.dataset_split.partition_mode)
    if hasattr(config.model, 'model_version') and config.model.model_version:
        output_dir = os.path.join(config.output_dir, config.model.model_name, config.model.model_version,
                                  'standard', config.name, config.dataset_split.partition_mode)
    os.makedirs(output_dir, exist_ok=True)

    logger = CustomLogger(__name__, log_directory=output_dir, log_file = f'log')

    logger.log_message(f"Configuration loaded: {config}")

    config.device = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    logger.log_message("Using device:", config.device)

    torch.manual_seed(config.dataset_split.random_state)
    torch.cuda.manual_seed_all(config.dataset_split.random_state)

    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    test_result_dir = os.path.join(output_dir, "test_results")
    os.makedirs(test_result_dir, exist_ok=True)

    dfs_train, dfs_val, df_test = DataSplit.split(logger, config.dataset_path,
                                                  os.path.join(config.dataset_split.dir, config.name),
                                                  config.dataset_split, label='label')

    if 'hyenadna' in config.model.model_name:
        ds_test = HyenaDNA_Dataset(df_test, config.model.max_length, config.model.use_padding)
    elif config.model.model_name == 'DeepBind':
        ds_test = DeepBindDataset(df_test, config.model.kernel_length)
    elif 'BERT-TFBS' in config.model.model_name:
        ds_test = BERT_TFBS_dataset(df_test, config.model.max_length)
    else:
        raise ValueError(f'Given model name ({config.model.model_name}) is not valid!')

    tt = Train_Test(logger, config.device, model_dir, config.training)

    if config.model.use_saved_model:  # saved_model_name should be present: for test
        config.training.model_params.batch_size = extract_single_value(config.training.model_params.batch_size)

        model = tt.load(config.model.saved_model_name)
        tt.model = model
        test_accuracy, test_auroc, test_auprc = tt.test(ds_test, config.model.saved_model_name, test_result_dir)

    else: # training
        model_param_values = list(vars(config.training.model_params).values())
        grid_combinations = list(product(*model_param_values))

        logger.log_message(f'There are {len(grid_combinations)} combination of parameters...')

        results = []
        
        # train based on each combination
        for batch_size, learning_rate, weight_decay, freeze_layer in grid_combinations:
            logger.log_message("\n********************************************************************")
            logger.log_message(f"Training with batch_size={batch_size}, learning_rate={learning_rate},"
                        f" weight_decay={weight_decay}, freeze_layer={freeze_layer}")

            # Update model_params for this combination
            config.training.model_params.batch_size = batch_size
            config.training.model_params.learning_rate = learning_rate
            config.training.model_params.weight_decay = weight_decay
            config.training.model_params.freeze_layers = freeze_layer

            # for each fold
            for fold in range(1, config.dataset_split.fold + 1):
                logger.log_message(f'**** Fold {fold}')

                model_name = (f'fold-{fold}_freeze_layer-{serialize_array(freeze_layer)}'
                              f'_{serialize_dict(config.training.model_params)}')

                project_name = f"{config.model.model_name}_{config.name}_{config.dataset_split.partition_mode}"

                # Reload the model fresh each time for the current combination
                if 'hyenadna' in config.model.model_name:
                    tt.model = HyenaDNAModel(logger, pretrained_model_name=config.model.model_name,
                                             use_head=True, device=config.device).load_pretrained_model()
                    ds_train = HyenaDNA_Dataset(dfs_train[fold - 1], config.model.max_length, config.model.use_padding)
                    ds_val = HyenaDNA_Dataset(dfs_val[fold - 1], config.model.max_length, config.model.use_padding)

                elif config.model.model_name == 'DeepBind':
                    tt.model = DeepBind(config.model.kernel_length)
                    ds_train = DeepBindDataset(dfs_train[fold - 1], config.model.kernel_length)
                    ds_val = DeepBindDataset(dfs_val[fold - 1], config.model.kernel_length)

                elif 'BERT-TFBS' in config.model.model_name:
                    ds_train = BERT_TFBS_dataset(dfs_train[fold - 1], config.model.max_length)
                    ds_val = BERT_TFBS_dataset(dfs_val[fold - 1], config.model.max_length)
                    tt.model = BERT_TFBS(config.model.max_length, config.model.pretrained_model_name,
                                         config.model.embedding_size, config.model.model_version)
                    project_name = str(config.model.model_version) + '_' + project_name
                else:
                    raise ValueError(f'Given model name ({config.model.model_name}) is not valid!')


                if config.wandb.enabled:  # visualization with wandb
                    _init_wandb(config.wandb, tt.model, project_name, model_name)

                trainable_params, best_epoch, last_val_acc, last_val_auroc, last_val_auprc\
                    = tt.train(ds_train, ds_val, model_name, wandb)

                test_accuracy, test_auroc, test_auprc = tt.test(ds_test, model_name, test_result_dir)

                results.append({
                        'fold': fold,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'freeze_layer': freeze_layer,
                        'trainable_params': trainable_params,
                        'best_epoch': best_epoch,
                        'last_val_acc': round(last_val_acc, 2),
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
