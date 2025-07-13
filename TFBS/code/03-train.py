import torch
import pandas as pd
import os
from itertools import product
import argparse
import pytz
from datetime import datetime
import wandb

from logger import CustomLogger
from utils import load_config, serialize_dict

from data_split import DataSplit

from datasets.deepbind_dataset import DeepBindDataset
from datasets.foundation_dataset import FoundationDataset

from train_test import Train_Test

from models.hyena_dna import HyenaDNAModel
from models.dna_bert_2 import DNABERT2
from models.deep_bind import DeepBind
from models.BERT_TFBS.bert_tfbs import BERT_TFBS
from models.agro_nt import AgroNTModel

# python 03-train.py --train_config_file [train_config_path] --data_config_file [data_config_path]
# python 03-train.py --train_config_file ../configs/train/HeynaDNA-config.yml --data_config_file ../configs/data_split/cross-species-config.yml


def _init_wandb(wandb_params, model, project_name, run_name):
    try:
        wandb.login(key=wandb_params.token)

        eastern = pytz.timezone(wandb_params.timezone)
        wandb.init(project=f'NEW-{project_name}', entity=wandb_params.entity_name,
                   name=f"{run_name}-{datetime.now(eastern).strftime(wandb_params.timezone_format)}")
        wandb.watch(model, log="all")
    except Exception as e:
        logger.log_message(f"Error initializing Wandb: {e}")
        raise

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_file", type=str, required=True, help="Path to the train config file.")
    parser.add_argument("--data_config_file", type=str, required=True, help="Path to the data_split config file.")
    return parser.parse_args()

def make_name(split_config, is_path=True):
    name = split_config.name
    # if split type is cross, add test_id to output_dir
    if getattr(split_config, 'test_split_type') == 'cross':
        # turn list into '_' delimited str
        raw_ids = split_config.test_ids
        id_str = '_'.join(str(x) for x in raw_ids)

        if is_path:
            name = os.path.join(name, f"test-{id_str}")
        else:
            name = f"{name}-test-{id_str}"
    return name

def make_dirs(config):
    name = make_name(config.split_config)

    output_dir = os.path.join(config.output_dir, name, config.model.model_name,
                              config.split_config.partition_mode)
    # if there's a non‐empty finetune_type, add it to output_dir
    if getattr(config.model, 'finetune_type', None):
        output_dir = os.path.join(config.output_dir, name, config.model.model_name,
                                  config.model.finetune_type, config.split_config.partition_mode)
    # if there's a non‐empty model_version, add it to output_dir
    if getattr(config.model, 'model_version', None):
        output_dir = os.path.join(config.output_dir, name, config.model.model_name,
                                  config.model.model_version, config.model.finetune_type,
                                  config.split_config.partition_mode)

    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    test_result_dir = os.path.join(output_dir, "test_results")
    os.makedirs(test_result_dir, exist_ok=True)

    return output_dir, model_dir, test_result_dir

if __name__ == "__main__":

    args = parse_arguments()
    config = load_config(args.train_config_file, args.data_config_file)
    output_dir, model_dir, test_result_dir = make_dirs(config)

    # make sure eval_batch_size is defined; otherwise use train_batch_size
    if not getattr(config, 'eval_batch_size', None):
        config.eval_batch_size = list(config.training.model_params.train_batch_size)[0]

    logger = CustomLogger(__name__, log_directory=output_dir, log_file=f'log')
    logger.log_message(f"Configuration loaded: {config}")
    logger.log_message(f"Output will save at: {output_dir}")

    config.device = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    logger.log_message("Using device:", config.device)

    torch.manual_seed(config.split_config.random_state)
    torch.cuda.manual_seed_all(config.split_config.random_state)

    dfs_train, dfs_val, df_test = DataSplit.get_splits(logger, config.split_config, label='label')

    if config.model.model_name == "HyenaDNA":
        tokenizer = (HyenaDNAModel(logger, pretrained_model_name=config.model.model_version, device=config.device)
                     .get_tokenizer(config.model.max_length))
        ds_test = FoundationDataset(config.model.model_name, tokenizer, df_test, config.model.max_length,
                                    config.model.use_padding)
    elif config.model.model_name == 'DeepBind':
        ds_test = DeepBindDataset(df_test, config.model.max_length, config.model.kernel_length)
    elif config.model.model_name == 'BERT-TFBS':
        tokenizer = (BERT_TFBS(config.model.max_length)
                     .get_tokenizer())
        ds_test = FoundationDataset(config.model.model_name, tokenizer, df_test, config.model.max_length)
    elif config.model.model_name == 'DNABERT-2':
        tokenizer = (DNABERT2().get_tokenizer())
        ds_test = FoundationDataset(config.model.model_name, tokenizer, df_test, config.model.max_length)
    elif config.model.model_name == "AgroNT":
        tokenizer = AgroNTModel(logger, device=config.device).get_tokenizer()
        ds_test = FoundationDataset(config.model.model_name, tokenizer, df_test, config.model.max_length)
    else:
        raise ValueError(f'Given model name ({config.model.model_name}) is not valid!')

    tt = Train_Test(logger, config.device, model_dir, config.eval_batch_size, config.training)

    results = []

    model_param_values = list(vars(config.training.model_params).values())
    grid_combinations = list(product(*model_param_values))

    logger.log_message(f'There are {len(grid_combinations)} combination of parameters...')

    # train based on each combination
    for freeze_layer, train_batch_size, learning_rate, weight_decay in grid_combinations:
        logger.log_message("\n********************************************************************")
        logger.log_message(f"Training with batch_size={train_batch_size}, learning_rate={learning_rate},"
                    f" weight_decay={weight_decay}, freeze_layer={freeze_layer}")

        # Update model_params for this combination
        config.training.model_params.train_batch_size = train_batch_size
        config.training.model_params.learning_rate = learning_rate
        config.training.model_params.weight_decay = weight_decay
        config.training.model_params.freeze_layers = freeze_layer

        # for each fold
        for fold in range(1, config.split_config.fold + 1):
            logger.log_message(f'**** Fold {fold}')

            run_name = (f'fold-{fold}_{serialize_dict(config.training.model_params)}')

            name = make_name(config.split_config, is_path=False)
            project_name = f"{config.model.model_name}_{name}_{config.split_config.partition_mode}"

            # Reload the model fresh each time for the current combination
            if config.model.model_name == "HyenaDNA":
                tt.model = (HyenaDNAModel(logger, pretrained_model_name=config.model.model_version, device=config.device)
                    .load_pretrained_model())
                ds_train = FoundationDataset(config.model.model_name, tokenizer, dfs_train[fold - 1],
                                            config.model.max_length, config.model.use_padding)
                ds_val = FoundationDataset(config.model.model_name, tokenizer, dfs_val[fold - 1],
                                          config.model.max_length, config.model.use_padding)
                project_name = config.model.model_version + '_' + project_name

            elif config.model.model_name == 'DeepBind':
                tt.model = DeepBind(config.model.kernel_length)
                ds_train = DeepBindDataset(dfs_train[fold - 1], config.model.max_length, config.model.kernel_length)
                ds_val = DeepBindDataset(dfs_val[fold - 1], config.model.max_length, config.model.kernel_length)

            elif config.model.model_name == 'BERT-TFBS':
                ds_train = FoundationDataset(config.model.model_name, tokenizer, dfs_train[fold - 1],
                                             config.model.max_length)
                ds_val = FoundationDataset(config.model.model_name, tokenizer, dfs_val[fold - 1], config.model.max_length)
                tt.model = BERT_TFBS(config.model.max_length)

            elif config.model.model_name == 'DNABERT-2':
                ds_train = FoundationDataset(config.model.model_name, tokenizer, dfs_train[fold - 1],
                                             config.model.max_length)
                ds_val = FoundationDataset(config.model.model_name, tokenizer, dfs_val[fold - 1],
                                           config.model.max_length)
                tt.model = DNABERT2()

            elif config.model.model_name=="AgroNT":
                tt.model = (AgroNTModel(logger, device=config.device)
                    .load_pretrained_model(config.model.finetune_type))
                ds_train = FoundationDataset(config.model.model_name, tokenizer, dfs_train[fold - 1],
                                             config.model.max_length, config.model.use_padding)
                ds_val = FoundationDataset(config.model.model_name, tokenizer, dfs_val[fold - 1],
                                           config.model.max_length, config.model.use_padding)
            else:
                raise ValueError(f'Given model name ({config.model.model_name}) is not valid!')

            if config.wandb.enabled:  # visualization with wandb
                _init_wandb(config.wandb, tt.model, project_name, run_name)

            trainable_params, best_epoch, last_val_acc, last_val_auroc, last_val_auprc, last_val_f1, last_val_mcc, train_time \
                = tt.train(ds_train, ds_val, run_name, wandb)

            logger.log_message("Getting test set accuracy...")
            test_accuracy, test_auroc, test_auprc, test_f1, test_mcc, test_time \
                = tt.test(ds_test, run_name, test_result_dir)

            results.append({
                'fold': fold,
                'train_batch_size': train_batch_size,
                'eval_batch_size': config.eval_batch_size,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'freeze_layer': freeze_layer,
                'trainable_params': trainable_params,
                'train_time(s)': train_time,
                'best_epoch': best_epoch,
                'last_val_acc': round(last_val_acc, 2),
                'last_val_f1': round(last_val_f1, 2),
                'last_val_mcc': round(last_val_acc, 2),
                'last_val_auroc': round(last_val_auroc, 2),
                'last_val_auprc': round(last_val_auprc, 2),
                'test_accuracy': round(test_accuracy, 2),
                'test_f1': round(test_f1, 2),
                'test_mcc': round(test_mcc, 2),
                'test_auroc': round(test_auroc, 2),
                'test_auprc': round(test_auprc, 2),
                'test_time(s)': test_time
            })

            if config.wandb.enabled:
                wandb.finish()

    # get metrics for test set
    df_results = pd.DataFrame(results)
    results_csv_file = os.path.join(output_dir, 'hyper_params.csv')
    df_results.to_csv(results_csv_file, index=False)
    logger.log_message(f"Grid search results saved to {results_csv_file}", use_time=True)
