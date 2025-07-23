import torch
import pandas as pd
import os
from itertools import product
import argparse
import wandb


from logger import CustomLogger
from utils import load_config, serialize_dict, make_dirs, make_folder_name, init_wandb
from model_utils import init_model_and_tokenizer, get_ds, set_device, get_model_name

from data_split import DataSplit
from train_test import Train_Test

# python 03-train.py --train_config_file [train_config_path] --split_config_file [data_config_path]
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_file", type=str, required=True, help="Path to the train config file.")
    parser.add_argument("--split_config_file", type=str, required=True, help="Path to the data_split config file.")
    return parser.parse_args()

def setup_logger_and_out(config):
    """Initialize logger and output directories."""
    output_dir, model_dir = make_dirs(config)

    test_result_dir = os.path.join(output_dir, "test_results")
    os.makedirs(test_result_dir, exist_ok=True)

    logger = CustomLogger(__name__, log_directory=output_dir, log_file=f'log')
    logger.log_message(f"Configuration loaded: {config}")
    logger.log_message(f"Output will save at: {output_dir}")

    return logger, output_dir, model_dir, test_result_dir

def get_wandb_params(config, fold):
    """Initialize and log to wandb if enabled."""
    name = make_folder_name(config.split_config, is_path=False)
    project_name =  f"{get_model_name(config.model)}_{name}_{config.split_config.partition_mode}"
    run_name = f'Fold-{fold}_{serialize_dict(config.training.model_params)}'
    return project_name, run_name

def update_config_for_grid_search(config, train_batch_size, learning_rate, weight_decay, freeze_layer):
    """Update model parameters based on the current grid search combination."""
    config.training.model_params.train_batch_size = train_batch_size
    config.training.model_params.learning_rate = learning_rate
    config.training.model_params.weight_decay = weight_decay
    config.training.model_params.freeze_layers = freeze_layer

if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.train_config_file, args.split_config_file)

    # make sure eval_batch_size is defined; otherwise use train_batch_size
    if not getattr(config, 'eval_batch_size', None):
        config.eval_batch_size = list(config.training.model_params.train_batch_size)[0]

    logger, output_dir, model_dir, test_result_dir = setup_logger_and_out(config)
    config.device = set_device(config.device)
    logger.log_message("Using device:", config.device)

    torch.manual_seed(config.split_config.random_state)
    torch.cuda.manual_seed_all(config.split_config.random_state)

    dfs_train, dfs_val, df_test = DataSplit.get_splits(logger, config.split_config, label='label')

    model, tokenizer = init_model_and_tokenizer(logger, config.model, config.device)
    base_sd = model.state_dict()

    ds_test = get_ds(config.model, tokenizer, df_test)

    tt = Train_Test(logger, config.model.max_length, config.device, config.eval_batch_size, training_params=config.training)

    model_param_values = list(vars(config.training.model_params).values())
    grid_combinations = list(product(*model_param_values))
    logger.log_message(f'There are {len(grid_combinations)} combination of parameters...')

    results = []
    # train based on each combination
    for freeze_layer, train_batch_size, learning_rate, weight_decay in grid_combinations:
        logger.log_message("\n********************************************************************")
        logger.log_message(f"Training with batch_size={train_batch_size}, learning_rate={learning_rate},"
                    f" weight_decay={weight_decay}, freeze_layer={freeze_layer}")

        # Update config for this combination
        update_config_for_grid_search(config, train_batch_size, learning_rate, weight_decay, freeze_layer)

        param_model_dir = os.path.join(model_dir, serialize_dict(config.training.model_params))
        os.makedirs(param_model_dir, exist_ok=True)
        param_test_result_dir = os.path.join(test_result_dir, serialize_dict(config.training.model_params))
        os.makedirs(param_test_result_dir, exist_ok=True)

        # loop over each fold
        for fold in range(1, config.split_config.fold + 1):
            logger.log_message(f'**** Fold {fold}')

            # reset to initial model
            model.load_state_dict(base_sd)
            tt.model = model

            ds_train = get_ds(config.model, tokenizer, dfs_train.get(fold))
            ds_val = get_ds(config.model, tokenizer, dfs_val.get(fold))

            if config.wandb.enabled:  # visualization with wandb
                project_name, run_name = get_wandb_params(config, fold)
                init_wandb(logger, config.wandb, model, project_name, run_name)

            cur_model_name = f'Fold-{fold}'
            (trainable_params, best_epoch, last_val_acc, last_val_auroc,
             last_val_auprc, last_val_f1, last_val_mcc, train_time)\
                = tt.train(ds_train, ds_val, cur_model_name, param_model_dir, wandb)

            logger.log_message("Getting test set accuracy...")
            test_accuracy, test_auroc, test_auprc, test_f1, test_mcc, test_time \
                = tt.test(ds_test, cur_model_name, param_test_result_dir)

            # Aapend results for this fold
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