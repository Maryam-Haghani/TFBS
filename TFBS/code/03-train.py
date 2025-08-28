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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_file", type=str, required=True, help="Path to the train config file.")
    parser.add_argument("--split_config_file", type=str, required=True, help="Path to the data_split config file.")
    return parser.parse_args()

def setup_logger_and_out(config):
    """Initialize logger and output directories."""
    output_dir, model_dir = make_dirs(config)

    logger = CustomLogger(__name__, log_directory=output_dir, log_file=f'log')
    logger.log_message(f"Configuration loaded: {config}")
    logger.log_message(f"Output will save at: {output_dir}")

    return logger, output_dir, model_dir

def get_wandb_params(config, prependix):
    """Initialize and log to wandb if enabled."""
    name = make_folder_name(config.split_config, is_path=False)
    project_name =  f"{get_model_name(config.model)}_{name}_{config.split_config.partition_mode}"
    run_name = f'{prependix}_{serialize_dict(config.training.model_params)}'
    return project_name, run_name

def update_config_for_grid_search(config, train_batch_size, learning_rate, weight_decay, freeze_layer, num_epochs):
    """Update model parameters based on the current grid search combination."""
    config.training.model_params.train_batch_size = train_batch_size
    config.training.model_params.learning_rate = learning_rate
    config.training.model_params.weight_decay = weight_decay
    config.training.model_params.freeze_layers = freeze_layer
    config.training.num_epochs = num_epochs

def set_best_params(config, logger, df_param_results):
    logger.log_message(f"Getting best hyper params based on {param_results_csv_file}........")

    # get best params
    config_cols = [
        'train_batch_size',
        'eval_batch_size',
        'learning_rate',
        'weight_decay',
        'freeze_layer',
        'best_epoch',
    ]

    # take the average of 5 folds during cross-validation phase.
    best_config = (
        df_param_results
        .groupby(config_cols)['best_val_f1']
        .mean()
        .reset_index(name='avg_val_f1')
        .sort_values('avg_val_f1', ascending=False)
        .head(1)
    ).iloc[0]

    best_params = best_config[config_cols].to_dict()
    best_avg_f1 = best_config['avg_val_f1']

    logger.log_message(f"Best params: {best_params} with Avg val‑F1: {best_avg_f1}")

    # update config based on best params
    update_config_for_grid_search(config, best_params['train_batch_size'], best_params['learning_rate'],
                                  best_params['weight_decay'], best_params['freeze_layer'], best_params['best_epoch'])

    return best_params

if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.train_config_file, args.split_config_file)

    # make sure eval_batch_size is defined; otherwise use train_batch_size
    if not getattr(config, 'eval_batch_size', None):
        config.eval_batch_size = list(config.training.model_params.train_batch_size)[0]

    logger, output_dir, model_dir = setup_logger_and_out(config)
    config.device = set_device(config.device)
    logger.log_message("Using device:", config.device)

    torch.manual_seed(config.split_config.random_state)
    torch.cuda.manual_seed_all(config.split_config.random_state)

    dfs_train, dfs_val, df_train_val, df_test = DataSplit.get_splits(logger, config.split_config, label='label')

    model, tokenizer = init_model_and_tokenizer(logger, config.model, config.device)
    base_sd = model.state_dict()

    tt = Train_Test(logger, config.model.max_length, config.device, config.eval_batch_size, training_params=config.training)

    model_param_values = list(vars(config.training.model_params).values())
    grid_combinations = list(product(*model_param_values))

    if len(grid_combinations) != 1:
        logger.log_message(f'There are {len(grid_combinations)} combination of parameters...')

        param_results_csv_file = os.path.join(output_dir, 'hyper_params.csv')
        if not os.path.exists(param_results_csv_file):
            param_results = []
            logger.log_message("*******Running with all parameters available to get the best one: *******")
            # train based on each combination
            for freeze_layer, train_batch_size, learning_rate, weight_decay in grid_combinations:
                logger.log_message("\n********************************************************************")
                logger.log_message(f"Training with batch_size={train_batch_size}, learning_rate={learning_rate},"
                            f" weight_decay={weight_decay}, freeze_layer={freeze_layer}")

                # Update config for this combination
                update_config_for_grid_search(config, train_batch_size, learning_rate, weight_decay, freeze_layer, config.training.num_epochs)

                # loop over each fold to find best params
                for fold in range(1, config.split_config.fold + 1):
                    logger.log_message(f'**** Fold {fold}')

                    # reset to initial model
                    model.load_state_dict(base_sd)
                    tt.model = model

                    ds_train = get_ds(config.model, tokenizer, dfs_train.get(fold))
                    ds_val = get_ds(config.model, tokenizer, dfs_val.get(fold))

                    cur_model_name = f'Fold-{fold}'

                    if config.wandb.enabled:  # visualization with wandb
                        project_name, run_name = get_wandb_params(config, cur_model_name)
                        init_wandb(logger, config.wandb, model, project_name, run_name)

                    (trainable_params, best_epoch, best_val_acc, best_val_auroc,
                     best_val_auprc, best_val_f1, best_val_mcc, train_time)\
                        = tt.train(ds_train, cur_model_name, wandb, ds_val=ds_val)

                    # Aapend results for this fold
                    param_results.append({
                        'fold': fold,
                        'params': serialize_dict(config.training.model_params),
                        'train_batch_size': train_batch_size,
                        'eval_batch_size': config.eval_batch_size,
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'freeze_layer': freeze_layer,
                        'trainable_params': trainable_params,
                        'train_time(s)': train_time,
                        'best_epoch': best_epoch,
                        'best_val_acc': round(best_val_acc, 2),
                        'best_val_f1': round(best_val_f1, 2),
                        'best_val_mcc': round(best_val_mcc, 2),
                        'best_val_auroc': round(best_val_auroc, 2),
                        'best_val_auprc': round(best_val_auprc, 2)
                    })

                    if config.wandb.enabled:
                        wandb.finish()

            df_param_results = pd.DataFrame(param_results)
            df_param_results.to_csv(param_results_csv_file, index=False)
            logger.log_message(f"Grid search results saved to {param_results_csv_file}", use_time=True)

        else:
            df_param_results = pd.read_csv(param_results_csv_file)

        logger.log_message("**************************************************************************")
        logger.log_message("Retrain the model using best params on whole training data")
        logger.log_message("**************************************************************************")

        best_params = set_best_params(config, logger, df_param_results)
        logger.log_message(f"Best hyperparams: {best_params}")

    else:
        logger.log_message("Only one combination of parameters,"
                           " starting to train model on whole training data...")
        freeze_layer, train_batch_size, learning_rate, weight_decay = grid_combinations[0]
        update_config_for_grid_search(config, train_batch_size, learning_rate, weight_decay, freeze_layer,
                                      config.training.num_epochs)

    ds_whole_train = get_ds(config.model, tokenizer, df_train_val)
    ds_test = get_ds(config.model, tokenizer, df_test)

    # loop over each seed
    training_stats = []
    for seed in config.model.seeds:
        logger.log_message(f'*********** Seed: {seed}')
        # reset to initial model
        model.load_state_dict(base_sd)
        tt.model = model
        cur_model_name = f'Seed-{seed}'

        if config.wandb.enabled:  # visualization with wandb
            project_name, run_name = get_wandb_params(config, cur_model_name)
            init_wandb(logger, config.wandb, model, project_name, run_name)

        trainable_params, train_time = tt.train(ds_whole_train, cur_model_name, wandb, model_dir= model_dir)

        training_stats.append({
            'seed': seed,
            'trainable_params': trainable_params,
            'train_time(s)': train_time
        })

        if config.wandb.enabled:
            wandb.finish()

    # get statistics for training on whole train data
    df_training_stats = pd.DataFrame(training_stats)
    training_stats_csv_file = os.path.join(output_dir, 'training_stats.csv')
    df_training_stats.to_csv(training_stats_csv_file, index=False)
    logger.log_message(f"\nTraining stats based on different seeds saved to {training_stats_csv_file}", use_time=True)