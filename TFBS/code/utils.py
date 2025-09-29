from omegaconf import OmegaConf
from types import SimpleNamespace
import types
import math
import os
from pathlib import Path
import pytz
from datetime import datetime
import wandb

def serialize_array(array):
    # convert the array into a comma-separated string
    array_str = "_".join([str(item) for item in array])
    return array_str

def serialize_dict(model_params):
    """
    Serialize model parameters into a string to include in the saved model name.
    The string will be in a format like: "num_epochs_60_batch_size_32_lr_1e-4"
    """
    if isinstance(model_params, types.SimpleNamespace):
        model_params_dict = vars(model_params)
    else:
        model_params_dict = model_params

    param_str = "_".join([f"{key}_{value}" for key, value in model_params_dict.items()])
    return param_str

# Recursively converts a dictionary into a SimpleNamespace.
def dict_to_namespace(d):
    if isinstance(d, dict):
        namespace = SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return namespace
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

def load_config(config_path, split_config_path=None):
    # load train + data_split config files
    config = OmegaConf.load(config_path)

    if split_config_path:
        split_cfg = OmegaConf.load(split_config_path)
        # inject the split_config node
        config.split_config = split_cfg

    # turn the *entire* merged OmegaConf tree into plain Python
    config = OmegaConf.to_container(config, resolve=True)
    config_namespace = dict_to_namespace(config)
    return config_namespace

def extract_value_as_list(value):
    if isinstance(value, list):
        return value
    else:
        return [value]

def extract_single_value(value):
    if isinstance(value, list):
        if len(value) == 1:
            value = value[0]
        else:
            raise ValueError("list must contain exactly one element")
    elif not isinstance(value, (int, float)):
        raise TypeError("must be a number or a list with one element")
    return value

def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min, lr_max, warmup=True):
    # warmup_epoch: Number of model warm-ups
    warmup_epoch = 5 if warmup else 0
    # Model warm-up phase
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    # Formal training phase of the model
    elif current_epoch < max_epoch:
        lr = lr_min + (lr_max - lr_min) * (
                1 + math.cos(math.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    else:
        lr = lr_min + (lr_max - lr_min) * (
                1 + math.cos(math.pi * (current_epoch - max_epoch) / max_epoch)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# make folder based on data spilt config to be a subdirectory inside out_dir
def make_folder_name(split_config, is_path=True):
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

def make_dirs(config, make_model_dir=True):
    name = make_folder_name(config.split_config)

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

    if make_model_dir:
        model_dir = os.path.join(output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        return output_dir, model_dir
    else:
        return output_dir



def get_models(model_dir):
    p = Path(model_dir)

    # model_dir should already exist and contain some saved models in .pt format
    if not p.exists():
        raise FileNotFoundError(f"Directory does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {p}")

    models_dict = {}

    # Collect .pt files directly inside model_dir
    root_files = [f.name for f in p.glob("*.pt")]
    if root_files:
        models_dict["__root__"] = root_files

    # Collect .pt files from each subdirectory
    for sub in p.iterdir():
        if sub.is_dir():
            pt_files = [f.name for f in sub.glob("*.pt")]
            if pt_files:
                models_dict[sub.name] = pt_files

    if not models_dict:
        raise FileNotFoundError(f"No .pt files found in {p} or its subdirectories.")

    return models_dict

def directory_not_empty(directory):
    return any(Path(directory).iterdir())

def init_wandb(logger, wandb_params, model, project_name, run_name):
    try:
        wandb.login(key=wandb_params.token)

        eastern = pytz.timezone(wandb_params.timezone)
        wandb.init(project=f'{project_name}',
                   entity=wandb_params.entity_name,
                   name=f"{run_name}-{datetime.now(eastern).strftime(wandb_params.timezone_format)}")
        wandb.watch(model, log="all")
    except Exception as e:
        logger.log_message(f"Error initializing Wandb: {e}")
        raise

def get_split_dirs(config):
    split_path = os.path.join(config.split_dir, config.name)
    # determine the test split path
    if config.test_split_type == 'cross':
        split_path = os.path.join(split_path,
                                  f'test-{config.test_ids}-{config.partition_mode}')
    elif config.test_split_type == 'random':
        split_path = os.path.join(split_path,
                                  f'test-{str(config.test_size)}-{config.partition_mode}')
    else:
        raise ValueError(
            f"Invalid test_split_type: {config.test_split_type}")

    # determine the validation split path
    if config.val_split_type == 'cross':
        val_split_path = os.path.join(split_path,
                                      f'val_{config.val_split_type}-{config.id_column}',
                                      f'val-{config.val_ids}')
    elif config.val_split_type == 'n-fold':
        val_split_path = os.path.join(split_path, f'val_{config.fold}-fold')
    else:
        raise ValueError(
            f"Invalid val_split_type: {config.val_split_type}")
    return split_path, val_split_path