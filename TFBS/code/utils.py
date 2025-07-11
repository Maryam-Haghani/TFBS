from omegaconf import OmegaConf
from types import SimpleNamespace
import types
import math
import os
import torch

def serialize_array(array):
    """
    Serialize an array into a string to include in the saved model name.
    The array is serialized as a comma-separated list of its elements.

    Args:
        array (list): The array (list) to serialize

    Returns:
        str: The serialized array as a string with its name.
    """
    # convert the array into a comma-separated string
    array_str = "_".join([str(item) for item in array])
    return array_str

def serialize_dict(model_params):
    """
    Serialize model parameters into a string to include in the saved model name.
    The string will be in a format like: "num_epochs_60_batch_size_32_lr_1e-4"

    Args:
        model_params (dict): Dictionary of model parameters (e.g., num_epochs, batch_size, etc.)

    Returns:
        str: Serialized string of model parameters
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


def load_model(model_dir, model_name, device):
    model_path = os.path.join(model_dir, model_name)
    state_dict = torch.load(model_path, map_location=torch.device(device))
    return state_dict
