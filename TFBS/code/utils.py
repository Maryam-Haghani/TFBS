import json
import os
import yaml
from types import SimpleNamespace
import types

def serialize_array(array):
    """
    Serialize an array into a string to include in the saved model name.
    The array is serialized as a comma-separated list of its elements.

    Args:
        array (list): The array (list) to serialize

    Returns:
        str: The serialized array as a string with its name.
    """
    # Convert the array into a comma-separated string
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

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
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


