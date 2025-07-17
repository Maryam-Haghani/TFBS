import os
import torch

from datasets.deepbind_dataset import DeepBindDataset
from datasets.foundation_dataset import FoundationDataset

from models.hyena_dna import HyenaDNAModel
from models.dna_bert_2 import DNABERT2
from models.deep_bind import DeepBind
from models.BERT_TFBS.bert_tfbs import BERT_TFBS
from models.agro_nt import AgroNTModel

def set_device(device):
    """Set the computation device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"

def load_model(model_dir, model_name, device):
    model_path = os.path.join(model_dir, model_name)
    state_dict = torch.load(model_path, map_location=torch.device(device))
    return state_dict

def get_model_head_attributes(model_name):
    """
    Return the correct model head attributes based on the model name.
    """
    head_map = {
        "HyenaDNA": ["head"],
        "DNABERT-2": ["pool", "classification_head"],
        "AgroNT": ["classifier"]
    }
    return head_map.get(model_name, [])

def get_available_models():
    return {
        "DeepBind": DeepBind,
        "BERT-TFBS": BERT_TFBS,
        "HyenaDNA": HyenaDNAModel,
        "DNABERT-2": DNABERT2,
        "AgroNT": AgroNTModel
    }

def get_foundation_models():
    return {
        "HyenaDNA": HyenaDNAModel,
        "DNABERT-2": DNABERT2,
        "AgroNT": AgroNTModel
    }

def get_model_class(model_name):
    model_map = get_available_models()

    model_class = model_map.get(model_name)
    if not model_class:
        raise ValueError(f"Given model name ({model_name}) is not valid!")
    return model_class

def init_model_and_tokenizer(logger, config, device):
    """
        Initialize the model and tokenizer based on the specified model name.
    """
    model_class = get_model_class(config.model_name)

    if config.model_name == "HyenaDNA":
        model_instance = model_class(logger, pretrained_model_name=config.model_version, device=device)
        model = model_instance.load_pretrained_model()
        tokenizer = model_instance.get_tokenizer(config.max_length)
    elif config.model_name == 'DeepBind':
        model_instance = model_class(config.kernel_length)
        model = model_instance
        tokenizer = None
    elif config.model_name == 'BERT-TFBS':
        model_instance = model_class(config.max_length)
        tokenizer = model_instance.get_tokenizer()
        model = model_instance
    elif config.model_name == 'DNABERT-2':
        model_instance = model_class()
        model = model_instance
        tokenizer = model_instance.get_tokenizer()
    elif config.model_name == "AgroNT":
        model_instance = model_class(logger, device=device)
        model = model_instance.load_pretrained_model(config.finetune_type)
        tokenizer = model_instance.get_tokenizer()
    else:
        raise ValueError(f'Given model name ({config.model_name}) is not valid!')
    return model, tokenizer

def get_ds(config, tokenizer, data, mode="df", window_size=None, stride=None):
    """
        Get test dataset based on the specified model name.
    """

    if config.model_name not in get_available_models():
        raise ValueError(f'Given model name ({config.model_name}) is not valid!')
    elif config.model_name == 'DeepBind':
        ds = DeepBindDataset(data, config.max_length, config.kernel_length)
    else:
        ds = FoundationDataset(mode, config.model_name, tokenizer, data, config.max_length,
                               window_size=window_size, stride=stride)

        if config.model_name == 'HyenaDNA':
            ds = FoundationDataset(mode, config.model_name, tokenizer, data, config.max_length,
                                   window_size=window_size, stride=stride, use_padding=config.use_padding)
    return ds

def get_model_name(config):
    name = config.model_name
    if getattr(config, 'model_version', None):
        name = name + '_' + config.model_version

    return name