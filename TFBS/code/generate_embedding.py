import torch
from torch.nn.functional import softmax
import torch.nn.functional as F
from Bio import SeqIO
import numpy as np
import pandas as pd
import os
import sys
import yaml
from types import SimpleNamespace

from standard_fine_tune import FineTune
from embedding import Embedding
from dna_dataset import DNADataset
from data_split import DataSplit
from hyena_dna import HyenaDNAModel

hyena_dna_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../hyena-dna"))
sys.path.insert(0, hyena_dna_dir)

from huggingface import HyenaDNAPreTrainedModel
from standalone_hyenadna import CharacterTokenizer


# Recursively converts a dictionary into a SimpleNamespace.
def dict_to_namespace(d):
    if isinstance(d, dict):
        namespace = SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return namespace
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join("../configs", "standard_config.yml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)


    # Override device if applicable
    device_config = config.get("device")
    device_type = device_config.get("type", "cpu")
    config["device"] = "cuda" if torch.cuda.is_available() and device_type == "cuda" else "cpu"
    print("Using device:", config["device"])

    config_namespace = dict_to_namespace(config)
    return config_namespace



if __name__ == "__main__":

    # Load the configuration
    config = load_config()
    print(f"Configuration loaded: {config}")

    df = pd.read_csv(config.paths.dataset_path)
    df['sequence'] = df['sequence'].str.upper()
    # randomly rearrange the rows of df (shuffle rows)
    random_state = 1972934
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    self.tokenizer = HyenaDNAModel.get_tokenizer(config.model.model_max_length)

    ds = DNADataset(df, tokenizer, config.model.model_max_length, config.model.use_padding)
    
    embd = Embedding(is_finetuned, embedding_dir, config.device)
    emdb.generate(ds)