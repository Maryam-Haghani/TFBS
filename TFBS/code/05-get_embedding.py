import torch
import torch.nn as nn
import pandas as pd
import os
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from logger import CustomLogger
from utils import load_config, make_dirs, get_models, get_split_dirs
from model_utils import load_model, get_model_head_attributes, init_model_and_tokenizer, get_ds, set_device, get_model_name
from visualization import visualize_embeddings
from data_split import DataSplit

# python 05-get_embedding.py --config_file [config_path]
# python 05-get_embedding.py --embed_config_file ../configs/embedding/HeynaDNA-config.yml --split_config_file ../configs/data_split/cross-species-config.yml

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_config_file", type=str, required=True, help="Path to the data_split config file.")
    parser.add_argument("--embed_config_file", type=str, required=True, help="Path to the embedding config file.")
    return parser.parse_args()


def setup_logger_and_out(config):
    """
    Returns logger and output directories after creating necessary directories.
    """
    if config.model.use_pretrained_model:
        model_data_dir, _ = get_split_dirs(config.split_config)
        output_dir = os.path.join(model_data_dir, "Embedding", get_model_name(config.model))
    else: # use_saved_model:
        # get saves_model_dir dynamically based on dataset_split_config
        _, model_data_dir = make_dirs(config)
        model_data_dir = os.path.join(model_data_dir, config.model.saved_model_name)
        output_dir = os.path.join(model_data_dir, "Embedding")

    os.makedirs(output_dir, exist_ok=True)
    log_name = "log"
    logger = CustomLogger(__name__, log_directory=str(output_dir), log_file=log_name)
    logger.log_message(f"Configuration loaded: {config}")
    logger.log_message(f"Saving outputs to: {output_dir}")
    return logger, model_data_dir, output_dir

# list datasets inside dataset_dir
def list_datasets(dataset_dir):
    """
    Return a map from underscored-relative-path → full Path for every .csv under root.
    """
    return {
        "_".join(p.relative_to(dataset_dir).parts): p
        for p in dataset_dir.rglob("*.csv")
    }

def replace_heads_with_identity(model, head_attr):
    """
    Replace the heads of the model with nn.Identity() for inference.
    This is useful when we don't want the model's heads to affect the output, like during embedding extraction.
    """
    for head in head_attr:
        if not hasattr(model, head):
            raise AttributeError(f"Model has no attribute '{head}'")
        setattr(model, head, nn.Identity())  # Replace head with nn.Identity() for inference

def load_embedding(logger, output_dir):
    logger.log_message(f'Embedding already exists at {output_dir}!\n Reading file...')
    split_embedding = torch.load(output_dir, map_location=torch.device('cpu'))
    logger.log_message(
        f"Read pooled (per_seq) embeddings in {output_dir} with shape {split_embedding['embeddings'].shape}")
    return split_embedding

def get_embeddings(logger, df, model, tokenizer, config, output_dir):
    """
    Collect embeddings for all batches in the loader.
    """
    logger.log_message(f"Getting embeddings...")

    ds = get_ds(config.model, tokenizer, df)
    loader = DataLoader(ds, batch_size=config.eval_batch_size, shuffle=True)

    all_embeddings, all_seqs, all_labels = [], [], []
    with torch.no_grad():
        for seqs, data, labels in loader:
            data = data.to(config.device)
            out = model(data)
            if config.model.model_name == "AgroNT":
                out = out.logits
            all_embeddings.append(out.cpu())
            all_seqs.extend(seqs)
            all_labels.extend(labels)

    emb_tensor = torch.cat(all_embeddings, dim=0)
    per_seq_embd = get_per_sequence_embedding(config.model.pooling_method, emb_tensor)

    embedding_dict = {
        "sequences": all_seqs,
        "embeddings": per_seq_embd,
        "labels": all_labels,
    }

    # Save
    torch.save(embedding_dict, output_dir)
    logger.log_message(f" Saved pooled (per_seq) embeddings to {output_dir} with shape {per_seq_embd.shape}"
                       f"(initial shape: {emb_tensor.shape})")

    return embedding_dict

def get_per_sequence_embedding(pooling_method, emb_tensor):
    """
        Get pooled sequence embeddings based on the pooling method.
    """
    pooling_methods = {
        "first_token": emb_tensor[:, 0, :],
        "last_token": emb_tensor[:, -1, :],
        "mean": emb_tensor.mean(dim=1),
    }
    if pooling_method not in pooling_methods:
        raise ValueError(f"Unknown pooling_method: {pooling_method}")
    return pooling_methods[pooling_method]

def make_split_embeddings_for_fold(fold, splits, model, tokenizer, config, logger, output_dir):
    """
    Extract embeddings for fold and save.
    """
    embedding_name = f"Fold-{fold}.pt"
    split_embeddings = {}
    for split_name, split in splits.items():
        logger.log_message(f'********* Dataset: {split_name} *********')

        if split_name == "test":
            df = split
            logger.log_message(f"df:{split_name} with {len(df)} rows")
            if config.model.use_pretrained_model:
                embedding_name = f"test.pt"
        else:
            df = split.get(fold)
            logger.log_message(f"df:{split_name}(Fold:{fold}) with {len(df)} rows")

        save_dir = os.path.join(output_dir, f"split-{split_name}")
        os.makedirs(save_dir, exist_ok=True)
        embedding_path = f"{save_dir}/{embedding_name}"

        if os.path.exists(embedding_path): # this will also check that embedding for test set once IF using pretrained model
            # Read
            split_embedding = load_embedding(logger, embedding_path)

        else:
            split_embedding = get_embeddings(logger, df, model, tokenizer, config, embedding_path)

        split_embeddings[split_name] = split_embedding
    return split_embeddings

if __name__ == "__main__":
    args = parse_arg()
    config = load_config(args.embed_config_file, args.split_config_file)

    logger, model_data_dir, output_dir = setup_logger_and_out(config)
    config.device = set_device(config.device)
    logger.log_message("Using device:", config.device)

    model, tokenizer = init_model_and_tokenizer(logger, config.model, config.device)
    head_attr = get_model_head_attributes(config.model.model_name)
    # store the initial state dict of the model
    base_sd = model.state_dict()

    # Swap out head in pretrained model
    replace_heads_with_identity(model, head_attr)
    model.to(config.device).eval()

    if not config.model.use_pretrained_model: # saved_model
        model_files = get_models(model_data_dir)
        logger.log_message(f'Using saved model(s) in {model_data_dir} for prediction:\n{model_files}')

    logger.log_message(f'Will save embeddings at {output_dir}')

    dfs_train, dfs_val, df_test = DataSplit.get_splits(logger, config.split_config, label='label')
    splits = {'train': dfs_train, 'val': dfs_val, 'test': df_test}

    for fold in range(1, config.split_config.fold + 1):
        logger.log_message(f' **************************Fold {fold} **************************')
        if not config.model.use_pretrained_model: #use_saved_model
            # reset the model to the initial state before loading the saved model.
            model.load_state_dict(base_sd, strict=False)

            model_name = f'Fold-{fold}.pt'

            if model_name not in model_files:
                raise ValueError(f"model_name {model_name} not in models files in 'save_model_dir'")

            logger.log_message(f"Loading checkpoint: {model_name}")
            sd = load_model(model_data_dir, model_name, config.device)
            model.load_state_dict(sd, strict=False)

            # Swap out head
            replace_heads_with_identity(model, head_attr)
            model.to(config.device).eval()

        fold_split_embeddings = make_split_embeddings_for_fold(
            fold, splits, model, tokenizer, config, logger, output_dir)

        logger.log_message(f"Visualize embedding for Fold: {fold}")
        plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        visualize_embeddings(fold_split_embeddings, plot_dir, f"Fold-{fold}")