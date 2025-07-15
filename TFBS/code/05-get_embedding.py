import torch
import torch.nn as nn
import pandas as pd
import os
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from logger import CustomLogger
from utils import load_config, make_dirs, get_models, directory_not_empty
from model_utils import load_model, get_model_head_attributes, init_model_and_tokenizer, get_ds, set_device

from data_split import DataSplit

# python 05-get_embedding.py --config_file [config_path]
# python 05-get_embedding.py --embed_config_file ../configs/embedding/HeynaDNA-config.yml
# python 05-get_embedding.py --embed_config_file ../configs/embedding/HeynaDNA-config.yml --split_config_file ../configs/data_split/cross-species-config.yml

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_config_file", type=str, required=False, help="Path to the data_split config file.")
    parser.add_argument("--embed_config_file", type=str, required=True, help="Path to the embedding config file.")
    return parser.parse_args()

def validate(args):
    # load the embed config
    embed_cfg = load_config(args.embed_config_file)

    use_saved = getattr(embed_cfg.model, "use_saved_model", None)

    # if NOT using a saved model, we need a dataset_dir in the embed config
    if not use_saved and not getattr(embed_cfg, "dataset_dir", None):
        raise ValueError(
            "ERROR: 'dataset_dir' must be specified in your embed config when not using a saved model."
        )

    # if using a saved model, the --split_config_file CLI arg becomes required
    if use_saved and args.split_config_file is None:
        raise ValueError(
            "ERROR: --split_config_file must be specified when 'use_saved_model' is true in the embed_config."
        )

    # 3) If using a saved model, we now load the train & split configs
    if use_saved:
        embed_cfg = load_config(args.embed_config_file, args.split_config_file)
    # 4) Otherwise, just return the embed config
    return embed_cfg


def setup_logger_and_out(config):
    """
    Returns logger and output directories after creating necessary directories.
    """
    if config.model.use_saved_model:
        # get saves_model_dir dynamically based on dataset_split_config
        _, model_data_dir = make_dirs(config)
        model_data_dir = os.path.join(model_data_dir, config.model.saved_model_name)
        log_name = "log"
    else:
        model_data_dir = Path(config.dataset_dir)
        log_name = f"log_{config.model.model_name}"

    output_dir = os.path.join(model_data_dir, "Embedding")
    os.makedirs(output_dir, exist_ok=True)
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

def get_embeddings(loader, model, config):
    """
    Collect embeddings for all batches in the loader.
    """
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

    return all_seqs, all_labels, emb_tensor, per_seq_embd

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

def make_split_embeddings_for_folds(splits, model, tokenizer, model_name, config, logger, save_dir, fold):
    """
    Extract embeddings for each fold and save them.
    """
    for split_name, split in splits.items():
        logger.log_message(f'********* Dataset: {split_name} *********')

        embedding_file = f"{split_name}-{model_name}"
        if embedding_file in os.listdir(save_dir):
            logger.log_message(f'Embedding for {embedding_file} already exists at {save_dir}!\n Skipping...')
            continue

        df = split if split_name == "test" else split.get(fold)

        logger.log_message(f"{split_name}(Fold-{fold}) with {len(df)} rows")

        ds = get_ds(config.model, tokenizer, df)
        loader = DataLoader(ds, batch_size=config.eval_batch_size, shuffle=True)

        logger.log_message(f"Getting embeddings...")
        all_seqs, all_labels, all_embeddings, per_seq_embd = get_embeddings(loader, model, config)
        out_dict = {
            "sequences": all_seqs,
            "embeddings": per_seq_embd,
            "labels": all_labels,
        }
        # Save
        out_file = f"{save_dir}/{embedding_file}"
        torch.save(out_dict, out_file)
        logger.log_message(f" Saved pooled (per_seq) embeddings to {out_file} with shape {per_seq_embd.shape}"
                           f"(initial shape: {all_embeddings.shape})")

def make_embeddings_for_pretrained_model(logger, csv_items, model, head_attr, tokenizer, output_dir, config):
    for csv_key, csv_path in csv_items:
        logger.log_message(f'****************************************************************\n'
                           f'\n--- Processing CSV {csv_key}')

        df = pd.read_csv(csv_path)
        logger.log_message(f" Read {len(df)} rows from {csv_path}")

        save_dir = os.path.join(output_dir, Path(csv_key).stem)
        os.makedirs(save_dir, exist_ok=True)

        if directory_not_empty(save_dir):
            raise ValueError(f'Embedding already exists at {save_dir}!')

        logger.log_message(f'Will save embeddings at {save_dir}')

        ds = get_ds(config.model, tokenizer, df)
        loader = DataLoader(ds, batch_size=config.eval_batch_size, shuffle=True)

        # Swap out head
        replace_heads_with_identity(model, head_attr)
        model.to(config.device).eval()

        logger.log_message(f"Getting embeddings for {config.model.model_name}...")
        all_seqs, all_labels, all_embeddings, per_seq_embd = get_embeddings(loader, model, config)
        out_dict = {
            "sequences": all_seqs,
            "embeddings": per_seq_embd,
            "labels": all_labels,
        }
        # Save
        name = config.model.model_name
        if getattr(config.model, 'model_version', None):
            name = name + '_' + config.model.model_version

        out_file = f"{save_dir}/{name}.pt"
        torch.save(out_dict, out_file)
        logger.log_message(f" Saved pooled (per_seq) embeddings to {out_file} with shape {per_seq_embd.shape}"
                           f"(initial shape: {all_embeddings.shape})")

if __name__ == "__main__":
    args = parse_arg()
    config = validate(args)

    logger, model_data_dir, output_dir = setup_logger_and_out(config)
    config.device = set_device(config.device)
    logger.log_message("Using device:", config.device)

    model, tokenizer = init_model_and_tokenizer(logger, config.model, config.device)
    head_attr = get_model_head_attributes(config.model.model_name)
    # store the initial state dict of the model
    base_sd = model.state_dict()

    if config.model.use_saved_model:
        model_files = get_models(model_data_dir)
        logger.log_message(f'Using saved model(s) in {model_data_dir} for prediction:\n{model_files}')
        logger.log_message(f'Will save embeddings at {output_dir}')

        dfs_train, dfs_val, df_test = DataSplit.get_splits(logger, config.split_config, label='label')
        splits = {'train': dfs_train, 'val': dfs_val, 'test': df_test}

        for fold in range(1, config.split_config.fold + 1):
            # reset the model to the initial state before loading the saved model.
            model.load_state_dict(base_sd, strict=False)
            logger.log_message(f' **************************{fold} **************************')
            model_name = f'Fold-{fold}.pt'

            if model_name not in model_files:
                raise ValueError(f"model_name {model_name} not in models files in 'save_model_dir'")

            logger.log_message(f"Loading checkpoint: {model_name}")
            sd = load_model(model_data_dir, model_name, config.device)
            model.load_state_dict(sd, strict=False)

            # Swap out head
            replace_heads_with_identity(model, head_attr)
            model.to(config.device).eval()

            make_split_embeddings_for_folds(splits, model, tokenizer, model_name, config, logger, output_dir, fold)

    else:  # pretrained model
        csv_items = list_datasets(Path(config.dataset_dir)).items()
        logger.log_message(f"{len(csv_items)} CSVs:\n{csv_items}")

        make_embeddings_for_pretrained_model(logger, csv_items, model, head_attr, tokenizer, output_dir, config)