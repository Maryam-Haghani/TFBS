import torch
import torch.nn as nn
import os
import argparse
from torch.utils.data import DataLoader

from logger import CustomLogger
from utils import load_config, make_dirs, get_models, get_split_dirs
from model_utils import load_model, get_model_head_attributes, init_model_and_tokenizer, get_ds, set_device, get_model_name
from visualization import visualize_embeddings
from data_split import DataSplit

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_config_file", type=str, required=True, help="Path to the data_split config file.")
    parser.add_argument("--embed_config_file", type=str, required=True, help="Path to the embedding config file.")
    return parser.parse_args()


def setup_logger_and_out(config):
    """
    Returns logger and output directories after creating necessary directories.
    """
    model_dir = ""
    if config.model.use_pretrained_model:
        data_dir, _ = get_split_dirs(config.split_config)
        output_dir = os.path.join(data_dir, "embedding", get_model_name(config.model))
    else: # use_saved_model:
        # get saves_model_dir dynamically based on dataset_split_config
        output_dir, model_dir = make_dirs(config)
        output_dir = os.path.join(output_dir, 'embeddings')

    os.makedirs(output_dir, exist_ok=True)
    log_name = "log"
    logger = CustomLogger(__name__, log_directory=str(output_dir), log_file=log_name)
    logger.log_message(f"Configuration loaded: {config}")
    logger.log_message(f"Saving outputs to: {output_dir}")

    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    return logger, model_dir, output_dir, plot_dir

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

def make_embeddings(logger, df, model, tokenizer, config, embedding_path):
    """
    Collect embeddings for all batches in the loader.
    """
    logger.log_message(f"Getting embeddings...")

    ds = get_ds(config.model, tokenizer, df)
    loader = DataLoader(ds, batch_size=config.eval_batch_size, shuffle=True)

    all_embeddings, all_seqs, all_labels = [], [], []
    with torch.no_grad():
        for seqs, data, uid, peak_start, peak_end, labels in loader:
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
    torch.save(embedding_dict, embedding_path)
    logger.log_message(f" Saved pooled (per_seq) embeddings to {embedding_path} with shape {per_seq_embd.shape}"
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

def get_embeddings(df, split_name, model, tokenizer, config, logger, output_dir, model_name=None):
    """
    Get embeddings for model and save.
    """
    embedding_name = split_name if config.model.use_pretrained_model else f"{model_name}-{split_name}"
    logger.log_message(f"df:{split_name} with {len(df)} rows")

    embedding_path = os.path.join(output_dir, f"{embedding_name}.pt")

    if os.path.exists(embedding_path):
        # read
        split_embedding = load_embedding(logger, embedding_path)
    else:
        split_embedding = make_embeddings(logger, df, model, tokenizer, config, embedding_path)
    return split_embedding

if __name__ == "__main__":
    args = parse_arg()
    config = load_config(args.embed_config_file, args.split_config_file)

    logger, parent_model_dir, output_dir, plot_dir = setup_logger_and_out(config)

    input(output_dir)

    config.device = set_device(config.device)
    logger.log_message("Using device:", config.device)

    model, tokenizer = init_model_and_tokenizer(logger, config.model, config.device)
    head_attr = get_model_head_attributes(config.model.model_name)
    # store the initial state dict of the model
    base_sd = model.state_dict()

    # Swap out head in pretrained model
    replace_heads_with_identity(model, head_attr)
    model.to(config.device).eval()

    _, _, df_train_val, df_test = DataSplit.get_splits(logger, config.split_config, label='label')

    if config.model.use_pretrained_model:
        embedding_train = get_embeddings(df_train_val, "train",
                                          model, tokenizer, config, logger, output_dir)
        embedding_test = get_embeddings(df_test, "test",
                                         model, tokenizer, config, logger, output_dir)
        data = {'train': embedding_train, 'test': embedding_test}

        logger.log_message(f"Visualize embedding for pretrained model")
        visualize_embeddings(data, plot_dir, 'pretrtained')

    else: # use_saved_model
        model_dict = get_models(config.model.saved_model_dir)
        logger.log_message(f'Using saved model(s) in {parent_model_dir} for prediction:')

        for freeze_layer, files in model_dict.items():
            logger.log_message(f"******************************** Directory: {freeze_layer} ********************************")
            for model_name in files:
                model_dir = os.path.join(freeze_layer, model_name)
                logger.log_message(f"********* Model: {model_dir} **********")
                logger.log_message(f"Loading checkpoint: {model_name}")
                # reset the model to the initial state before loading the saved model.
                model.load_state_dict(base_sd, strict=False)
                sd = load_model(parent_model_dir, model_dir, config.device)
                model.load_state_dict(sd, strict=False)
                # Swap out head
                replace_heads_with_identity(model, head_attr)
                model.to(config.device).eval()

                model_name_without_format = model_name.split(".")[0]

                embedding_train = get_embeddings(df_train_val, "train", model, tokenizer,
                                                  config, logger, output_dir, model_name=model_name_without_format)
                embedding_test = get_embeddings(df_test, "test", model, tokenizer,
                                                 config, logger, output_dir, model_name=model_name_without_format)
                data = {'train': embedding_train, 'test': embedding_test}

                logger.log_message(f"Visualize embedding for model: {model_name}")
                visualize_embeddings(data, plot_dir, model_name_without_format)