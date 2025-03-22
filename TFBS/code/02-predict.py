import torch
from torch.nn.functional import softmax
import torch.nn.functional as F
from Bio import SeqIO
import numpy as np
import pandas as pd
import os
import sys
from standard_fine_tune import FineTune
from embedding import Embedding
from dna_dataset import DNADataset

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, parent_dir)

from huggingface import HyenaDNAPreTrainedModel
from standalone_hyenadna import CharacterTokenizer



dataset_path = "../inputs/ata_training_shuffle_neg_stride_200.csv"
dataset_split_path = "../inputs/ata_training_shuffle_neg_stride_200"
pretrained_model_name = 'hyenadna-tiny-1k-seqlen'
checkpoint_path = '../models/checkpoints'
model_max_length = 350
n_classes = 2

use_padding = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# finetuned_model_path = None
is_finetuned = True
finetuned_model_path = '../models/fine_tuned_hyena_model.pth'
use_saved_model = False
model_params = {
    'num_epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 0.1
}

generate_embedding = False
embedding_dir = '../outputs/embeddings/'

loss_dir = '../outputs/loss/'
os.makedirs(loss_dir, exist_ok=True)

freeze_layers = []
# 'backbone.embeddings': Freezes the embedding layer.
# 'backbone.layers'    : Freezes all transformer blocks.
# 'backbone.layers.[i]': Freezes a specific transformer block.
# 'backbone'           : Freezes the entire backbone (embeddings + all transformer layers).
# 'head'               : Freezes the output classification head.



def split_dataset(ds, dataset_split_path):
    if os.path.exists(dataset_split_path):
        ds_train = torch.load(os.path.join(dataset_split_path, 'train_dataset.pt'))
        ds_val = torch.load(os.path.join(dataset_split_path, 'val_dataset.pt'))
        ds_test = torch.load(os.path.join(dataset_split_path, 'test_dataset.pt'))

    else:
        # Split dataset into training and validation
        train_val_size = int(0.8 * len(ds))
        print(f"train_val size: {train_val_size}")

        test_size = len(ds) - train_val_size

        ds_train_val, ds_test = torch.utils.data.random_split(ds, [train_val_size, test_size])

        train_size = int(0.8 * len(ds_train_val))
        val_size = len(ds_train_val) - train_size

        ds_train, ds_val = torch.utils.data.random_split(ds_train_val, [train_size, val_size])

        # Save all datasets
        os.makedirs(dataset_split_path, exist_ok=True)

        torch.save(ds_train, os.path.join(dataset_split_path, 'train_dataset.pt'))
        torch.save(ds_val, os.path.join(dataset_split_path, 'val_dataset.pt'))
        torch.save(ds_test, os.path.join(dataset_split_path, 'test_dataset.pt'))
        print(f"Datasets saved at {dataset_split_path}")

    print(f"train size: {len(ds_train)}")
    print(f"val size: {len(ds_val)}")
    print(f"test size: {len(ds_test)}")

    return ds_train, ds_val, ds_test


# predict binding site across entire sequence
def predict_binding_sites(sequence, model, tokenizer, window_size=200, stride=50):
    model.eval()
    results = []

    with torch.no_grad():
        for i in range(0, len(sequence) - window_size + 1, stride):
            window_seq = sequence[i:i + window_size]

            inputs = tokenizer(
                window_seq,
                add_special_tokens=False,
                padding="max_length",
                max_length=window_size,
                truncation=True,
            )
            input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0).to(device)

            # Get model predictions
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # normalizes the logits so that they lie between 0 and 1 and sum up to 1 across the class dimension
            probs = softmax(logits, dim=-1)

            # get probability of the positive class (class 1) for the first sequence in the batch
            prob_positive = probs[0, 1].item()

            results.append((i, i + window_size, prob_positive))

    return results






df = pd.read_csv(dataset_path)
df['sequence'] = df['sequence'].str.upper()
# randomly rearrange the rows of df (shuffle rows)
random_state = 1972934
df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

print(f"*** Getting pretrained model '{pretrained_model_name}'")
# Load the pre-trained HyenaDNA model
model = HyenaDNAPreTrainedModel.from_pretrained(
    checkpoint_path,
    pretrained_model_name,
    download=False,
    device=device,
    use_head=is_finetuned,  # Set to False to output embeddings, not classification
    n_classes=n_classes
)

# Create the tokenizer
tokenizer = CharacterTokenizer(
    characters=['A', 'C', 'G', 'T', 'N'],
    model_max_length=model_max_length,
    add_special_tokens=False,
    padding_side='left',
)

ds = DNADataset(df, tokenizer, model_max_length, use_padding)

if is_finetuned:
    ds_train, ds_val, ds_test = split_dataset(ds, dataset_split_path)

    if use_saved_model:  # finetuned_model_path should be present
        print(f"*** Loading finetuned model '{finetuned_model_path}'")
        model.load_state_dict(torch.load(finetuned_model_path))

    else:
        print(f"*** Performing finetuning on '{dataset_path}'")

        ft = FineTune(model, tokenizer, model_max_length, use_padding, device, loss_dir, model_params, freeze_layers)
        model = ft.finetune(ds_train, ds_val)

        torch.save(model.state_dict(), finetuned_model_path)
        print(f"Finetuned model saved to {finetuned_model_path}")

model.to(device)

if generate_embedding:
    print(f"*** Generating embedding")
    embd = Embedding(model, tokenizer, model_max_length, use_padding, is_finetuned, embedding_dir, device)

    if is_finetuned:
        embd.generate(ds_train, "train")
        embd.plot_tsne()

        embd.generate(ds_val, "val")
        embd.plot_tsne()

        embd.generate(ds_test, "test")
        embd.plot_tsne()

    else:
        embd.generate(ds, "All")
        embd.plot_tsne()

# # Test across the entire sequence (both finetuned and not finetuned positions)
# record = next(SeqIO.parse('./extracted_sequence.fasta', 'fasta'))
# sequence = str(record.seq)
#
# results = predict_binding_sites(sequence, model, tokenizer) # test with different strides (10)
#
# # Extract start, end, and probability values
# positions = [start for start, end, _ in results]
# probabilities = [prob for _, _, prob in results]
#
# # Plotting the peaks
# plt.figure(figsize=(12, 6))
# plt.plot(positions, probabilities, color='blue', linewidth=2, label='TF Binding Probability')
# # Adding labels and title
# plt.title('TF Binding Probability Across Sequence')
# plt.xlabel('Sequence Position')
# plt.ylabel('Probability')
# plt.grid(True)
# plt.legend()
# plt.show()