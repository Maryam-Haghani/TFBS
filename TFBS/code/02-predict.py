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
from data_split import DataSplit

hyena_dna_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../hyena-dna"))
sys.path.insert(0, hyena_dna_dir)

from huggingface import HyenaDNAPreTrainedModel
from standalone_hyenadna import CharacterTokenizer

dataset_path = "../inputs/ata_training_shuffle_neg_stride_200.csv"
dataset_split_path = "../inputs/ata_training_shuffle_neg_stride_200"
split_type =  'random' # 'cross_id'
id = 'chromosomeId'
train_ids = [1, 3, 5]
val_ids = [2]
test_ids = [4]

checkpoint_path = '../models/checkpoints'
n_classes = 2

use_padding = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

is_finetuned = True
use_saved_model = False
model_dir = '../models'
saved_finetuned_model_name ='fine_tuned_hyena_hyenadna-tiny-1k-seqlen_cross_id_epoch_3.pt'
#  'fine_tuned_hyena_hyenadna-tiny-1k-seqlen_random_0.7_0.15_0.15_epoch_6.pt'
pretrained_model_name = 'hyenadna-tiny-1k-seqlen'
# 'hyenadna-tiny-1k-seqlen'
# 'hyenadna-tiny-1k-seqlen-d256'
# 'hyenadna-tiny-16k-seqlen-d128''
# 'hyenadna-small-32k-seqlen'
# 'hyenadna-medium-160k-seqlen'
# 'hyenadna-medium-450k-seqlen'
# 'hyenadna-large-1m-seqlen'


model_max_length = 350
freeze_layers = []
# 'backbone.embeddings': Freezes the embedding layer.
# 'backbone.layers'    : Freezes all transformer blocks.
# 'backbone.layers.[i]': Freezes a specific transformer block.
# 'backbone'           : Freezes the entire backbone (embeddings + all transformer layers).
# 'head'               : Freezes the output classification head.
model_params = {
    'num_epochs': 60,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 0.1
}

generate_embedding = False
embedding_dir = '../outputs/embeddings/'

test_result_path = f'../outputs/test_results_{split_type}.csv'

loss_dir = f'../outputs/loss/{split_type}'
os.makedirs(loss_dir, exist_ok=True)


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

print(f"Getting pretrained model '{pretrained_model_name}'...")
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

if is_finetuned:
    data_split = DataSplit(df, dataset_split_path, split_type)
    df_train, df_val, df_test = data_split.split(id, train_ids, val_ids, test_ids)

    ds_train = DNADataset(df_train, tokenizer, model_max_length, use_padding)
    ds_val = DNADataset(df_val, tokenizer, model_max_length, use_padding)
    ds_test = DNADataset(df_test, tokenizer, model_max_length, use_padding)
    
    ft = FineTune(model, tokenizer, model_max_length, use_padding, device, model_dir, model_params, freeze_layers)

    if use_saved_model:  # saved_finetuned_model_name should be present
        model = ft.load(saved_finetuned_model_name)
    else:
        finetuned_model_name = f'fine_tuned_hyena_{pretrained_model_name}_{split_type}'
        model = ft.finetune(ds_train, ds_val, loss_dir, finetuned_model_name)

    ft.test(ds_test, test_result_path)

model.to(device)

if generate_embedding:
    print(f"Generating embedding...")
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