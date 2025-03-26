import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    generate_embedding = False
    embedding_dir = '../outputs/embeddings/'

    if generate_embedding:
        print(f"Generating embedding...")
        embd = Embedding(model, model_max_length, use_padding, is_finetuned, embedding_dir, device)

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