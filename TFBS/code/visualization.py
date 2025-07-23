import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import torch
from pathlib import Path

# def plot_loss(param_combinations, train_loss_per_param, val_loss_per_param, plot_dir, plot_name):
#     plt.figure(figsize=(8, 6))
#     for i, param in enumerate(param_combinations):
#         plt.plot(train_loss_per_param[i], label=f'Train ({param})', linestyle='-', linewidth=2)
#         plt.plot(val_loss_per_param[i], label=f'Validation ({param})', linestyle='--', linewidth=2)
#
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title(f'Training and Validation Losses for Different Hyperparameter Combinations')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), fontsize='small')
#
#     plot_path = os.path.join(plot_dir, plot_name)
#     plt.savefig(plot_path)
#     print(f"Saved loss plots as PDF files at {plot_path}.")

# def plot_auroc_auprc(type, param_combinations, values, plot_dir, name):
#     plt.figure(figsize=(8, 6))
#     for i, param in enumerate(param_combinations):
#         plt.plot(values[i], label=f'{type} ({param})', linestyle=':')
#     plt.xlabel('Epochs')
#     plt.ylabel(type)
#     plt.title(f'{type} for Different Parameter Combinations')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), fontsize='small')
#     plt.grid(True)
#     plot_path = os.path.join(plot_dir, f'{type}_{name}.pdf')
#     plt.savefig(plot_path)
#     print(f"Saved loss plots as PDF files at {plot_path}.")

def plot_roc_pr(type, true_labels, probs, x_name, y_name, plot_dir, name):
    plt.clf()
    if type == 'ROC':
        fpr, tpr, thresholds = roc_curve(true_labels, probs)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUROC=0.5)')
        x, y = fpr, tpr

        # optimal threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)

    elif type == 'PR':
        precision, recall, thresholds = precision_recall_curve(true_labels, probs)
        x, y = recall, precision
        # Add baseline
        pos_proportion = np.mean(true_labels)
        plt.axhline(y=pos_proportion, color='gray', linestyle='--', label=f'Random (AUPR={pos_proportion:.2f})')

        # optimal threshold (closest to (1,1))
        dist_to_corner = (1 - recall) ** 2 + (1 - precision) ** 2
        optimal_idx = np.argmin(dist_to_corner)
    else:
        raise ValueError("Type must be 'ROC' or 'PR'")

    au = auc(x, y)

    optimal_threshold = thresholds[optimal_idx]
    plt.scatter(x[optimal_idx], y[optimal_idx], marker='o', color='red',
                label=f'Optimal Threshold = {optimal_threshold:.2f}')

    plt.plot(x, y, color='b', lw=2, label=f'AU = {au:.2f}')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f'{type} Curve')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f'{type}-{name}.pdf')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved {type} as PDF file at {plot_path}.")


# visualizes the embeddings from the different splits (train, val, test)
# using PCA, T-SNE, and UMAP
def visualize_embeddings(fold_split_embeddings, output_dir, fold_name):
    """
    fold_split_embeddings : {train, train_embed, val: val_embed: test: test_embed}
    each embed is a dict of keys: "sequences", "embeddings", "labels"
      train_embed embeddings of shape (N_train,  emb_size)
      val_embed embeddings of shape (N_valid,  emb_size)
      test_embed embeddings of shape (N_test,   emb_size)

    do dimension reduction for each embedding based on 1."PCA" 2."T-SNE" 3."UMAP"
    shape code each label
    color code each split
    """
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)
    umap_model = umap.UMAP(n_components=2)

    # define different colors for train, validation, and test
    colors = {
        'train': '#AEC6CF',  # Pastel Blue
        'val': '#FFD1DC',  # Pastel Pink
        'test': '#FFB347',  # Pastel Orange
    }
    # define different markers for labels 0 and 1
    markers = {0: 'o', 1: '^'}

    # loop through each dimensionality reduction method
    for method, model in zip(["PCA", "T-SNE", "UMAP"], [pca, tsne, umap_model]):
        plt.figure(figsize=(8, 6))

        # loop through each split
        for split, embed_data in fold_split_embeddings.items():
            embeddings = embed_data['embeddings']
            labels = embed_data['labels']

            # ensure embeddings are on CPU and convert to numpy array
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()


            if isinstance(labels, list):
                # convert labels to numpy array if they are a list
                try:
                    labels = np.concatenate([np.array(label) for label in labels], axis=0)
                except ValueError:
                    raise ValueError(f"Error: Inconsistent label shapes in {split}.")
            elif isinstance(labels, torch.Tensor):
                # move tensor to CPU and convert to numpy array
                labels = labels.cpu().numpy()

            # apply dimensionality reduction
            reduced_embeddings = model.fit_transform(embeddings)

            # Scatter plot
            for label in np.unique(labels):  # loop over unique labels (0 and 1)
                label_indices = np.where(labels == label)[0]
                plt.scatter(
                    reduced_embeddings[label_indices, 0], reduced_embeddings[label_indices, 1],
                    c=[colors[split]] * len(label_indices),  # Use color for the split
                    marker=markers[label],  # Use different marker for label 0 and 1
                    s=5
                )

        plt.title(f'{method} Embedding Visualization for {fold_name}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')

        # separate legend for color (split)
        for split, color in colors.items():
            plt.scatter([], [], color=color, label=f'Split: {split}')

        # separate legend for shapes (labels)
        for label, marker in markers.items():
            plt.scatter([], [], color='black', marker=marker, label=f'Label: {label}')

        plt.legend()

        # save the plot
        output_path = os.path.join(output_dir, method)
        os.makedirs(output_path, exist_ok=True)
        output_path = f"{output_path}/{fold_name}-embedding.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

def plot_peaks(predictions, output_dir, name):
    plt.figure(figsize=(12, 6))

    # plot each sliding window
    for start, end, prob in predictions:
        plt.plot([start, end], [prob, prob], color='blue', linewidth=1)  # Line with constant probability

    # plot smooth line based on start positions
    start_positions, end_positions, probabilities = zip(*predictions)
    plt.plot(start_positions, probabilities, color='gray', linewidth=1,
             linestyle='dashed', label='Smooth Probability Curve based on start positions')

    # Pplot the horizontal line Y = 0.5
    plt.axhline(y=0.5, color='red', linestyle='--', label='Random Predictor')

    plt.title(f'TF Binding Probability Across Sequence for {name}')
    plt.xlabel('Sequence Position')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.legend()

    # save the plot
    output_path = f"{output_dir}/peak-{name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def adjust_lightness(color, amount: float = 1.5):
    """
    Adjust the lightness of a matplotlib color.

    :param color: A matplotlib color string or RGB tuple.
    :param amount: Factor by which to multiply the lightness component.
                   >1 → lighter, 0<amount<1 → darker.
    :return: New RGB tuple with adjusted lightness.
    """
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * amount))
    return colorsys.hls_to_rgb(h, l, s)

def plot_single_saliency(seq_scores, sequence, uid, label, peak_range, save_path, fig_size= (20, 3)):
    """
    Plot and save a saliency map for one sample.
    """
    start, end = peak_range
    seq_len = seq_scores.shape[0]

    base_color = 'darkgreen' if label == 1 else 'maroon'
    light_color = adjust_lightness(base_color, amount=1.5)
    boundary_color = adjust_lightness(base_color, amount=2.5)

    bar_colors = [
        light_color if start <= idx <= end else base_color
        for idx in range(seq_len)
    ]

    fig, ax = plt.subplots(figsize=fig_size)
    ax.bar(np.arange(seq_len), seq_scores, color=bar_colors, width=1.0)

    # boundary lines around the peak
    ax.axvline(start - 0.5, color=boundary_color, linestyle='--', linewidth=2)
    ax.axvline(end + 0.5, color=boundary_color, linestyle='--', linewidth=2)

    ax.set_title(
        f"ID: {uid} | Peak: {start}-{end} | Label: {label} | SeqLen: {seq_len}",
        fontsize=12
    )
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Attribution Score", fontsize=12)

    ax.set_xticks(np.arange(seq_len))
    ax.set_xticklabels(list(sequence), fontsize=10)
    for idx, tick in enumerate(ax.get_xticklabels()):
        tick.set_fontweight('bold' if start <= idx <= end else 'normal')

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_saliency_maps_from_file(logger, npz_file_path, output_dir):
    """
    Load data from an .npz and produce one saliency plot per sample.
    """
    npz_path = Path(npz_file_path)

    if not npz_path.is_file():
        logger.log_message(f"Error: File not found – {npz_path}")
        return

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            attributions = data['attributions']
            sequences = data['sequences']
            labels = data['labels']
            uids = data['uids']
            peak_start_ends = data['peak_start_ends']
            correct_labels = data['correct_labels']
    except KeyError as e:
        logger.log_message(f"Error: Missing expected array in NPZ – {e}")
        return
    except Exception as e:
        logger.log_message(f"Error loading {npz_path}: {e}")
        return

    num_samples = len(attributions)
    if num_samples == 0:
        logger.log_message("No samples found in the file.")
        return

    logger.log_message(f"{num_samples} samples found; generating plots…")

    # loop through each sample and plot
    for attrs, seq, lbl, uid, peak, correct_label in zip(
            attributions, sequences, labels, uids, peak_start_ends, correct_labels
    ):
        # sum the scores across 128 embedding channels to get one score per position
        seq_scores = attrs.sum(axis=1)

        out_dir = os.path.join(output_dir, f"Correctly_Predicted:{correct_label}")
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"ID-{uid}.png")
        plot_single_saliency(seq_scores, seq, uid, int(lbl), tuple(peak), filename)

    logger.log_message(f"Successfully saved {num_samples} plots to {output_dir}")
