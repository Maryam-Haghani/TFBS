import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import torch

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