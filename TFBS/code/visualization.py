import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

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