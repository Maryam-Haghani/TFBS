import os
import matplotlib.pyplot as plt

def plot_loss(param_combinations, train_loss_per_param, val_loss_per_param, plot_dir, plot_name):
    plt.figure(figsize=(8, 6))
    for i, param in enumerate(param_combinations):
        plt.plot(train_loss_per_param[i], label=f'Train ({param})', linestyle='-', linewidth=2)
        plt.plot(val_loss_per_param[i], label=f'Validation ({param})', linestyle='--', linewidth=2)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Losses for Different Hyperparameter Combinations')
    plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), fontsize='small')

    plot_path = os.path.join(plot_dir, plot_name)
    plt.savefig(plot_path)
    print(f"Saved loss plots as PDF files at {plot_path}.")

def plot_auroc_auprc(type, param_combinations, values, plot_dir, name):
    plt.figure(figsize=(8, 6))
    for i, param in enumerate(param_combinations):
        plt.plot(values[i], label=f'{type} ({param})', linestyle=':')
    plt.xlabel('Epochs')
    plt.ylabel(type)
    plt.title(f'{type} for Different Parameter Combinations')
    plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.grid(True)
    plot_path = os.path.join(plot_dir, f'{type}_{name}.pdf')
    plt.savefig(plot_path)
    print(f"Saved loss plots as PDF files at {plot_path}.")