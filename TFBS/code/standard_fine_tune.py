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

class FineTune:
    def __init__(self, model, tokenizer, max_seq_length, use_padding, device, loss_dir, model_params, freeze_layers=[]):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_padding = use_padding
        self.device = device

        self.num_epochs = model_params.get('num_epochs', 3)
        self.batch_size = model_params.get('batch_size', 32)
        self.learning_rate = model_params.get('learning_rate', 1e-4)
        self.weight_decay = model_params.get('weight_decay', 0.1)

        self.loss_dir = loss_dir

        self.freeze_layers = freeze_layers

        self._freeze_layers()

    # freeze the specified layers by setting `requires_grad` to False.
    def _freeze_layers(self):
        for name, param in self.model.named_parameters():
            # If the layer name matches any of the freeze_layers, freeze that
            if any(layer in name for layer in self.freeze_layers):
                param.requires_grad = False
                print(f"Freezing layer: {name}")

    def train(self, train_loader, optimizer, epoch, loss_fn, log_interval=10):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)

            loss = loss_fn(output, target.squeeze())  # the average loss across batch
            batch_loss = loss.item() * data.size(0)  # multiply loss by batch size to get total loss for this batch
            train_loss += batch_loss

            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        # Calculate average loss over all samples
        avg_train_loss = train_loss / len(train_loader.dataset)

        return avg_train_loss


    def test(self, test_loader, loss_fn):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # Compute the loss for the batch and accumulate it
                batch_avg_loss = loss_fn(output, target.squeeze())  # calculates the average loss per sample in the batch.
                batch_loss = batch_avg_loss.item() * data.size(0)  # multiply loss by batch size (data.size(0)) to get total loss for this batch
                test_loss += batch_loss

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Calculate average loss over all samples
        avg_test_loss = test_loss / len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(
            f'\nValidation set: Average loss: {avg_test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

        return avg_test_loss

    def _plot_loss(self, train_losses, val_losses):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(self.loss_dir, 'loss.png'), dpi=300, bbox_inches="tight")


    def finetune(self, ds_train, ds_val):
        train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False)

        loss_fn = nn.CrossEntropyLoss()

        # self.model.parameters(): the optimizer will update (fine-tuned) all parameters of the model during backpropagation.
        # This includes the pre-trained weights and the classification head that have been added to the model.
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.model.to(self.device)

        train_losses = []
        val_losses = []

        # Training
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')

            train_loss = self.train(train_loader, optimizer, epoch, loss_fn)
            val_loss = self.test(val_loader, loss_fn)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

        self._plot_loss(train_losses, val_losses)
        return self.model