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
from early_stop import EarlyStopping

class FineTune:
    def __init__(self, model, tokenizer, max_seq_length, use_padding, device, model_dir, model_params, freeze_layers=[]):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_padding = use_padding
        self.device = device
        self.model_dir = model_dir

        self.num_epochs = model_params.get('num_epochs', 3)
        self.batch_size = model_params.get('batch_size', 32)
        self.learning_rate = model_params.get('learning_rate', 1e-4)
        self.weight_decay = model_params.get('weight_decay', 0.1)

        self.freeze_layers = freeze_layers

        # Early stopping parameters (optional)
        self.early_stopping_patience = model_params.get('early_stopping_patience', 5)
        self.early_stopping_delta = model_params.get('early_stopping_delta', 0.0)

        self._freeze_layers()

    # freeze specified layers by setting `requires_grad` to False.
    def _freeze_layers(self):
        for name, param in self.model.named_parameters():
            # freeze the layer if its name matches any of the freeze_layers
            if any(layer in name for layer in self.freeze_layers):
                param.requires_grad = False
                print(f"Freezing layer: {name}")

    def _train(self, train_loader, optimizer, epoch, loss_fn, log_interval=10):
        self.model.train()
        train_loss = 0
        total = 0
        for batch_idx, (sequence, data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            total += data.size(0)

            loss = loss_fn(output, target.squeeze())  # the average loss across batch
            batch_loss = loss.item() * data.size(0)  # multiply loss by batch size to get total loss for this batch
            train_loss += batch_loss

            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        # average loss over all samples
        avg_train_loss = train_loss / total

        return avg_train_loss

    def _test(self, test_loader, loss_fn=None, test_result_path=None):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        results = []
        with torch.no_grad():
            for sequence, data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                probs = F.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

                if loss_fn is not None: # validating
                    # Compute the loss for the batch and accumulate it
                    batch_avg_loss = loss_fn(output, target.squeeze())  # calculates the average loss per sample in the batch.
                    batch_loss = batch_avg_loss.item() * data.size(0)  # multiply loss by batch size (data.size(0)) to get total loss for this batch
                    test_loss += batch_loss
                else:
                    # Collect results for each sample in the batch
                    for seq, label, p, prob in zip(sequence, target, pred, probs):
                        results.append({'sequence': seq,
                                        'true_label': label.item(),
                                        'predicted_label': p.item(),
                                        'prediction_probability': prob[p.item()].item()
                                         })

        accuracy = 100.0 * correct / total if total > 0 else 0
        print(f'Accuracy: {accuracy:.2f}%\n')
        
        if loss_fn is not None:  # validating
            # average loss over all samples
            avg_test_loss = test_loss / total
            print(f'\nValidation loss: {avg_test_loss:.4f}\n')
            return avg_test_loss
        else: # save test results
            df = pd.DataFrame(results)
            df.to_csv(test_result_path, index=False)
            print(f'Test results saved to {test_result_path}')

    def _plot_loss(self, train_losses, val_losses, loss_dir, best_epoch):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(loss_dir, f'loss_{best_epoch}.png'), dpi=300, bbox_inches="tight")

    # compute metrics for test dataset
    def test(self, ds_test, test_result_path):
        print("Getting test set accuracy...")
        test_loader = DataLoader(ds_test, batch_size=self.batch_size, shuffle=True)
        self._test(test_loader, test_result_path=test_result_path)

    # finetune the model based on the given dataset
    def finetune(self, ds_train, ds_val, loss_dir, model_name):
        print(f"Performing finetuning...")
        train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False)

        loss_fn = nn.CrossEntropyLoss()

        # self.model.parameters(): the optimizer will update (fine-tuned) all parameters of the model during backpropagation.
        # This includes the pre-trained weights and the classification head that have been added to the model.
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.model.to(self.device)

        train_losses = []
        val_losses = []

        early_stopping = EarlyStopping(
            patience=self.early_stopping_patience,
            verbose=True,
            delta=self.early_stopping_delta
        )

        # training
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')

            train_loss = self._train(train_loader, optimizer, epoch, loss_fn)
            val_loss = self._test(val_loader, loss_fn=loss_fn)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # check early stopping criteria
            early_stopping(val_loss, self.model, epoch)
            if early_stopping.early_stop:
                print(f"Early stopping triggered. Stopping training at epoch {epoch+1}.")
                break

        if early_stopping.early_stop:
            best_epoch = early_stopping.best_epoch + 1  # converting 0-indexed to 1-indexed
        else:
            best_epoch = self.num_epochs
            
            
        # reload the best model weights saved during training
        if early_stopping.best_model_state is not None:
            self.model.load_state_dict(early_stopping.best_model_state)
            print(f"Loaded best model weights from epoch {best_epoch}.")

            # save the model
            model_path = os.path.join(self.model_dir, f'{model_name}_epoch_{best_epoch}.pt')
            torch.save(self.model.state_dict(), model_path)
            print(f"Best model saved as: {model_path}")

        self._plot_loss(train_losses, val_losses, loss_dir, best_epoch)
        return self.model

    # load the saved parameters to the model
    def load(self, finetuned_model_name):
        model_path = os.path.join(self.model_dir, finetuned_model_name)
        print(f"Loading finetuned model '{model_path}'...")
        self.model.load_state_dict(torch.load(model_path))
        return self.model.to(self.device)