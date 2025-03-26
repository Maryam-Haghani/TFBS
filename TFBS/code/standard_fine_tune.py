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
from utils import serialize_dict, serialize_array

from hyena_dna import HyenaDNAModel

class FineTune:
    def __init__(self, logger, pretrained_model_name, device, model_dir, training_params):
        self.logger = logger
        
        self.device = device
        self.model_dir = model_dir
        
        self.training = training_params

        hyena_model = HyenaDNAModel(pretrained_model_name=pretrained_model_name, use_head=True, device=self.device)
        self.model = hyena_model.load_pretrained_model()

        self._freeze_layers()
        
        self.model_name = (f'fine-tuned_model_{serialize_dict(self.training.model_params)}'
                           f'_freeze_layers-{serialize_array(self.training.freeze_layers)}')

    # freeze specified layers by setting `requires_grad` to False.
    def _freeze_layers(self):
        for name, param in self.model.named_parameters():
            # freeze the layer if its name matches any of the freeze_layers
            if any(layer in name for layer in self.training.freeze_layers):
                param.requires_grad = False
                self.logger.info(f"Freezing layer: {name}")

    def _train(self, train_loader, optimizer, epoch, loss_fn, log_interval=10):
        self.model.train()
        train_loss = 0
        total = 0
        for batch_idx, (sequence, data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            output = self.model(data)

            total += data.size(0)

            loss = loss_fn(output, target.squeeze(1))  # the average loss across batch
            batch_loss = loss.item() * data.size(0)  # multiply loss by batch size to get total loss for this batch
            train_loss += batch_loss

            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
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
                    batch_avg_loss = loss_fn(output, target.squeeze(1))  # calculates the average loss per sample in the batch.
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
        self.logger.info(f'Accuracy: {accuracy:.2f}%\n')
        
        if loss_fn is not None:  # validating
            # average loss over all samples
            avg_test_loss = test_loss / total
            self.logger.info(f'\nValidation loss: {avg_test_loss:.4f}\n')
            return accuracy, avg_test_loss
        else: # save test results
            df = pd.DataFrame(results)
            df.to_csv(test_result_path, index=False)
            self.logger.info(f'Test results saved to {test_result_path}')
            return accuracy

    def _plot_loss(self, train_losses, val_losses, loss_dir, plot_name):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(loss_dir, plot_name), dpi=300, bbox_inches="tight")

    # compute metrics for test dataset
    def test(self, ds_test, test_result_dir):
        self.logger.info("Getting test set accuracy...")
        
        test_result_path = os.path.join(test_result_dir, f'{self.model_name}.csv')

        test_loader = DataLoader(ds_test, batch_size=self.training.model_params.batch_size, shuffle=True)
        acc = self._test(test_loader, test_result_path=test_result_path)
        return acc

    # finetune the model based on the given dataset
    def finetune(self, ds_train, ds_val, loss_dir):
        self.logger.info(f"Training with batch_size={self.training.model_params.batch_size},"
              f"learning_rate={self.training.model_params.learning_rate},"
              f"weight_decay={self.training.model_params.weight_decay}")

        self.logger.info(f"Performing finetuning...")
        train_loader = DataLoader(ds_train, batch_size=self.training.model_params.batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=self.training.model_params.batch_size, shuffle=False)

        loss_fn = nn.CrossEntropyLoss()

        # self.model.parameters(): the optimizer will update (fine-tuned) all parameters of the model during backpropagation.
        # This includes the pre-trained weights and the classification head that have been added to the model.
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=float(self.training.model_params.learning_rate),
                                weight_decay=float(self.training.model_params.weight_decay))

        self.model.to(self.device)

        train_losses = []
        val_losses = []

        early_stopping = EarlyStopping(
            patience=self.training.early_stopping.patience,
            verbose=True,
            delta=self.training.early_stopping.delta
        )

        # training
        for epoch in range(self.training.num_epochs):
            self.logger.info(f'Epoch {epoch+1}/{self.training.num_epochs}')

            train_loss = self._train(train_loader, optimizer, epoch, loss_fn)
            _, val_loss = self._test(val_loader, loss_fn=loss_fn)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # check early stopping criteria
            early_stopping(val_loss, self.model, epoch)
            if early_stopping.early_stop:
                self.logger.info(f"Early stopping triggered. Stopping training at epoch {epoch+1}.")
                break

        if early_stopping.early_stop:
            best_epoch = early_stopping.best_epoch + 1  # converting 0-indexed to 1-indexed
        else:
            best_epoch = self.training.num_epochs
            best_epoch = self.training.num_epochs
            
            
        # reload the best model weights saved during training
        if early_stopping.best_model_state is not None:
            self.model.load_state_dict(early_stopping.best_model_state)
            self.logger.info(f"Loaded best model weights from epoch {best_epoch}.")

            # save the model
            model_path = os.path.join(self.model_dir, f'{self.model_name}.pt')
            torch.save(self.model.state_dict(), model_path)
            self.logger.info(f"Best model saved as: {model_path}")

        self._plot_loss(train_losses, val_losses, loss_dir, f'{self.model_name}.png')
        
        return self.model.to(self.device)

    # load the saved parameters to the model
    def load(self, finetuned_model_name):
        model_path = os.path.join(self.model_dir, finetuned_model_name)
        self.logger.info(f"Loading finetuned model '{model_path}'...")

        state_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict)
        return self.model.to(self.device)