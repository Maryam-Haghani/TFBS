import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from early_stop import EarlyStopping
from utils import serialize_dict, serialize_array
from sklearn.metrics import roc_auc_score, average_precision_score
from visualization import plot_roc_pr

class FineTune:
    def __init__(self, logger, device, model_dir, training_params):
        self.logger = logger
        
        self.device = device
        self.model_dir = model_dir
        
        self.training = training_params

    # freeze specified layers by setting `requires_grad` to False.
    def _freeze_layers(self):
        for name, param in self.model.named_parameters():
            # freeze the layer if its name matches any of the freeze_layers
            if any(layer in name for layer in self.training.freeze_layers):
                param.requires_grad = False
                self.logger.log_message(f"Freezing layer: {name}")

    def _train(self, train_loader, optimizer, epoch, loss_fn, log_interval=10):
        self.model.train()
        train_loss = 0
        total = 0
        for batch_idx, (sequence, data, labels) in enumerate(train_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            output = self.model(data)

            total += data.size(0)

            loss = loss_fn(output, labels.squeeze(1))  # the average loss across batch
            batch_loss = loss.item() * data.size(0)  # multiply loss by batch size to get total loss for this batch
            train_loss += batch_loss

            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                self.logger.log_message('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()), use_time=True)

        # average loss over all samples
        avg_train_loss = train_loss / total

        return avg_train_loss

    def _test(self, test_loader, loss_fn=None, test_result_dir=None):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_pos_probs = []  # probabilities of positive class (class 1)
        all_labels = []  # true labels
        results = []
        with torch.no_grad():
            for sequence, data, true_labels in test_loader:
                data, true_labels = data.to(self.device), true_labels.to(self.device)
                output = self.model(data)

                probs = F.softmax(output, dim=1) # probabilities for each class
                pred_labels = output.argmax(dim=1, keepdim=True)  # predicted class index
                correct += pred_labels.eq(true_labels.view_as(pred_labels)).sum().item()
                total += data.size(0)

                all_pos_probs.append(probs[:, 1].cpu().numpy())
                all_labels.append(true_labels.cpu().numpy())

                if loss_fn is not None: # validating
                    # Compute the loss for the batch and accumulate it
                    batch_avg_loss = loss_fn(output, true_labels.squeeze(1))  # calculates the average loss per sample in the batch.
                    batch_loss = batch_avg_loss.item() * data.size(0)  # multiply loss by batch size (data.size(0)) to get total loss for this batch
                    test_loss += batch_loss
                else:
                    # Collect results for each sample in the batch
                    for seq, true_label, pred_label, prob in zip(sequence, true_labels, pred_labels, probs):
                        results.append({'sequence': seq,
                                        'true_label': true_label.item(),
                                        'probability':  prob.cpu().numpy(),
                                        'predicted_label': pred_label.item(),
                                        'prediction_probability': prob[pred_label.item()].item(),
                                        'positive_probability': prob[1].item()
                                         })

        # flatten the lists
        all_pos_probs = np.concatenate(all_pos_probs)
        all_labels = np.concatenate(all_labels)

        accuracy = 100.0 * correct / total if total > 0 else 0
        try:
            auroc = roc_auc_score(all_labels, all_pos_probs)
        except ValueError:
            auroc = float('nan')  # when only one class present
        auprc = average_precision_score(all_labels, all_pos_probs)

        self.logger.log_message(f'Accuracy: {accuracy:.2f}%\n')
        self.logger.log_message(f'AUROC: {auroc:.2f}\n')
        self.logger.log_message(f'AUPRC: {auprc:.2f}\n')
        
        if loss_fn is not None:  # validating
            # average loss over all samples
            avg_test_loss = test_loss / total
            self.logger.log_message(f'\nValidation loss: {avg_test_loss:.4f}\n')
            return accuracy, auroc, auprc, avg_test_loss
        else: # save test results
            test_result_path = os.path.join(test_result_dir, f'{self.model_name}.csv')
            df = pd.DataFrame(results)
            df.to_csv(test_result_path, index=False)
            self.logger.log_message(f'Test results saved to {test_result_path}')

            plot_roc_pr('ROC', all_labels, all_pos_probs,'False Positive Rate',
                        'True Positive Rate', test_result_dir, self.model_name)
            plot_roc_pr('PR', all_labels, all_pos_probs, 'Recall',
                        'Precision', test_result_dir, self.model_name)

            return accuracy, auroc, auprc

    # compute metrics for test dataset
    def test(self, ds_test, test_result_dir):
        self.logger.log_message("Getting test set accuracy...")

        test_loader = DataLoader(ds_test, batch_size=self.training.model_params.batch_size, shuffle=True)
        acc, auroc, auprc = self._test(test_loader, test_result_dir=test_result_dir)
        return acc, auroc, auprc

    # finetune the model based on the given dataset
    def finetune(self, ds_train, ds_val):
        self._freeze_layers()

        self.model_name = (f'fine-tuned_model_{serialize_dict(self.training.model_params)}'
                           f'_freeze_layer-{serialize_array(self.training.freeze_layers)}')

        self.logger.log_message(f'model name: {self.model_name}')

        # total number of parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.log_message(f"Total parameters: {total_params}")

        # trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.log_message(f"Trainable parameters: {trainable_params}")


        self.logger.log_message(f"Performing finetuning...")
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
        auroc_per_epoch = []
        auprc_per_epoch = []

        early_stopping = EarlyStopping(
            patience=self.training.early_stopping.patience,
            verbose=True,
            delta=self.training.early_stopping.delta
        )

        # training
        for epoch in range(self.training.num_epochs):
            self.logger.log_message(f'Epoch {epoch+1}/{self.training.num_epochs}')

            train_loss = self._train(train_loader, optimizer, epoch, loss_fn)
            _, auroc, auprc, val_loss = self._test(val_loader, loss_fn=loss_fn)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            auroc_per_epoch.append(auroc)
            auprc_per_epoch.append(auprc)

            # check early stopping criteria
            early_stopping(val_loss, self.model, epoch)
            if early_stopping.early_stop:
                self.logger.log_message(f"Early stopping triggered. Stopping training at epoch {epoch+1}.")
                break

        if early_stopping.early_stop:
            best_epoch = early_stopping.best_epoch + 1  # converting 0-indexed to 1-indexed
        else:
            best_epoch = self.training.num_epochs
            best_epoch = self.training.num_epochs
            
            
        # reload the best model weights saved during training
        if early_stopping.best_model_state is not None:
            self.model.load_state_dict(early_stopping.best_model_state)
            self.logger.log_message(f"Loaded best model weights from epoch {best_epoch}.")

            # save the model
            model_path = os.path.join(self.model_dir, f'{self.model_name}.pt')
            torch.save(self.model.state_dict(), model_path)
            self.logger.log_message(f"Best model saved as: {model_path}")

        return (self.model.to(self.device), f'{trainable_params} / {total_params}', train_losses,
                val_losses, auroc_per_epoch, auprc_per_epoch)

    # load the saved parameters to the model
    def load(self, finetuned_model_name):
        model_path = os.path.join(self.model_dir, finetuned_model_name)
        self.logger.log_message(f"Loading finetuned model '{model_path}'...")

        state_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict)
        return self.model.to(self.device)