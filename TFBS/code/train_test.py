import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef
)

from early_stop import EarlyStopping
from visualization import plot_roc_pr
from utils import adjust_learning_rate

class Train_Test:
    def __init__(self, logger, device, eval_batch_size, training_params=None):
        self.logger = logger
        self.device = device
        self.eval_batch_size = eval_batch_size
        self.training = training_params

    # freeze specified layers by setting `requires_grad` to False.
    def _freeze_layers(self):
        if self.training.model_params.freeze_layers != 'none':
            for name, param in self.model.named_parameters():
                # freeze the layer if its name matches any of the freeze_layers
                if any(layer in name for layer in self.training.model_params.freeze_layers):
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
            logits = output.logits if isinstance(output, SequenceClassifierOutput) else output

            total += data.size(0)

            loss = loss_fn(logits, labels.squeeze(1))  # the average loss across batch
            batch_loss = loss.item() * data.size(0)  # multiply loss by batch size to get total loss for this batch
            train_loss += batch_loss

            loss.backward()

            if self.training.adjustable_LR.enabled:
                # warm-up and LR decay
                adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=self.training.num_epochs,
                                     lr_min=float(self.training.adjustable_LR.min),
                                     lr_max=float(self.training.adjustable_LR.max), warmup=True)

            # model weight update
            optimizer.step()

            # if self.training.adjustable_LR.enabled:
            #     self.logger.log_message("train lr is ", optimizer.state_dict()["param_groups"][0]["lr"])

            if batch_idx % log_interval == 0:
                self.logger.log_message('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()), use_time=True)

        # average loss over all samples
        avg_train_loss = train_loss / total

        return avg_train_loss

    def _test(self, test_loader, model_name, loss_fn=None, test_result_dir=None):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_pos_probs = []  # probabilities of positive class (class 1)
        all_labels = []  # true labels
        all_preds = [] # predicted labels
        results = []
        with torch.no_grad():
            for sequence, data, true_labels in test_loader:
                data, true_labels = data.to(self.device), true_labels.to(self.device)
                output = self.model(data)
                logits = output.logits if isinstance(output, SequenceClassifierOutput) else output

                probs = F.softmax(logits, dim=1) # probabilities for each class
                pred_labels = logits.argmax(dim=1, keepdim=True)  # predicted class index
                correct += pred_labels.eq(true_labels.view_as(pred_labels)).sum().item()
                total += data.size(0)

                all_pos_probs.append(probs[:, 1].cpu().numpy())
                all_labels.append(true_labels.cpu().numpy())
                all_preds.append(pred_labels.view(-1).cpu().numpy())

                if loss_fn is not None: # validating
                    # Compute the loss for the batch and accumulate it
                    batch_avg_loss = loss_fn(logits, true_labels.squeeze(1))  # calculates the average loss per sample in the batch.
                    batch_loss = batch_avg_loss.item() * data.size(0)  # multiply loss by batch size (data.size(0)) to get total loss for this batch
                    test_loss += batch_loss
                else:
                    # Collect results for each sample in the batch
                    for seq, true_label, pred_label, prob in zip(sequence, true_labels, pred_labels, probs):
                        results.append({'sequence': seq,
                                        'true_label': true_label.item(),
                                        'probability':  np.round(prob.cpu().numpy(), 2),
                                        'predicted_label': pred_label.item(),
                                        'prediction_probability': round(prob[pred_label.item()].item(), 2),
                                        'positive_probability': round(prob[1].item(), 2)
                                         })

        # flatten the lists
        all_pos_probs = np.concatenate(all_pos_probs)
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)

        # compute metrics
        accuracy = correct / total if total > 0 else 0
        try:
            auroc = roc_auc_score(all_labels, all_pos_probs)
        except ValueError:
            auroc = float('nan')  # when only one class present
        auprc = average_precision_score(all_labels, all_pos_probs)
        f1 = f1_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)

        self.logger.log_message(f'Accuracy: {accuracy:.2f}\n')
        self.logger.log_message(f'AUROC:    {auroc:.2f}\n')
        self.logger.log_message(f'AUPRC:    {auprc:.2f}\n')
        self.logger.log_message(f'F1 Score: {f1:.2f}\n')
        self.logger.log_message(f'MCC:      {mcc:.2f}')

        if loss_fn is not None:  # validating
            # average loss over all samples
            avg_test_loss = test_loss / total
            self.logger.log_message(f'\nValidation loss: {avg_test_loss:.4f}\n')
            return accuracy, auroc, auprc, f1, mcc, avg_test_loss
        else: # save test results
            test_result_path = os.path.join(test_result_dir, f'{model_name}.csv')
            df = pd.DataFrame(results)
            df.to_csv(test_result_path, index=False)
            self.logger.log_message(f'Test results saved to {test_result_path}')

            test_result_dir = os.path.join(test_result_dir, 'plots')
            os.makedirs(test_result_dir, exist_ok=True)

            plot_roc_pr('ROC', all_labels, all_pos_probs,'False Positive Rate',
                        'True Positive Rate', test_result_dir, model_name)
            plot_roc_pr('PR', all_labels, all_pos_probs, 'Recall',
                        'Precision', test_result_dir, model_name)

            return accuracy, auroc, auprc, f1, mcc

    # compute metrics for test dataset
    def test(self, ds_test, model_name, test_result_dir):
        test_loader = DataLoader(ds_test, batch_size=self.eval_batch_size, shuffle=True)

        start_time = time.time()  # START TEST TIME
        acc, auroc, auprc, f1, mcc  = self._test(test_loader, model_name, test_result_dir=test_result_dir)
        test_time = time.time() - start_time  # END TEST TIME
        return acc, auroc, auprc, f1, mcc, test_time

    def train(self, ds_train, ds_val, model_name, model_dir, wandb):
        self._freeze_layers()

        self.logger.log_message(f'model name: {model_name}')

        # total number of parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.log_message(f"Total parameters: {total_params}")

        # trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.log_message(f"Trainable parameters: {trainable_params}")


        self.logger.log_message(f"Training...")
        train_loader = DataLoader(ds_train, batch_size=self.training.model_params.train_batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=self.eval_batch_size, shuffle=False)

        loss_fn = nn.CrossEntropyLoss()

        # self.model.parameters(): the optimizer will update all parameters of the model during backpropagation.
        # This includes the pre-trained weights and the classification head that have been added to the model.
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=float(self.training.model_params.learning_rate),
                                weight_decay=float(self.training.model_params.weight_decay))

        self.model.to(self.device)

        early_stopping = EarlyStopping(
            logger = self.logger,
            patience=self.training.early_stopping.patience,
            verbose=True,
            delta=self.training.early_stopping.delta
        )

        start_time = time.time()  # START TRAINING TIME

        # training
        for epoch in range(self.training.num_epochs):
            self.logger.log_message(f'Epoch {epoch+1}/{self.training.num_epochs}')

            train_loss = self._train(train_loader, optimizer, epoch, loss_fn)
            val_acc, val_auroc, val_auprc, val_f1, val_mcc, val_loss = self._test(
                val_loader, model_name, loss_fn=loss_fn
            )

            if wandb is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "train loss": train_loss,
                    "val loss": val_loss,
                    "val ACC": val_acc,
                    "val AUROC": val_auroc,
                    "val AUPRC": val_auprc,
                    "val F1": val_f1,
                    "val MCC": val_mcc
                })
            else:
                self.logger.log_message("wandb is not enabled. Skipping logging.")

            # check early stopping criteria
            early_stopping(val_loss, self.model, epoch)
            if early_stopping.early_stop:
                self.logger.log_message(f"Early stopping triggered. Stopping training at epoch {epoch+1}.")
                break

        training_time = time.time() - start_time  # END TRAINING TIME

        if early_stopping.early_stop:
            best_epoch = early_stopping.best_epoch
        else:
            best_epoch = self.training.num_epochs -1

        # reload the best model weights saved during training
        if early_stopping.best_model_state is not None:
            self.model.load_state_dict(early_stopping.best_model_state)
            self.logger.log_message(f"Loaded best model weights from epoch {best_epoch+1}.") # converting 0-indexed to 1-indexed

            # save the model
            model_path = os.path.join(model_dir, f'{model_name}.pt')
            torch.save(self.model.state_dict(), model_path)
            self.logger.log_message(f"Best model saved as: {model_path}")

        return f'{trainable_params} / {total_params}', best_epoch+1, val_acc, val_auroc, val_auprc, val_f1, val_mcc, training_time