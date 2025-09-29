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
    def __init__(self, logger, max_length, device, eval_batch_size, training_params=None):
        self.logger = logger
        self.max_length = max_length
        self.device = device
        self.eval_batch_size = eval_batch_size
        self.training = training_params

    # freeze specified layers by setting `requires_grad` to False.
    def _freeze_layers(self):
        # reset all params to trainable before setting a new freeze pattern
        for _, p in self.model.named_parameters():
            p.requires_grad = True

        freeze_layers = self.training.model_params.freeze_layers
        self.logger.log_message(f"Freeze layers: {freeze_layers}")

        if freeze_layers != 'none':
            for name, param in self.model.named_parameters():
                # freeze the layer if its name contains the freeze_layers string
                if freeze_layers in name:
                    param.requires_grad = False
                    self.logger.log_message(f"Freezing layer: {name}")

    def _train(self, train_loader, optimizer, epoch, loss_fn, log_interval=10):
        self.model.train()
        train_loss = 0
        total = 0
        for batch_idx, (sequence, data, uid, peak_start, peak_end, labels) in enumerate(train_loader):
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

    def train(self, ds_train, ds_val, model_name, model_params, wandb, model_dir = None):

        save_model = model_dir is not None
        self.training.model_params = model_params

        self._freeze_layers()

        # total parameters (all, trainable + frozen)
        total_params = sum(p.numel() for p in self.model.parameters())
        # trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # frozen parameters
        frozen_params = total_params - trainable_params

        self.logger.log_message(f"Total parameters: {total_params}")
        self.logger.log_message(f"Trainable parameters: {trainable_params}")
        self.logger.log_message(f"Frozen parameters: {frozen_params}")


        self.logger.log_message(f"Training...")

        loss_fn = nn.CrossEntropyLoss()

        # self.model.parameters(): the optimizer will update all parameters of the model during backpropagation.
        # This includes the pre-trained weights and the classification head that have been added to the model.
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=float(self.training.model_params.learning_rate),
                                weight_decay=float(self.training.model_params.weight_decay))

        self.model.to(self.device)

        train_loader = DataLoader(ds_train, batch_size=self.training.model_params.train_batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=self.eval_batch_size, shuffle=False)
        early_stopping = EarlyStopping(
            logger = self.logger,
            patience=self.training.early_stopping.patience,
            verbose=True,
            delta=self.training.early_stopping.delta
        )

        start_time = time.time()  # START TRAINING TIME
        # training
        val_metrics = {}
        for epoch in range(self.training.num_epochs):
            self.logger.log_message(f'Epoch {epoch+1}/{self.training.num_epochs}')

            train_loss = self._train(train_loader, optimizer, epoch, loss_fn)
            val_acc, val_auroc, val_auprc, val_f1, val_mcc, val_loss = self._test(
                val_loader, model_name, loss_fn=loss_fn)
            val_metrics[epoch] = (round(val_acc, 3), round(val_auroc, 3), round(val_auprc, 3),
                                  round(val_f1, 3), round(val_mcc, 3))

            log_params = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_AUROC": val_auroc,
                "val_AUPRC": val_auprc,
                "val_F1": val_f1,
                "val_MCC": val_mcc
            }

            if wandb is not None:
                wandb.log(log_params, step=epoch+1)
            else:
                self.logger.log_message("wandb is not enabled. Skipping logging.")

            # check early stopping criteria
            early_stopping(val_loss, self.model, epoch)
            if early_stopping.early_stop:
                best_epoch = early_stopping.best_epoch
                best_val_acc, best_val_auroc, best_val_auprc, best_val_f1, best_val_mcc = val_metrics[
                    best_epoch]
                self.logger.log_message(f"Early stopping triggered. Stopping training at epoch {epoch+1}.\nbest epoch: {best_epoch+1}")
                self.logger.log_message(f"Metrics for best epoch: best_val_acc: {best_val_acc}, best_val_auroc: {best_val_auroc},"
                                        f" best_val_auprc: {best_val_auprc}, best_val_f1: {best_val_f1}, best_val_mcc: {best_val_mcc}")
                break

        training_time = time.time() - start_time  # END TRAINING TIME

        if save_model: # save the model trained on whole data
            # save the model
            model_path = os.path.join(model_dir, f'{model_name}.pt')
            torch.save(self.model.state_dict(), model_path)
            self.logger.log_message(f"Model saved as: {model_path}", use_time=True)

            return f'{trainable_params} / {total_params}', training_time

        return f'{trainable_params} / {total_params}', best_epoch+1, best_val_acc, best_val_auroc, best_val_auprc, best_val_f1, best_val_mcc, training_time

    def _calculate_evaluation_metrics(self, all_pos_probs, all_true_labels, predicted_labels, correct, total):
        self.logger.log_message(f'\n--- Full Test Set Evaluation ---')

        accuracy = correct / total if total > 0 else 0
        auroc = roc_auc_score(all_true_labels, all_pos_probs) if len(np.unique(all_true_labels)) > 1 else float('nan')
        auprc = average_precision_score(all_true_labels, all_pos_probs)
        f1 = f1_score(all_true_labels, predicted_labels)
        mcc = matthews_corrcoef(all_true_labels, predicted_labels)

        self.logger.log_message(f'Accuracy: {accuracy:.2f}\n')
        self.logger.log_message(f'AUROC:    {auroc:.2f}\n')
        self.logger.log_message(f'AUPRC:    {auprc:.2f}\n')
        self.logger.log_message(f'F1 Score: {f1:.2f}\n')
        self.logger.log_message(f'MCC:      {mcc:.2f}')

        return accuracy, auroc, auprc, f1, mcc

    def _save_prediction_results(self, results, test_result_dir, model_name, all_true_labels, all_pos_probs):
        test_result_path = os.path.join(test_result_dir, f'{model_name}.csv')
        df = pd.DataFrame(results)
        df.to_csv(test_result_path, index=False)
        self.logger.log_message(f'Test results saved to {test_result_path}')

        test_result_dir = os.path.join(test_result_dir, 'plots')
        os.makedirs(test_result_dir, exist_ok=True)

        plot_roc_pr('ROC', all_true_labels, all_pos_probs,'False Positive Rate',
                    'True Positive Rate', test_result_dir, model_name)
        plot_roc_pr('PR', all_true_labels, all_pos_probs, 'Recall',
                    'Precision', test_result_dir, model_name)

    def _test(self, test_loader, model_name, loss_fn=None, test_result_dir=None):
        """
            Runs either a full validation (with loss) or a prediction pass (saving per‐sample results).
            Returns:
              - validation: (accuracy, auroc, auprc, f1, mcc, avg_loss)
              - prediction: (accuracy, auroc, auprc, f1, mcc)
            """

        prediction_mode = (loss_fn is None)

        self.model.eval()

        # initialization for full evaluation
        test_loss, correct, total = 0, 0, 0
        all_pos_probs = []  # probabilities of positive class (class 1)
        all_true_labels, predicted_labels = [], []
        results = [] # only in prediction_mode

        # for each batch
        for sequences, data, uids, peak_starts, peak_ends, true_labels in test_loader:
            data, true_labels = data.to(self.device), true_labels.to(self.device)
            # predictions
            with torch.no_grad():
                output = self.model(data)

            logits = output.logits if isinstance(output, SequenceClassifierOutput) else output
            pred_labels = logits.argmax(dim=1)  # predicted class index
            probs = F.softmax(logits, dim=1)  # probabilities for each class
            correct += pred_labels.eq(true_labels.view_as(pred_labels)).sum().item()
            total += data.size(0)

            all_pos_probs.append(probs[:, 1].cpu().numpy())
            all_true_labels.append(true_labels.cpu().numpy())
            predicted_labels.append(pred_labels.view(-1).cpu().numpy())
            probs = probs.cpu().numpy()

            if prediction_mode:
                # collect results for each sample in the batch
                for seq, uid, peak_start, peak_end, true_label, pred_label, prob\
                    in zip(sequences, uids, peak_starts, peak_ends, true_labels, pred_labels, probs):
                    results.append({'peak_uid': uid.item(),
                                    'sequence': seq,
                                    'peak_start_end': (peak_start.item(), peak_end.item()),
                                    'true_label': true_label.item(),
                                    'probability': np.round(prob, 2),
                                    'predicted_label': pred_label.item(),
                                    'prediction_probability': round(prob[pred_label.item()].item(), 2),
                                    'positive_probability': round(prob[1].item(), 2),
                                    'correct_label': true_label.item() == pred_label.item()
                                    })

            else:  # validating
                # compute the loss for the batch and accumulate it
                batch_avg_loss = loss_fn(logits, true_labels.squeeze(1))  # calculates the average loss per sample in the batch.
                test_loss += batch_avg_loss.item() * data.size(0)  # multiply loss by batch size (data.size(0)) to get total loss for this batch


        # calculate and report full evaluation metrics
        all_pos_probs = np.concatenate(all_pos_probs)
        all_true_labels = np.concatenate(all_true_labels)
        predicted_labels = np.concatenate(predicted_labels)
        accuracy, auroc, auprc, f1, mcc = self._calculate_evaluation_metrics(
                                            all_pos_probs, all_true_labels, predicted_labels, correct, total)

        if prediction_mode:
            self._save_prediction_results(results, test_result_dir, model_name, all_true_labels, all_pos_probs)
            return accuracy, auroc, auprc, f1, mcc

        else: # validating
            # average loss over all samples
            avg_test_loss = test_loss / total
            self.logger.log_message(f'\nValidation loss: {avg_test_loss:.4f}\n')
            return accuracy, auroc, auprc, f1, mcc, avg_test_loss

    def predict(self, model, ds, test_result_dir=None, model_name=None):
        self.model = model
        self.model.to(self.device)

        return self._predict_df(ds, model_name, test_result_dir)


    # compute metrics for test dataset
    def _predict_df(self, ds_test, model_name, test_result_dir):
        test_loader = DataLoader(ds_test, batch_size=self.eval_batch_size, shuffle=True)

        start_time = time.time()  # START TEST TIME
        acc, auroc, auprc, f1, mcc  = self._test(test_loader, model_name, test_result_dir=test_result_dir)
        test_time = time.time() - start_time  # END TEST TIME
        return acc, auroc, auprc, f1, mcc, test_time