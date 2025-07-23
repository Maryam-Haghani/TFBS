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
from captum.attr import Saliency, NoiseTunnel, IntegratedGradients, DeepLift

from early_stop import EarlyStopping
from visualization import plot_roc_pr, plot_saliency_maps_from_file
from utils import adjust_learning_rate

class Train_Test:
    def __init__(self, logger, max_length, device, eval_batch_size, test_mode="df", training_params=None):
        self.logger = logger
        self.max_length = max_length
        self.device = device
        self.eval_batch_size = eval_batch_size
        self.test_mode = test_mode
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

    def _init_interpret(self, num_saliency_samples, method):
        self.interp_state = {
            'num_wanted': num_saliency_samples,
            'correct_found': 0,
            'incorrect_found': 0,
            'records': []
        }

        if method == 'smooth':
            base = Saliency(self.model)
            self.interpreter = NoiseTunnel(base)
            # defaults for NoiseTunnel
            self._interp_kwargs = {
                'nt_type': method,
                'nt_samples': 50,
                'stdevs': 0.02
            }
        elif method == 'integrated':
            self.interpreter = IntegratedGradients(self.model)
            self._interp_kwargs = {
                'n_steps': 100
            }
        elif method == 'deeplift':
            self.interpreter = DeepLift(self.model)
            self._interp_kwargs = {}  # no extra args
        else:  # plain Saliency
            self.interpreter = Saliency(self.model)
            self._interp_kwargs = {}  # no extra args

    def _get_attributions(self, mask, sequences, uids, starts, ends, pred_labels, true_labels, embeddings, correct):
        idxs = torch.where(mask)[0]

        found = (
            self.interp_state['correct_found']
            if correct
            else self.interp_state['incorrect_found']
        )
        to_take = min(
                len(idxs),
                self.interp_state['num_wanted'] - found
            )

        if to_take == 0:
            return None

        chosen = idxs[:to_take]
        # compute saliency
        chosen_emb = embeddings[chosen]
        chosen_tgt = pred_labels[chosen]  # model's correct prediction
        # IntegratedGradients and DeepLift need a baseline
        if isinstance(self.interpreter, IntegratedGradients)\
                or isinstance(self.interpreter, DeepLift):
            # choose baseline to be all PAD tokens
            baseline = torch.full_like(chosen_emb, 4, requires_grad=True)
            attributions = self.interpreter.attribute(
                chosen_emb,
                baselines=baseline,
                target=chosen_tgt,
                **self._interp_kwargs
            )
        else:
            # covers Saliency and NoiseTunnel
            attributions = self.interpreter.attribute(
                chosen_emb,
                target=chosen_tgt,
                **self._interp_kwargs
            )
        # package results
        attributions_info = [
            (int(idx), sequences[idx], uids[idx], (starts[idx], ends[idx]),
             int(true_labels[idx]), attributions[j].cpu().detach())
            for j, idx in enumerate(chosen)
        ]
        return attributions_info
    def _interpret(self, sequences, uids, starts, ends, pred_labels, true_labels, embeddings, correct=True):
        """
            Runs whichever attribution method was selected.
            Returns a list of tuples
              (batch_idx, seq, uid, (start,end), true_label, attribution_tensor)
            or None if no correct samples in this batch.
        """
        if correct:
            mask = pred_labels == true_labels.squeeze()
        else:
            mask = pred_labels != true_labels.squeeze()

        attributions_info = self._get_attributions(mask, sequences, uids, starts, ends, pred_labels, true_labels, embeddings, correct)
        return attributions_info

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

    def _gather_interpretation(self, attributions_info, correct=True):
        """Store per‐example attribution"""
        for idx, seq, uid, se, true_lab, attr in attributions_info:
            self.interp_state['records'].append({
                'seq': seq,
                'uid': uid,
                'start_end': se,
                'label': true_lab,
                'attr': attr.cpu().numpy()[self.max_length-len(seq):], # get attribute for sequence embeddings
                'correct': correct
            })
            if correct:
                self.interp_state['correct_found'] += 1
            else:
                self.interp_state['incorrect_found'] += 1

    def _save_interpretation_results(self, model_name, test_result_dir, saliency_method):
        """Save all collected attributions to .npz and plot."""
        recs = self.interp_state['records']
        self.logger.log_message(
            f"\nCollected attributions for {len(recs)} samples."
        )
        # unpack arrays
        seqs = np.array([r['seq'] for r in recs], dtype=object)
        correct_labs = np.array([r['correct'] for r in recs], dtype=bool)
        uids = np.array([r['uid'] for r in recs], dtype=object)
        ses = np.array([r['start_end'] for r in recs], dtype=object)
        labs = np.array([r['label'] for r in recs], dtype=int)
        attrs = np.empty(len(recs), dtype=object)
        for i, r in enumerate(recs):
            attrs[i] = r['attr']

        saliency_dir = os.path.join(test_result_dir, 'saliency_map', saliency_method)
        os.makedirs(saliency_dir, exist_ok=True)

        saliency_path = os.path.join(saliency_dir, f'{model_name}_saliency_results.npz')
        np.savez_compressed(
            saliency_path,
            attributions=attrs,
            sequences=seqs,
            peak_start_ends=ses,
            uids=uids,
            labels=labs,
            correct_labels=correct_labs
        )
        self.logger.log_message(f'Saliency map results saved to {saliency_dir}')

        plot_dir = os.path.join(saliency_dir, 'plots', f'{model_name}')
        os.makedirs(plot_dir, exist_ok=True)
        self.logger.log_message(f'Visualizing Saliency maps into {plot_dir}...')
        plot_saliency_maps_from_file(self.logger, npz_file_path=saliency_path, output_dir=plot_dir)

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

    def _test(self, test_loader, model_name,
                            loss_fn=None, test_result_dir=None, num_saliency_samples=0, saliency_method='smooth'):
        """
            Runs either a full validation (with loss) or a prediction pass (saving per‐sample results
            and optionally collecting saliency attributions for a handful of correct samples).
            Returns:
              - validation: (accuracy, auroc, auprc, f1, mcc, avg_loss)
              - prediction: (accuracy, auroc, auprc, f1, mcc)
            """
        # Mode flags
        prediction_mode = (loss_fn is None)
        do_interpret = prediction_mode and num_saliency_samples > 0

        self.model.eval()

        if do_interpret:
            self.logger.log_message(f"During prediction will interpret results for {num_saliency_samples} correct and "
                                    f"{num_saliency_samples} incorrect predicted samples based on '{saliency_method}' method")
            self._init_interpret(num_saliency_samples, saliency_method)

        # initialization for full evaluation
        test_loss, correct, total = 0, 0, 0
        all_pos_probs = []  # probabilities of positive class (class 1)
        all_true_labels, predicted_labels = [], []
        results = [] # only in prediction_mode

        # for each batch
        for sequences, data, uids, peak_starts, peak_ends, true_labels in test_loader:
            data, true_labels = data.to(self.device), true_labels.to(self.device)
            # predictions
            if do_interpret:
                # get embeddings and enable grads
                embeddings = self.model.backbone.embeddings(data)  # FloatTensor [B,L,D]
                if do_interpret:
                    embeddings.requires_grad_()
                output = self.model(embeddings)
            else:
                with torch.no_grad():
                    output = self.model(data)

            logits = output.logits if isinstance(output, SequenceClassifierOutput) else output
            pred_labels = logits.argmax(dim=1)  # predicted class index
            probs = F.softmax(logits, dim=1)  # probabilities for each class
            correct += pred_labels.eq(true_labels.view_as(pred_labels)).sum().item()
            total += data.size(0)

            if do_interpret: # because model is computing grads, have to detach first
                all_pos_probs.append(probs[:, 1].detach().cpu().numpy())
                all_true_labels.append(true_labels.detach().cpu().numpy())
                predicted_labels.append(pred_labels.detach().view(-1).cpu().numpy())
                probs = probs.detach().cpu().numpy()
            else:
                all_pos_probs.append(probs[:, 1].cpu().numpy())
                all_true_labels.append(true_labels.cpu().numpy())
                predicted_labels.append(pred_labels.view(-1).cpu().numpy())
                probs = probs.cpu().numpy()

            if prediction_mode:
                # collect results for each sample in the batch
                for seq, uid, peak_start, peak_end, true_label, pred_label, prob\
                    in zip(sequences, uids, peak_starts, peak_ends, true_labels, pred_labels, probs):
                    results.append({'uid': uid.item(),
                                    'sequence': seq,
                                    'peak_start_end': (peak_start.item(), peak_end.item()),
                                    'true_label': true_label.item(),
                                    'probability': np.round(prob, 2),
                                    'predicted_label': pred_label.item(),
                                    'prediction_probability': round(prob[pred_label.item()].item(), 2),
                                    'positive_probability': round(prob[1].item(), 2),
                                    'correct_label': true_label.item() == pred_label.item()
                                    })

                if do_interpret:
                    if self.interp_state['correct_found'] < self.interp_state['num_wanted']:
                        attributions_info = self._interpret(sequences, uids, peak_starts, peak_ends,
                                                           pred_labels, true_labels, embeddings)
                        if attributions_info:
                            self._gather_interpretation(attributions_info)

                    if self.interp_state['incorrect_found'] < self.interp_state['num_wanted']:
                        attributions_info = self._interpret(sequences, uids, peak_starts, peak_ends,
                                                           pred_labels, true_labels, embeddings, correct=False)
                        if attributions_info:
                            self._gather_interpretation(attributions_info, correct=False)
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

            if do_interpret:
                self._save_interpretation_results(model_name, test_result_dir, saliency_method)

            return accuracy, auroc, auprc, f1, mcc
        else: # validating
            # average loss over all samples
            avg_test_loss = test_loss / total
            self.logger.log_message(f'\nValidation loss: {avg_test_loss:.4f}\n')
            return accuracy, auroc, auprc, f1, mcc, avg_test_loss


    # compute metrics for test dataset
    def test(self, ds_test, model_name, test_result_dir, num_saliency_samples=0, saliency_method='smooth'):
        test_loader = DataLoader(ds_test, batch_size=self.eval_batch_size, shuffle=num_saliency_samples==0)

        start_time = time.time()  # START TEST TIME
        acc, auroc, auprc, f1, mcc  = self._test(test_loader, model_name, test_result_dir=test_result_dir,
                                                 num_saliency_samples=num_saliency_samples, saliency_method=saliency_method)
        test_time = time.time() - start_time  # END TEST TIME
        return acc, auroc, auprc, f1, mcc, test_time

    def predict(self, model, ds, test_result_dir=None, model_name=None, num_saliency_samples=10, saliency_method='smooth'):
        self.model = model
        self.model.to(self.device)

        if self.test_mode=='df':
            return self.test(ds, model_name, test_result_dir, num_saliency_samples=num_saliency_samples,
                             saliency_method=saliency_method)
        elif self.test_mode=='genome':
            return self._predict(ds)
        else:
            raise ValueError(f'Given mode {self.test_mode} is not valid!')

    def _predict(self, ds):
        dataloader = DataLoader(ds, batch_size=self.eval_batch_size, shuffle=False)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for sequences, data, start_idx, end_idx in dataloader:
                data = data.to(self.device)
                output = self.model(data)
                logits = output.logits if isinstance(output, SequenceClassifierOutput) else output

                probs = F.softmax(logits, dim=1)  # probabilities for each class
                pos_probs = probs[:, 1].cpu().numpy()
                start = start_idx.numpy()
                end = end_idx.numpy()

                predictions.extend(list(zip(start, end, pos_probs)))
        return predictions