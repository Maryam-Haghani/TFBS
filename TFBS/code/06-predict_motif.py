import pandas as pd
import os
import time
import subprocess
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score

from logger import CustomLogger
from utils import load_config, make_dirs

from data_split import DataSplit

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif_config_file", type=str, required=True, help="Path to the motif config file.")
    parser.add_argument("--split_config_file", type=str, required=True, help="Path to the data_split config file.")
    return parser.parse_args()

def setup_logger_and_out(config):
    """Initialize logger and output directories."""
    output_dir = make_dirs(config, make_model_dir=False)

    logger = CustomLogger(__name__, log_directory=output_dir, log_file=f'log-test-time')
    logger.log_message(f"\n\nConfiguration loaded: {config}", use_time=True)
    logger.log_message(f"Output will save at: {output_dir}")
    return logger, output_dir

def write_fasta(df, filename):
    """Write a FASTA file from a DataFrame"""
    with open(filename, 'w') as out:
        for _, row in df.iterrows():
            out.write(f">{row['peak_uid']}_{row['label']} Label:{row['label']} Peak:{row['peak']}\n{row['sequence']}\n")

def run_meme(pos_input_fasta, neg_input_fasta, out_dir, nmotifs=5, seed=0, evt= 0.05, meme_path="meme"):
    start_time = time.time()
    """Run MEME to generate motif PWM."""
    parent_meme_out = os.path.join(out_dir, f"meme-{nmotifs}")
    meme_out = os.path.join(parent_meme_out, f"meme_seed-{seed}")
    os.makedirs(meme_out, exist_ok=True)
    pwm_file = os.path.join(meme_out, "meme.txt")

    logger.log_message("Getting pwm file...")
    if os.path.exists(pwm_file) and os.path.getsize(pwm_file) > 0:
        logger.log_message(f"Using existing MEME results in {pwm_file}...")
    else:
        cmd = [
            meme_path,
            pos_input_fasta,
            "-oc", meme_out,
            "-dna",
            "-nmotifs", str(nmotifs),
            "-seed", str(seed),
            # "-evt", str(evt),
            "-minw", "8",
            "-maxw", "65",
            "-revcomp",
            "-neg", neg_input_fasta,
            "-objfun", "de"  # Discriminative Mode - when there is a control set
        ]
        logger.log_message(f"Running MEME\nCommand: {cmd}")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            logger.log_message(f"Error running MEME: {e}.")
    return pwm_file, time.time()-start_time  # Output PWM file

def run_fimo(pwm_file, fasta_file, fimo_out, pval_thresh, fimo_path="fimo"):
    start_time = time.time()
    """Run FIMO with specified PWM and threshold."""
    tsv_file = os.path.join(fimo_out, "fimo.tsv")
    if os.path.exists(tsv_file) and os.path.getsize(tsv_file) > 0:
        logger.log_message(f"Using existing FIMO results for threshold {pval_thresh} in {tsv_file}...")
    else:
        logger.log_message(f"Running FIMO for threshold {pval_thresh}...")
        os.makedirs(fimo_out, exist_ok=True)
        cmd = [
            fimo_path,
            "--thresh", str(pval_thresh),
            "--qv-thresh",
            "--oc", fimo_out,
            pwm_file,
            fasta_file
        ]
        logger.log_message(f"Command: {cmd}")
        subprocess.run(cmd, check=True)
        subprocess.run(cmd, check=True)
    return tsv_file, time.time()-start_time

def evaluate_fimo(df_ground, fimo_output):
    """Evaluate predictions (placeholder, depends on your metric)."""
    try:
        df_predicted = pd.read_csv(fimo_output, sep='\t', comment="#")
        logger.log_message(f"length of sequences in predicted: {len(df_predicted)}")
        df_predicted = df_predicted.drop_duplicates(subset="sequence_name")
        df_predicted["peak_uid"] = df_predicted["sequence_name"].str.split("_").str[0].astype(int)
        df_predicted["label"] = df_predicted["sequence_name"].str.split("_").str[1].astype(int)

        logger.log_message(f"length of unique sequences in predicted: {len(df_predicted)}")

    except Exception as e:
        raise ValueError(f"Error reading predicted file: {e}.")

    df = df_ground.merge(df_predicted, how='left', on=["peak_uid", "label"])
    logger.log_message(f"length of merged df: {len(df)}")
    # indicator
    df['predicted_exists'] = df['sequence_name'].notna()  # True if a prediction exists

    # TP (in both and label=1)
    true_positive = ((df['predicted_exists']) & (df['label'] == 1)).sum()
    # FN (in ground label=1 but not in predicted)
    false_negative = ((~df['predicted_exists']) & (df['label'] == 1)).sum()
    # TN (in ground label=0 but not in predicted)
    true_negative = ((~df['predicted_exists']) & (df['label'] == 0)).sum()
    # FP (in ground label=0 but in predicted)
    false_positive = ((df['predicted_exists']) & (df['label'] == 0)).sum()

    confusion_matrix = {
        'True Positive': true_positive,
        'False Negative': false_negative,
        'True Negative': true_negative,
        'False Positive': false_positive
    }
    logger.log_message(confusion_matrix)

    # MCC
    mcc_numerator = (true_positive * true_negative) - (false_positive * false_negative)
    mcc_denominator = np.sqrt(
        (true_positive + false_positive) *
        (true_positive + false_negative) *
        (true_negative + false_positive) *
        (true_negative + false_negative)
    )
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0.0

    # F1 Score
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Accuracy
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    # AUROC and AUPRC
    auroc = 0
    auprc = 0
    if len(df_predicted) > 0:
        y_true = df['label']

        # for sequences with no match, score is NaN
        min_score = df['score'].min()
        fill_value = min_score - 1 # guaranteed to be smaller than that lowest score.
        y_scores = df['score'].fillna(fill_value)

        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)

    metrics = {
        'test_accuracy': round(accuracy, 2),
        'test_f1': round(f1_score, 2),
        'test_mcc': round(mcc, 2),
        'test_auroc': round(auroc, 2),
        'test_auprc': round(auprc, 2)
    }

    logger.log_message(metrics)
    return metrics


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.motif_config_file, args.split_config_file)

    logger, output_dir = setup_logger_and_out(config)

    dfs_train, dfs_val, df_train, df_holdout_val, df_test = DataSplit.get_splits(logger, config.split_config, label='label')
    logger.log_message("***************************************************************************")
    """Phase 1: Get best q-val parameter based on cross-validation."""
    logger.log_message("PHASE 1: Get best q-val parameter based on cross-validation.")
    if len(config.qval_thresholds) > 1:
        fold_scores = {pval: [] for pval in config.qval_thresholds}
        for i in range(1, len(dfs_train)+1):
            logger.log_message(f"*************   Fold-{i}   ********************")
            ## make fasta files and run meme to get PWM file for current fold
            current_out_dir = os.path.join(output_dir, f"Fold-{i}")
            os.makedirs(current_out_dir, exist_ok=True)

            df_train = dfs_train[i]
            df_val = dfs_val[i]

            # make fasta files
            train_pos_fasta = os.path.join(current_out_dir, 'train_dataset-pos.fasta')
            write_fasta(df_train[df_train['label'] == 1], train_pos_fasta)

            train_neg_fasta = os.path.join(current_out_dir, 'train_dataset-neg.fasta')
            write_fasta(df_train[df_train['label'] == 0], train_neg_fasta)

            val_fasta = os.path.join(current_out_dir, 'val_dataset.fasta')
            write_fasta(df_val, val_fasta)

            # run meme on train to get PWM
            pwm_file, train_time = run_meme(train_pos_fasta, train_neg_fasta, current_out_dir, nmotifs=config.nmotifs)

            ## run fimo across different thresholds for pwm and evaluate resulting motifs based on true motifs
            logger.log_message("Running FIMO based on different thresholds")
            for qval in config.qval_thresholds:
                logger.log_message(f"****** Qval: {qval} *******")
                fimo_out = os.path.join(current_out_dir, f"fimo_q{qval}")
                fimo_output, test_time = run_fimo(pwm_file, val_fasta, fimo_out, qval)
                # evaluate resulting motifs based on true motifs
                score = evaluate_fimo(df_val, fimo_output)
                fold_scores[qval].append(score)
                logger.log_message(f"fold_scores[{qval}]:\t{score}")

        logger.log_message(f"fold_scores: {fold_scores}\nGetting best threshold across folds...")
        best_qval = max(fold_scores, key=lambda k: np.nanmean([d['test_f1'] for d in fold_scores[k]]))
        logger.log_message("Q-val for FIMO scanning with highest mean F1 Score across folds:", best_qval)
    else:
        best_qval = config.qval_thresholds[0]
        logger.log_message(f"Only one Q-val, no need to get the best one across folds.\nUsing {best_qval}")

    """Phase 2: Get PWM on all data using different seeds and evaluate on test set."""
    logger.log_message("***************************************************************************")
    logger.log_message("PHASE 2: Get PWM on all data using different seeds and evaluate on test set.")

    whole_train_fasta_pos = os.path.join(output_dir, "whole_train_pos.fasta")
    whole_train_fasta_neg = os.path.join(output_dir, "whole_train_neg.fasta")

    write_fasta(df_train[df_train['label'] == 1], whole_train_fasta_pos)
    write_fasta(df_train[df_train['label'] == 0], whole_train_fasta_neg)

    results = []
    for seed in config.seeds:
        logger.log_message(f"*************   SEED-{seed}   ********************")
        # run meme on whole train data with current seed to get final pwm
        final_pwm_file, train_time = run_meme(whole_train_fasta_pos, whole_train_fasta_neg, output_dir, nmotifs=config.nmotifs, seed=seed)

        # make test fasta
        test_fasta = os.path.join(output_dir, 'test_dataset.fasta')
        write_fasta(df_test, test_fasta)

        ## run fimo on test data using the final pwm
        parent_fimo_out = os.path.join(output_dir, "final_fimo_test_time")
        fimo_out = os.path.join(parent_fimo_out, f"fimo_seed-{seed}")
        fimo_output, test_time = run_fimo(final_pwm_file, test_fasta, fimo_out, best_qval)

        result = evaluate_fimo(df_test, fimo_output)

        result["train_time(s)"] = train_time
        result["test_time(s)"] = test_time
        result["seed"] = seed
        result["fimo p_val"] = best_qval

        results.append(result)

    df_results = pd.DataFrame(results)
    results_csv_file = os.path.join(output_dir, 'prediction_results_test_time.csv')
    df_results.to_csv(results_csv_file, index=False)
    logger.log_message(f"Results on test set based on FIMO Q-val {best_qval} on different seeds has been saved at: {results_csv_file}", use_time=True)