import torch
import pandas as pd
import os
import argparse
from pathlib import Path

from logger import CustomLogger
from utils import load_config

from datasets.deepbind_dataset import DeepBindDataset
from datasets.hyenadna_dataset import HyenaDNA_Dataset
from datasets.dnabert2_dataset import DNABERT2_dataset
from datasets.agro_nt_dataset import AgroNT_Dataset

from train_test import Train_Test

from models.hyena_dna import HyenaDNAModel
from models.dna_bert_2 import DNABERT2
from models.deep_bind import DeepBind
from models.BERT_TFBS.bert_tfbs import BERT_TFBS
from models.agro_nt import AgroNTModel

# python 04-predict.py --config_file [config_path]
# python 04-predict.py --config_file ../configs/predict/HeynaDNA-config.yml


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file.")
    return parser.parse_args()

def validate(config):
    p = Path(config.saved_model_dir)
    #  model_name should be a directory in `saved_model_dir`
    if not config.model_name in p.parts:
        raise ValueError(f'Given model name ({config.model_name})'
                         f'is not in saved_model_dir ({config.saved_model_dir})!')

    if config.model_version:
        #  model_version should be a directory in `saved_model_dir`
        if not config.model_version in p.parts:
            raise ValueError(f'Given model version ({config.model_version})'
                             f'is not in saved_model_dir ({config.saved_model_dir})!')

if __name__ == "__main__":

    args = parse_arguments()
    config = load_config(args.config_file)

    validate(config.model)

    output_dir = os.path.join(config.model.saved_model_dir, Path(config.dataset_dir).stem, 'predictions')
    os.makedirs(output_dir, exist_ok=True)

    logger = CustomLogger(__name__, log_directory=output_dir, log_file=f'log')
    logger.log_message(f"Configuration loaded: {config}")
    logger.log_message(f"Output will save at: {output_dir}")

    config.device = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    logger.log_message("Using device:", config.device)
    logger.log_message(f'Using saved model(s) in {config.model.saved_model_dir} for prediction...')

    df_test = pd.read_csv(config.dataset_dir)

    tt = Train_Test(logger, config.device, config.model.saved_model_dir, config.eval_batch_size)

    if config.model.model_name == "HyenaDNA":
        tokenizer = (HyenaDNAModel(logger, pretrained_model_name=config.model.model_version, device=config.device)
                     .get_tokenizer(config.model.max_length))
        ds_test = HyenaDNA_Dataset(tokenizer, df_test, config.model.max_length, config.model.use_padding)
        tt.model = (HyenaDNAModel(logger, pretrained_model_name=config.model.model_version, device=config.device)
                    .load_pretrained_model())
    elif config.model.model_name == 'DeepBind':
        ds_test = DeepBindDataset(df_test, config.model.max_length, config.model.kernel_length)
        tt.model = DeepBind(config.model.kernel_length)
    elif config.model.model_name == 'BERT-TFBS':
        tokenizer = (BERT_TFBS(config.model.max_length).get_tokenizer())
        tt.model = BERT_TFBS(config.model.max_length)
        ds_test = DNABERT2_dataset(tokenizer, df_test, config.model.max_length)
    elif config.model.model_name == 'DNABERT-2':
        tokenizer = (DNABERT2().get_tokenizer())
        ds_test = DNABERT2_dataset(tokenizer, df_test, config.model.max_length)
        tt.model = DNABERT2()
    elif config.model.model_name == "AgroNT":
        tokenizer = AgroNTModel(logger, device=config.device).get_tokenizer()
        ds_test = AgroNT_Dataset(tokenizer, df_test, config.model.max_length)
        tt.model = (AgroNTModel(logger, device=config.device)
                    .load_pretrained_model(config.model.finetune_type))
    else:
        raise ValueError(f'Given model name ({config.model.model_name}) is not valid!')

    # get predictions for all folds
    results = []
    for model_name in os.listdir(config.model.saved_model_dir):
        if model_name.endswith(".pt"):
            logger.log_message(f'********************************\n'
                               f'Getting prediction based on {model_name}')
            model = tt.load(model_name)

            name, _ext = os.path.splitext(model_name)
            test_accuracy, test_auroc, test_auprc, test_f1, test_mcc, test_time \
                = tt.test(ds_test, name, output_dir)
    
            results.append({
                'model_name': model_name,
                'test_accuracy': round(test_accuracy, 2),
                'test_f1': round(test_f1, 2),
                'test_mcc': round(test_mcc, 2),
                'test_auroc': round(test_auroc, 2),
                'test_auprc': round(test_auprc, 2),
                'test_time(s)': test_time
            })

    df_results = pd.DataFrame(results)
    results_csv_file = os.path.join(output_dir, 'prediction_results.csv')
    df_results.to_csv(results_csv_file, index=False)
    logger.log_message(f"Results saved to {results_csv_file}", use_time=True)
