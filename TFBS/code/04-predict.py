import pandas as pd
import os
import argparse
from pathlib import Path

from logger import CustomLogger
from utils import load_config, get_models
from model_utils import init_model_and_tokenizer, get_ds, set_device, load_model
from train_test import Train_Test

# python 04-predict.py --config_file [config_path]
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file.")
    return parser.parse_args()

def validate_saved_model_dir(config):
    """Ensure that the model and version exist in the saved_model_dir."""
    p = Path(config.saved_model_dir)
    if not config.model_name in p.parts:
        raise ValueError(f'Given model name ({config.model_name})'
                         f'not found in saved_model_dir ({config.saved_model_dir})!')

    if config.model_version:
        #  model_version should be a directory in `saved_model_dir`
        if not config.model_version in p.parts:
            raise ValueError(f'Given model version ({config.model_version})'
                             f'not found in saved_model_dir ({config.saved_model_dir})!')

def setup_logger(output_dir):
    """Initialize logger with the given output directory."""
    logger = CustomLogger(__name__, log_directory=output_dir, log_file="log")
    logger.log_message(f"Output will save at: {output_dir}")
    return logger

if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config_file)

    validate_saved_model_dir(config.model)
    model_files = get_models(config.model.saved_model_dir)

    output_dir = os.path.join(config.model.saved_model_dir, 'Predictions', Path(config.dataset_dir).stem)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(output_dir)

    logger.log_message(f'Using saved model(s) in {config.model.saved_model_dir} for prediction...')

    config.device = set_device(config.device)
    logger.log_message("Using device:", config.device)

    df = pd.read_csv(config.dataset_dir)

    tt = Train_Test(logger, config.device, config.eval_batch_size)

    model, tokenizer = init_model_and_tokenizer(logger, config.model, config.device)
    base_sd = model.state_dict()
    ds_test = get_ds(config.model, tokenizer, df)

    # get predictions for all folds
    results = []
    for model_name in model_files:
        logger.log_message(f'********************************\n'
                           f'Getting prediction based on {model_name}')
        # reset model to the initial model
        model.load_state_dict(base_sd)

        logger.log_message(f"Loading model '{config.model.saved_model_dir}/{model_name}'...")
        state_dict = load_model(config.model.saved_model_dir, model_name, config.device)
        model.load_state_dict(state_dict)
        model.to(config.device)

        name, _ext = os.path.splitext(model_name)
        tt.model = model
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