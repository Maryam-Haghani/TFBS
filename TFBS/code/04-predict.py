import pandas as pd
import os
import argparse
from pathlib import Path

from logger import CustomLogger
from utils import load_config, get_models
from model_utils import init_model_and_tokenizer, get_ds, set_device, load_model
from train_test import Train_Test

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

    if getattr(config, "model_version", None):
        #  model_version should be a directory in `saved_model_dir`
        if not config.model_version in p.parts:
            raise ValueError(f'Given model version ({config.model_version})'
                             f'not found in saved_model_dir ({config.saved_model_dir})!')

def setup_logger(output_dir):
    """Initialize logger with the given output directory."""
    logger = CustomLogger(__name__, log_directory=output_dir, log_file="log")
    logger.log_message(f"Output will save at: {output_dir}")
    return logger

def load_current_model(config, model, base_sd, model_name):
    """
    Loads the base state dictionary, then updates the model with a saved state dictionary from the specified directory.
    """
    # reset model to the initial model
    model.load_state_dict(base_sd)
    logger.log_message(f"Loading model '{config.model.saved_model_dir}/{model_name}'...")
    state_dict = load_model(config.model.saved_model_dir, model_name, config.device)
    model.load_state_dict(state_dict)

def sliding_window_sequence(sequence, window_size):
    """
    Generate sliding windows of size `window_size` from a given sequence.
    """
    return [sequence[i:i + window_size] for i in range(0, len(sequence) - window_size + 1)]

def load_data(config, logger):
    data = pd.read_csv(config.input_dir)
    return data, None


def get_output_dir(config):
    input_name = Path(config.input_dir).stem
    parent_model_dir = os.sep.join(config.model.saved_model_dir.split(os.sep)[:-1])
    output_dir = os.path.join(parent_model_dir, 'predictions', input_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config_file)
    config.num_saliency_samples = getattr(config, "num_saliency_samples", 0) # set to 0 if None
    config.saliency_method = getattr(config, "saliency_method", 'smooth') # set to 'smooth' if None

    validate_saved_model_dir(config.model)
    model_dict = get_models(config.model.saved_model_dir)

    output_dir = get_output_dir(config)

    logger = setup_logger(output_dir)
    logger.log_message(f"Configuration loaded: {config}")
    logger.log_message(f'Using saved model(s) in {config.model.saved_model_dir} for prediction...')

    config.device = set_device(config.device)
    logger.log_message("Using device:", config.device)

    model, tokenizer = init_model_and_tokenizer(logger, config.model, config.device)
    base_sd = model.state_dict()

    tt = Train_Test(logger, config.model.max_length, config.device, config.eval_batch_size)

    data, abs_start = load_data(config, logger)
    window_size = getattr(config, 'window_size', None)
    stride = getattr(config, 'stride', None)
    ds = get_ds(config.model, tokenizer, data, window_size=window_size, stride=stride)
    logger.log_message(f"len of ds: {len(ds)}")

    # get predictions for all models######
    results = []

    for freeze_layer, files in model_dict.items():
        logger.log_message(f"******************************** Directory: {freeze_layer} ********************************")
        for model_name in files:
            model_dir = os.path.join(freeze_layer, model_name)
            logger.log_message(f"********* Model: {model_dir} **********")
            load_current_model(config, model, base_sd, model_dir)

            logger.log_message(f'Getting prediction based on {model_dir}')
            name, _ext = os.path.splitext(model_name)
            current_output_dir = os.path.join(output_dir, freeze_layer)
            os.makedirs(current_output_dir, exist_ok=True)
            test_accuracy, test_auroc, test_auprc, test_f1, test_mcc, test_time \
                = tt.predict(model, ds, current_output_dir, name,
                             num_saliency_samples=config.num_saliency_samples,
                             saliency_method=config.saliency_method)

            results.append({
                'freeze layer': freeze_layer,
                'model_name': model_name,
                'test_accuracy': round(test_accuracy, 2),
                'test_f1': round(test_f1, 2),
                'test_mcc': round(test_mcc, 2),
                'test_auroc': round(test_auroc, 2),
                'test_auprc': round(test_auprc, 2),
                'test_time(s)': test_time
            })

    # save results to csv
    df_results = pd.DataFrame(results)
    results_csv_file = os.path.join(output_dir, "prediction_results.csv")
    df_results.to_csv(results_csv_file, index=False)
    logger.log_message(f"Results saved to {results_csv_file}", use_time=True)
