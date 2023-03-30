import os
import argparse
import torch
import json
import numpy as np
import pandas as pd

from scripts.training_utils import load_model_with_missing_hyperparams
from scripts.setup_logging import setup_logging

from nli_mixed_models.modules.nli_random_intercepts import (
    UnitRandomIntercepts,
    CategoricalRandomIntercepts,
)
from nli_mixed_models.modules.nli_random_slopes import (
    UnitRandomSlopes,
    CategoricalRandomSlopes,
)
from nli_mixed_models.predict.nli_predict import UnitPredict, CategoricalPredict
from glob import glob

CAT_OUTPUT_DIM = 3
UNIT_OUTPUT_DIM = 1

NUM_FOLDS = 5
BATCH_SIZE = 128
RANDOM_SEED = 42

LOG = setup_logging()

# set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)



def load_model(ckpt_path, device):
    """Loads a saved model"""

    # Determine the type of model
    if "categorical" in ckpt_path:
        output_dim = CAT_OUTPUT_DIM
        if "random_intercepts" in ckpt_path or "random-intercepts" in ckpt_path or "standard" in ckpt_path or "fixed" in ckpt_path:
            model_cls = CategoricalRandomIntercepts
        else:
            model_cls = CategoricalRandomSlopes
    elif "unit" in ckpt_path:
        output_dim = UNIT_OUTPUT_DIM
        if "random_intercepts" in ckpt_path or "random-intercepts" in ckpt_path or "standard" in ckpt_path or "fixed" in ckpt_path:
            model_cls = UnitRandomIntercepts
        else:
            model_cls = UnitRandomSlopes
    else:
        raise ValueError(f"Unknown model type!")

    missing_hyperparams = {"output_dim": output_dim}

    # Load the model
    model, hyper_params = load_model_with_missing_hyperparams(model_cls, ckpt_path, missing_hyperparams)
    model = model.to(torch.device(device))

    return model, hyper_params


def main(args):
    ckpt_path = args.model_ckpt_path
    model_type = "categorical" if "categorical" in ckpt_path else "unit"
    data_file_name = args.dataset_filename
    if '.tsv' in data_file_name:
      data = pd.read_csv(data_file_name, sep='\t')
    else:
      data = pd.read_csv(data_file_name)

    ckpt_file_name = ckpt_path + "/" + model_type + ".pt"
    model, hyperparams = load_model(ckpt_file_name, args.device)
    LOG.info(
        f"Evaluating {model.__class__.__name__} model with the following hyperparameters:"
    )
    LOG.info(json.dumps(hyperparams, indent=4))
 
    subtask = "b" if 'participant' not in data.columns else "a"
    LOG.info(f"Evaluating subtask {subtask}")
    if model_type == "categorical":
        nli_pred = CategoricalPredict(model, subtask, device=args.device)
        pred_mean, _ = nli_pred.predict(data, BATCH_SIZE)
        data['target_pred_mean'] = pred_mean
    else:
        nli_pred = UnitPredict(model, subtask, device=args.device)
        pred_mean, pred_precision = nli_pred.predict(data, BATCH_SIZE)
        data['target_pred_mean'] = pred_mean
        data['target_pred_precision'] = pred_precision

    output_file_name = args.output_filename

    if output_file_name:
      if '.tsv' in output_file_name:
        data.to_csv(output_file_name, sep='\t', index=False)
      else:
        data.to_csv(output_file_name, index=False)
    else:
      if '.tsv' in data_file_name:
        new_file_name = data_file_name.split('.tsv')[0] + '_pred.tsv'
        data.to_csv(new_file_name, sep='\t', index=False)
      else:
        new_file_name = data_file_name.split('.csv')[0] + '_pred.csv'
        data.to_csv(new_file_name, index=False)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_ckpt_path",
        help="the directory containing the "
        "model checkpoints, as well as the CSV of splits",
    )
    parser.add_argument(
      "dataset_filename",
      help="the filename containing the data to generate predictions for"
    )
    parser.add_argument(
      "--output_filename", help="the filename to output the predictions"
    )
    parser.add_argument(
        "--device", default="cpu", help="the device on which to run the script"
    )
    args = parser.parse_args()
    main(args)
