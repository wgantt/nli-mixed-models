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
from nli_mixed_models.eval.nli_eval import UnitEval, CategoricalEval
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


def dump_random_effects(ckpt_base_dir, output_dir):
    splits = ["random", "predicate", "syntax", "annotator"]
    model_type = ["random_intercepts", "random_slopes"]

    cat_base_dir = os.path.join(ckpt_base_dir, "categorical")
    unit_base_dir = os.path.join(ckpt_base_dir, "unit")

    for m in model_type:
        for s in splits:

            cat_ckpts = glob(os.path.join(cat_base_dir, m, s, "*.pt"))
            unit_ckpts = glob(os.path.join(unit_base_dir, m, s, "*.pt"))

            for i, cat_ckpt in enumerate(cat_ckpts):
                fname = "-".join(["categorical", m, s, str(i)]) + ".csv"
                fname = os.path.join(output_dir, fname)
                cat_model = torch.load(cat_ckpt)
                if m == "random_intercepts":
                    random_effects = cat_model["state_dict"]["random_effects"]
                    df = pd.DataFrame(random_effects).astype("float")
                    df.rename(lambda x: "bias_dim-" + str(x), axis=1, inplace=True)
                else:
                    n_participants = cat_model["curr_hyper"]["n_participants"]
                    hidden_dim = cat_model["curr_hyper"]["hidden_dim"]
                    weight_cols = [
                        "-".join(["predictor", "weight", str(i), "dim", str(j)])
                        for i in range(hidden_dim)
                        for j in range(3)
                    ]
                    bias_cols = [
                        "-".join(["predictor", "bias", "dim", str(i)]) for i in range(3)
                    ]
                    cols = weight_cols + bias_cols
                    for i in range(n_participants):
                        weight_param_name = ".".join(
                            ["predictor_heads", str(i), "weight"]
                        )
                        bias_param_name = ".".join(["predictor_heads", str(i), "bias"])
                        curr_weight = cat_model["state_dict"][
                            weight_param_name
                        ].T.flatten()
                        curr_bias = cat_model["state_dict"][bias_param_name]
                        if i > 0:
                            new_row = torch.cat([curr_weight, curr_bias]).unsqueeze(0)
                            all_params = torch.cat([all_params, new_row])
                        else:
                            all_params = torch.cat([curr_weight, curr_bias]).unsqueeze(
                                0
                            )
                    df = pd.DataFrame(all_params, columns=cols).astype("float")
                df.to_csv(fname, index=False)
            for i, unit_ckpt in enumerate(unit_ckpts):
                fname = "-".join(["unit", m, s, str(i)]) + ".csv"
                fname = os.path.join(output_dir, fname)
                unit_model = torch.load(unit_ckpt)
                if m == "random_intercepts":
                    random_effects = unit_model["state_dict"]["random_effects"]
                    df = pd.DataFrame(
                        random_effects, columns=["mean", "precision"]
                    ).astype("float")
                else:
                    n_participants = unit_model["curr_hyper"]["n_participants"]
                    hidden_dim = unit_model["curr_hyper"]["hidden_dim"]
                    cols = [
                        "-".join(["predictor", "weight", str(i)])
                        for i in range(hidden_dim)
                    ]
                    cols = cols + ["predictor-bias"]
                    for i in range(n_participants):
                        weight_param_name = ".".join(
                            ["predictor_heads", str(i), "weight"]
                        )
                        bias_param_name = ".".join(["predictor_heads", str(i), "bias"])
                        curr_weight = unit_model["state_dict"][weight_param_name]
                        curr_bias = unit_model["state_dict"][bias_param_name]
                        if i > 0:
                            new_row = torch.cat(
                                [curr_weight, curr_bias.unsqueeze(0)], axis=1
                            )
                            all_params = torch.cat([all_params, new_row])
                        else:
                            all_params = torch.cat(
                                [curr_weight, curr_bias.unsqueeze(0)], axis=1
                            )
                    df = pd.DataFrame(all_params, columns=cols).astype("float")
                df.to_csv(fname, index=False)


def load_old_style_model(ckpt_path, device):
    """Loads a model that was saved in a now-obsolete format

	Previously, our models did not have the mean and covariance of the
	random effects saved as instance variables. This function loads
	such a model, adding those instance variables along with it.
	"""

    # Determine the type of model
    if "categorical" in ckpt_path:
        if "random_intercepts" in ckpt_path or "standard" in ckpt_path:
            model_cls = CategoricalRandomIntercepts
        else:
            model_cls = CategoricalRandomSlopes
    elif "unit" in ckpt_path:
        if "random_intercepts" in ckpt_path or "standard" in ckpt_path:
            model_cls = UnitRandomIntercepts
        else:
            model_cls = UnitRandomSlopes
    else:
        raise ValueError(f"Unknown model type!")

    # Determine the model output dimension
    output_dim = CAT_OUTPUT_DIM if "categorical" in ckpt_path else UNIT_OUTPUT_DIM

    # Required model constructor parameters that are missing for some
    # reason from the models I saved.
    missing_hyperparams = {
        "hidden_dim": 128,
        "output_dim": output_dim,
        "use_sampling": False,
        "n_samples": 10000,
    }

    # Load the model with the missing hyperparameters
    model, hyper_params = load_model_with_missing_hyperparams(
        model_cls, ckpt_path, missing_hyperparams
    )

    # Estimate the mean and the covariance from the saved random effects
    model = model.to(torch.device(device))
    if "random_intercepts" in ckpt_path:
        random = model._random_effects()
        model.mean = random.mean(0).to(torch.device(device))
        model.cov = (
            torch.matmul(torch.transpose(random, 1, 0), random)
            / (model.n_participants - 1)
        ).to(torch.device(device))
    elif "random_slopes" in ckpt_path:
        if "categorical" in ckpt_path:
            combined = model._random_effects()
        else:
            weights, variance = model._random_effects()
            combined = torch.cat([weights, variance.unsqueeze(1)], dim=1)
        model.mean = combined.mean(0).to(torch.device(device))
        model.cov = (
            torch.matmul(torch.transpose(combined, 1, 0), combined)
            / (model.n_participants - 1)
        ).to(torch.device(device))

    return model, hyper_params


def main(args):
    ckpt_path = args.model_ckpt_path
    model_type = "categorical" if "categorical" in ckpt_path else "unit"
    data_file_name = (
        ckpt_path
        + "/"
        + "-".join([model_type, "partition", args.partition, "data"])
        + ".csv"
    )
    data = pd.read_csv(data_file_name)

    fixed_loss_all = []
    metric_all = []
    best_all = []
    worst_all = []

    spearman_all = []
    best_spearman_all = []
    prop_best_spearman_all = []
    score_mod_all = []

    for test_fold in range(NUM_FOLDS):
        ckpt_file_name = (
            ckpt_path
            + "/"
            + "-".join(
                [model_type, "fold", str(test_fold), "partition", args.partition]
            )
            + ".pt"
        )
        model, hyperparams = load_old_style_model(ckpt_file_name, args.device)
        LOG.info(
            f"Evaluating {model.__class__.__name__} model on the {args.partition} partition with the following hyperparameters:"
        )
        LOG.info(json.dumps(hyperparams, indent=4))
        test_data = data[data.fold == test_fold]
        subtask = "b" if args.partition == "annotator" else "a"
        LOG.info(f"Evaluating subtask {subtask}")
        if model_type == "categorical":
            nli_eval = CategoricalEval(model, subtask, device=args.device)
        else:
            nli_eval = UnitEval(model, subtask, device=args.device)
        (
            loss_mean,
            fixed_loss_mean,
            random_loss_mean,
            metric_mean,
            best_mean,
            predicted_spearman,
            best_spearman,
            worst_mean,
        ) = nli_eval.eval(test_data, BATCH_SIZE)
        fixed_loss_all.append(fixed_loss_mean)
        metric_all.append(metric_mean)
        best_all.append(best_mean)
        worst_all.append(worst_mean)
        spearman_all.append(predicted_spearman)
        best_spearman_all.append(best_spearman)
        prop_best_spearman_all.append(predicted_spearman / best_spearman)
        score_mod_all.append((metric_mean - worst_mean) / (best_mean - worst_mean))
        LOG.info(f"Test results for fold {test_fold}, subtask {subtask}")
        LOG.info(f"Mean fixed loss:     {fixed_loss_mean}")
        LOG.info(f"Mean metric:         {metric_mean}")
        LOG.info(f"Best metric: {best_mean}")
        LOG.info(f"Worst metric: {worst_mean}")
        LOG.info(f"Score mod: {score_mod_all[-1]}")
        LOG.info(f"Predicted Spearman:       {predicted_spearman}")
        LOG.info(f"Best Spearman:       {best_spearman}")
        LOG.info(f"Prop. best Spearman: {predicted_spearman / best_spearman}")

    LOG.info("Finished k-fold cross evaluation.")
    LOG.info(f"Subtask {subtask} results:")
    LOG.info(f"Mean fixed loss:           {np.round(np.mean(fixed_loss_all), 4)}")
    LOG.info(f"Mean metric:       {np.round(np.mean(metric_all), 4)}")
    LOG.info(f"Mean best metric:  {np.round(np.mean(best_all), 4)}")
    LOG.info(f"Mean worst metric: {np.round(np.mean(worst_all), 4)}")
    LOG.info(f"Mean score mod: {np.round(np.mean(score_mod_all), 4)}")
    # LOG.info(f"Prop. best possible: {np.round(np.mean(best_all), 4)}")
    LOG.info(f"Mean predicted Spearman:       {np.round(np.mean(spearman_all), 4)}")
    LOG.info(
        f"Mean best Spearman:            {np.round(np.mean(best_spearman_all), 4)}"
    )
    LOG.info(
        f"Mean prop. best spearman: {np.round(np.mean(prop_best_spearman_all), 4)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_ckpt_path",
        help="the directory containing the "
        "model checkpoints, as well as the CSV of splits",
    )
    parser.add_argument(
        "partition", help="the type of partition the model was trained on."
    )
    parser.add_argument(
        "--device", default="cpu", help="the device on which to run the script"
    )
    args = parser.parse_args()
    main(args)
