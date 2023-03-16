import argparse, json, torch
import numpy as np
import os

from nli_mixed_models.trainers.nli_trainer import (
    UnitRandomInterceptsTrainer,
    UnitRandomSlopesTrainer,
)
from nli_mixed_models.eval.nli_eval import UnitEval
from scripts.setup_logging import setup_logging
from scripts.training_utils import (
    parameter_grid,
    load_unit_data,
    load_neg_raising,
    load_difficulty,
    load_intensionality,
    save_model_with_args,
)

LOG = setup_logging()


def main(args):

    # Load parameters from the supplied parameters file
    with open(args.parameters) as f:
        params = json.load(f)

    settings = params["settings"]
    checkpoints = params["checkpoints"]
    save_checkpoints = checkpoints["save_ckpts"]
    checkpoint_dir = checkpoints["ckpt_dir"]
    checkpoint_file_name = checkpoints["ckpt_file_name"]

    os.makedirs(checkpoint_dir)

    for hyperparams in parameter_grid(params["hyper"]):
        for trainparams in parameter_grid(params["training"]):

            # Load data
            if "dataset" in settings and settings["dataset"] == "difficulty":
                templatic = "templatic" in settings and settings["templatic"]
                LOG.info(f"Loading Difficulty data... (templatic: {templatic})")
                data = load_difficulty(templatized=templatic)
            elif "dataset" in settings and settings["dataset"] == "intensionality":
                templatic = "templatic" in settings and settings["templatic"]
                LOG.info(f"Loading Intensionality data... (templatic: {templatic})")
                data = load_intensionality(templatized=templatic)
            elif "dataset" in settings and settings["dataset"] == "negraising":
                LOG.info("Loading MegaNegRaising data...")
                data = load_neg_raising()
            elif "dataset" in settings:
                templatic = "templatic" in settings and settings["templatic"]
                LOG.info("Loading unit data...")
                data = load_unit_data(settings["dataset"], templatized=templatic)
            else:
                LOG.info("Loading MegaNegRaising data...")
                data = load_neg_raising()
            hyperparams["n_participants"] = data.participant.unique().shape[0]
            LOG.info("...Complete.")

            # Log the device being used
            device = hyperparams["device"]
            LOG.info(f"Using device {device}")

            # Save the DataFrame used for training in case
            # we need to recover it later
            data_file_name = (
                checkpoint_dir
                + "/"
                + "-".join([checkpoint_file_name, "data"])
                + ".csv"
            )
            data.to_csv(data_file_name, index=False)

            # Dump current training settings
            LOG.info(
                "Initializing unit NLI model with the following hyperparameters:"
            )
            LOG.info(json.dumps(hyperparams, indent=4))
            LOG.info("Using the following training parameters:")
            LOG.info(json.dumps(trainparams, indent=4))
            LOG.info("And the following settings:")
            LOG.info(json.dumps(settings, indent=4))

            # Only the random intercepts models have been modified to handle
            # the standard setting in addition to the extended one.
            if (
                hyperparams["setting"] == "extended"
                and settings["use_random_slopes"]
            ):
                unli_trainer = UnitRandomSlopesTrainer(**hyperparams)
            else:
                unli_trainer = UnitRandomInterceptsTrainer(**hyperparams)

            train_data = data

            # Fit the model on the train folds
            unit_model = unli_trainer.fit(train_data=train_data, **trainparams)
            LOG.info("Finished training.")

            # Save the model
            if save_checkpoints:
                LOG.info("Saving model...")
                model_name = checkpoint_file_name + ".pt"
                save_model_with_args(
                    params, unit_model, hyperparams, checkpoint_dir, model_name
                )
                LOG.info("Model saved.")

            # Evaluate the model
            unli_eval = UnitEval(unit_model, "a", device=device)
            unli_eval_b = UnitEval(unit_model, "b", device=device)

            LOG.info("Evaluating subtask a")
            (
                loss_mean,
                fixed_loss_mean,
                random_loss_mean,
                error_mean,
                best_mean,
                spearman,
                best_spearman,
                _
            ) = unli_eval.eval(train_data, trainparams["batch_size"])

            LOG.info(f"Test results for subtask a")
            LOG.info(f"Mean fixed loss:     {fixed_loss_mean}")
            LOG.info(f"Mean error:       {error_mean}")
            LOG.info(f"Prop. best possible: {best_mean}")
            LOG.info(f"Mean spearman:       {spearman}")
            LOG.info(f"Best spearman:       {best_spearman}")
            LOG.info(f"Prop. best spearman: {spearman / best_spearman}")

            LOG.info("Evaluating subtask b")
            (
                loss_mean,
                fixed_loss_mean,
                random_loss_mean,
                error_mean,
                best_mean,
                spearman,
                best_spearman,
                _
            ) = unli_eval_b.eval(train_data, trainparams["batch_size"])

            LOG.info(f"Test results for subtask b")
            LOG.info(f"Mean fixed loss:     {fixed_loss_mean}")
            LOG.info(f"Mean error:          {error_mean}")
            LOG.info(f"Prop. best possible: {best_mean}")
            LOG.info(f"Mean spearman:       {spearman}")
            LOG.info(f"Best spearman:       {best_spearman}")
            LOG.info(f"Prop. best spearman: {spearman / best_spearman}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a unit NLI mixed effects model")
    parser.add_argument(
        "--parameters",
        type=str,
        default="scripts/unit/train_unit_nli.json",
        help="Path to a JSON containing parameters to sweep",
    )
    args = parser.parse_args()
    main(args)
