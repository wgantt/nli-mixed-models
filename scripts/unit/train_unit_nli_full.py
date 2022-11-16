import argparse, json, torch
import numpy as np
import os

from nli_mixed_models.trainers.nli_trainer import (
    UnitRandomInterceptsTrainer,
    UnitRandomSlopesTrainer,
)
from scripts.setup_logging import setup_logging
from scripts.training_utils import (
    parameter_grid,
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

            LOG.info("Finished training full model.")


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
