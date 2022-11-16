import argparse, json, torch
import numpy as np
import os

from nli_mixed_models.trainers.nli_trainer import (
    CategoricalRandomInterceptsTrainer,
    CategoricalRandomSlopesTrainer,
)
from nli_mixed_models.eval.nli_eval import CategoricalEval
from scripts.setup_logging import setup_logging
from scripts.training_utils import (
    parameter_grid,
    load_veridicality,
    generate_splits,
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

            # Load MegaVeridicality data
            LOG.info("Loading MegaVeridicality data...")
            data = load_veridicality()
            hyperparams["n_participants"] = data.participant.unique().shape[0]
            LOG.info("...Complete.")

            # Log the device being used
            device = hyperparams["device"]
            LOG.info(f"Using device {device}")

            # Save the DataFrame used for training, in case
            # we need to recover it later.
            data_file_name = (
                checkpoint_dir
                + "/"
                + "-".join([checkpoint_file_name, "data"])
                + ".csv"
            )
            data.to_csv(data_file_name, index=False)


            # Dump current training settings
            LOG.info(
                "Initializing categorical NLI model with the following hyperparameters:"
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
                cnli_trainer = CategoricalRandomSlopesTrainer(**hyperparams)
            else:
                cnli_trainer = CategoricalRandomInterceptsTrainer(**hyperparams)

            train_data = data

            # Fit the model on the train folds
            cat_model = cnli_trainer.fit(train_data=train_data, **trainparams)
            LOG.info("Finished training.")

            # Save the model
            if save_checkpoints:
                LOG.info("Saving model...")
                model_name = checkpoint_file_name + ".pt"
                save_model_with_args(
                    params, cat_model, hyperparams, checkpoint_dir, model_name
                )
                LOG.info("Model saved.")

            # Evaluate the model on the test fold on subtask 'a'
            LOG.info("Evaluating subtask a")
            cnli_eval = CategoricalEval(cat_model, "a", device=device)
            (
                loss_mean,
                fixed_loss_mean,
                random_loss_mean,
                acc_mean,
                best_mean,
                spearman_mean,
                best_spearman_mean,
                _
            ) = cnli_eval.eval(train_data, trainparams["batch_size"])

            LOG.info(f"Test results for fold {i}, subtask a")
            LOG.info(f"Mean loss:           {loss_mean}")
            LOG.info(f"Mean fixed loss:     {fixed_loss_mean}")
            LOG.info(f"Mean random loss:    {random_loss_mean}")
            LOG.info(f"Mean accuracy:       {acc_mean}")
            LOG.info(f"Prop. best possible: {best_mean}")

            # Evaluate the model on the test fold on subtask 'b'
            LOG.info("Evaluating subtask b")
            cnli_eval = CategoricalEval(cat_model, "b", device=device)
            (
                loss_mean,
                fixed_loss_mean,
                random_loss_mean,
                acc_mean,
                best_mean,
                spearman_mean,
                best_spearman_mean,
                _
            ) = cnli_eval.eval(train_data, trainparams["batch_size"])

            LOG.info(f"Test results for fold {i}, subtask b")
            LOG.info(f"Mean loss:           {loss_mean}")
            LOG.info(f"Mean fixed loss:     {fixed_loss_mean}")
            LOG.info(f"Mean random loss:    {random_loss_mean}")
            LOG.info(f"Mean accuracy:       {acc_mean}")
            LOG.info(f"Prop. best possible: {best_mean}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a categorical NLI mixed effects model"
    )
    parser.add_argument(
        "--parameters",
        type=str,
        default="scripts/categorical/train_categorical_nli.json",
        help="Path to a JSON containing parameters to sweep",
    )
    args = parser.parse_args()
    main(args)
