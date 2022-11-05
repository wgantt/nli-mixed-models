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
            ver = load_veridicality()
            hyperparams["n_participants"] = ver.participant.unique().shape[0]
            LOG.info("...Complete.")

            # Log the device being used
            device = hyperparams["device"]
            LOG.info(f"Using device {device}")

            # Assign folds to each veridicality example
            k_folds = settings["k_folds"]
            split_type = settings["split_type"]
            setting = hyperparams["setting"]
            ver = generate_splits(ver, split_type, k_folds=k_folds, datatype="v")
            LOG.info(
                f"Beginning training with {k_folds}-fold cross-validation in the {setting} setting, partitioning based on {split_type}."
            )

            # Save the DataFrame used for training + eval, in case
            # we need to recover it later.
            data_file_name = (
                checkpoint_dir
                + "/"
                + "-".join([checkpoint_file_name, "partition", split_type, "data"])
                + ".csv"
            )
            ver.to_csv(data_file_name, index=False)

            # Perform k-fold cross-validation
            # Note: we don't really need to be keeping track of the
            # random loss, since it should be constant. This is just
            # a sanity check.
            loss_all = []
            fixed_loss_all = []
            random_loss_all = []
            acc_all = []
            best_all = []

            loss_all_b = []
            fixed_loss_all_b = []
            random_loss_all_b = []
            acc_all_b = []
            best_all_b = []
            for i in range(k_folds):

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

                # Select the folds
                test_fold = i
                train_folds = [j for j in range(k_folds) if j != test_fold]
                LOG.info(f"Beginning training with test fold={test_fold}.")

                train_data = ver[ver.fold.isin(train_folds)]
                test_data = ver[ver.fold == test_fold]

                # Fit the model on the train folds
                cat_model = cnli_trainer.fit(train_data=train_data, **trainparams)
                LOG.info("Finished training.")

                # Save the model
                if save_checkpoints:
                    LOG.info("Saving model...")
                    model_name = (
                        "-".join(
                            [
                                checkpoint_file_name,
                                "fold",
                                str(i),
                                "partition",
                                split_type,
                            ]
                        )
                        + ".pt"
                    )
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
                ) = cnli_eval.eval(test_data, trainparams["batch_size"])
                loss_all.append(loss_mean)
                fixed_loss_all.append(fixed_loss_mean)
                random_loss_all.append(random_loss_mean)
                acc_all.append(acc_mean)
                best_all.append(best_mean)
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
                ) = cnli_eval.eval(test_data, trainparams["batch_size"])
                loss_all_b.append(loss_mean)
                fixed_loss_all_b.append(fixed_loss_mean)
                random_loss_all_b.append(random_loss_mean)
                acc_all_b.append(acc_mean)
                best_all_b.append(best_mean)
                LOG.info(f"Test results for fold {i}, subtask b")
                LOG.info(f"Mean loss:           {loss_mean}")
                LOG.info(f"Mean fixed loss:     {fixed_loss_mean}")
                LOG.info(f"Mean random loss:    {random_loss_mean}")
                LOG.info(f"Mean accuracy:       {acc_mean}")
                LOG.info(f"Prop. best possible: {best_mean}")

            LOG.info("Finished k-fold cross evaluation.")
            LOG.info("Subtask a results:")
            LOG.info(f"Mean loss:           {np.round(np.mean(loss_all), 4)}")
            LOG.info(
                f"Mean fixed loss:           {np.round(np.mean(fixed_loss_all), 4)}"
            )
            LOG.info(
                f"Mean random loss:           {np.round(np.mean(random_loss_all), 4)}"
            )
            LOG.info(f"Mean accuracy:       {np.round(np.mean(acc_all), 4)}")
            LOG.info(f"Prop. best possible: {np.round(np.mean(best_all), 4)}")
            LOG.info("Subtask b results:")
            LOG.info(f"Mean loss:           {np.round(np.mean(loss_all_b), 4)}")
            LOG.info(
                f"Mean fixed loss:           {np.round(np.mean(fixed_loss_all_b), 4)}"
            )
            LOG.info(
                f"Mean random loss:           {np.round(np.mean(random_loss_all_b), 4)}"
            )
            LOG.info(f"Mean accuracy:       {np.round(np.mean(acc_all_b), 4)}")
            LOG.info(f"Prop. best possible: {np.round(np.mean(best_all_b), 4)}")


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
