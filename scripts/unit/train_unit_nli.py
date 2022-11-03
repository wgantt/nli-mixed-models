import argparse, json, torch
import numpy as np

from nli_mixed_models.trainers.nli_trainer import (
    UnitRandomInterceptsTrainer,
    UnitRandomSlopesTrainer,
)
from nli_mixed_models.eval.nli_eval import UnitEval
from scripts.setup_logging import setup_logging
from scripts.training_utils import (
    parameter_grid,
    load_neg_raising,
    load_difficulty,
    load_intensionality,
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

    for hyperparams in parameter_grid(params["hyper"]):
        for trainparams in parameter_grid(params["training"]):

            # Load data
            if "dataset" in settings and settings["dataset"] == "difficulty":
                LOG.info("Loading Difficulty data...")
                data = load_difficulty()
            elif "dataset" in settings and settings["dataset"] == "intensionality":
                LOG.info("Loading Intensionality data...")
                data = load_intensionality()
            else:
                LOG.info("Loading MegaNegRaising data...")
                data = load_neg_raising()
            hyperparams["n_participants"] = data.participant.unique().shape[0]
            LOG.info("...Complete.")

            # Log the device being used
            device = hyperparams["device"]
            LOG.info(f"Using device {device}")

            # Assign folds to each veridicality example
            k_folds = settings["k_folds"]
            split_type = settings["split_type"]
            setting = hyperparams["setting"]
            data = generate_splits(data, split_type, k_folds=k_folds, datatype="n")
            LOG.info(
                f"Beginning training with {k_folds}-fold cross-validation in the {setting} setting, partitioning based on {split_type}."
            )

            # Save the DataFrame used for training + eval in case
            # we need to recover it later
            data_file_name = (
                checkpoint_dir
                + "/"
                + "-".join([checkpoint_file_name, "partition", split_type, "data"])
                + ".csv"
            )
            data.to_csv(data_file_name, index=False)

            # Perform k-fold cross-validation
            # Note: we really don't need to be keeping track of the
            # random loss, since it should be constant. This is just
            # a sanity check.
            loss_all = []
            fixed_loss_all = []
            random_loss_all = []
            error_all = []
            best_all = []
            spearman_all = []
            best_spearman_all = []

            loss_all_b = []
            fixed_loss_all_b = []
            random_loss_all_b = []
            error_all_b = []
            best_all_b = []
            spearman_all_b = []
            best_spearman_all_b = []
            for i in range(k_folds):

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

                # Select the folds
                test_fold = i
                train_folds = [j for j in range(k_folds) if j != test_fold]
                LOG.info(f"Beginning training with test fold={test_fold}.")

                train_data = data[data.fold.isin(train_folds)]
                test_data = data[data.fold == test_fold]

                # Fit the model on the train folds
                unit_model = unli_trainer.fit(train_data=train_data, **trainparams)
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
                        params, unit_model, hyperparams, checkpoint_dir, model_name
                    )
                    LOG.info("Model saved.")

                # Evaluate the model on the test fold
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
                ) = unli_eval.eval(test_data, trainparams["batch_size"])
                loss_all.append(loss_mean)
                fixed_loss_all.append(fixed_loss_mean)
                random_loss_all.append(random_loss_mean)
                error_all.append(error_mean)
                best_all.append(best_mean)
                spearman_all.append(spearman)
                best_spearman_all.append(best_spearman)
                LOG.info(f"Test results for fold {i}, subtask a")
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
                ) = unli_eval_b.eval(test_data, trainparams["batch_size"])
                loss_all_b.append(loss_mean)
                fixed_loss_all_b.append(fixed_loss_mean)
                random_loss_all_b.append(random_loss_mean)
                error_all_b.append(error_mean)
                best_all_b.append(best_mean)
                spearman_all_b.append(spearman)
                best_spearman_all_b.append(best_spearman)
                LOG.info(f"Test results for fold {i}, subtask b")
                LOG.info(f"Mean fixed loss:     {fixed_loss_mean}")
                LOG.info(f"Mean error:          {error_mean}")
                LOG.info(f"Prop. best possible: {best_mean}")
                LOG.info(f"Mean spearman:       {spearman}")
                LOG.info(f"Best spearman:       {best_spearman}")
                LOG.info(f"Prop. best spearman: {spearman / best_spearman}")

            LOG.info("Finished k-fold cross evaluation.")
            LOG.info("Subtask (a) results (avg. across folds):")
            LOG.info(f"Mean loss:                    {np.round(np.mean(loss_all), 4)}")
            LOG.info(f"Mean fixed loss:              {np.round(np.mean(fixed_loss_all), 4)}")
            LOG.info(f"Mean random loss:             {np.round(np.mean(random_loss_all), 4)}")
            LOG.info(f"Mean error:                   {np.round(np.mean(error_all), 4)}")
            LOG.info(f"Mean spearman:                {np.round(np.mean(spearman_all), 4)}")
            LOG.info(f"Best spearman:                {np.round(np.mean(best_spearman_all), 4)}")
            LOG.info(f"Prop. best possible spearman: {np.round(np.mean(best_all), 4)}\n")
            LOG.info("Subtask (b) results (avg. across folds):")
            LOG.info(f"Mean loss:                    {np.round(np.mean(loss_all_b), 4)}")
            LOG.info(f"Mean fixed loss:              {np.round(np.mean(fixed_loss_all_b), 4)}")
            LOG.info(f"Mean random loss:             {np.round(np.mean(random_loss_all_b), 4)}")
            LOG.info(f"Mean error:                   {np.round(np.mean(error_all_b), 4)}")
            LOG.info(f"Mean spearman:                {np.round(np.mean(spearman_all_b), 4)}")
            LOG.info(f"Best spearman:                {np.round(np.mean(best_spearman_all_b), 4)}")
            LOG.info(f"Prop. best possible spearman: {np.round(np.mean(best_all_b), 4)}")


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
