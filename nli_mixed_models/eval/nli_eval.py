import torch
import logging
import numpy as np
import pandas as pd

from torch.optim import Adam
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from ..modules.nli_base import NaturalLanguageInference
from ..modules.nli_random_intercepts import (
    UnitRandomIntercepts,
    CategoricalRandomIntercepts,
)
from scripts.setup_logging import setup_logging
from ..modules.nli_random_slopes import UnitRandomSlopes, CategoricalRandomSlopes
from ..trainers.nli_trainer import BetaLogProbLoss, beta_mode
from scripts.eval_utils import (
    accuracy,
    absolute_error,
    accuracy_best,
)
from torch.distributions import Beta

LOG = setup_logging()


class NaturalLanguageInferenceEval:
    def __init__(
        self, model: NaturalLanguageInference, subtask: str = "a", device="cpu"
    ):
        self.nli = model
        self.subtask = subtask
        self.lossfunc = self.LOSS_CLASS()
        self.device = device

    def eval(self, test_data: pd.DataFrame, batch_size: int = 32):

        self.nli.eval()

        with torch.no_grad():
            n_batches = np.ceil(test_data.shape[0] / batch_size)
            test_data = test_data.sample(frac=1).reset_index(drop=True)
            test_data.loc[:, "batch_idx"] = np.repeat(np.arange(n_batches), batch_size)[
                : test_data.shape[0]
            ]

            loss_trace = []
            fixed_loss_trace = []
            random_loss_trace = []
            metric_trace = []
            best_trace = []

            # Tensors for accumulating predictions, targets, and modal
            # responses across batches.
            if isinstance(self.nli, CategoricalRandomIntercepts) or isinstance(
                self.nli, CategoricalRandomSlopes
            ):
                all_predictions = torch.FloatTensor().to(self.device)
                all_targets = torch.LongTensor().to(self.device)
                all_modal_responses = torch.LongTensor().to(self.device)
                all_best = torch.LongTensor().to(self.device)

                naive_prediction = test_data.target.mode().item()
                naive_acc = len(test_data[test_data.target == naive_prediction]) / len(
                    test_data
                )
                LOG.info(f"naive accuracy across fold: {naive_acc}")
            else:
                all_predictions = torch.FloatTensor().to(self.device)
                all_targets = torch.FloatTensor().to(self.device)
                all_modal_responses = torch.FloatTensor().to(self.device)
                all_best = torch.LongTensor().to(self.device)

            # Calculate metrics for each batch in test set
            for batch, items in test_data.groupby("batch_idx"):
                LOG.info("evaluating batch [%s/%s]" % (int(batch), int(n_batches)))

                if self.subtask == "a":
                    participant = torch.LongTensor(items.participant.values)
                else:
                    participant = None

                # Get target values of appropriate type
                if isinstance(self.nli, CategoricalRandomIntercepts) or isinstance(
                    self.nli, CategoricalRandomSlopes
                ):
                    target = torch.LongTensor(items.target.values).to(self.device)
                    modal_response = torch.LongTensor(items.modal_response.values).to(
                        self.device
                    )
                else:
                    target = torch.FloatTensor(items.target.values).to(self.device)
                    modal_response = torch.FloatTensor(items.modal_response.values).to(
                        self.device
                    )
                all_targets = torch.cat((all_targets, target))
                all_modal_responses = torch.cat((all_modal_responses, modal_response))

                # Embed items
                embedding = self.nli.embed(items)

                # Calculate model prediction and compute fixed & random loss
                if isinstance(self.nli, UnitRandomIntercepts) or isinstance(
                    self.nli, UnitRandomSlopes
                ):
                    prediction, random_loss = self.nli(embedding, participant)
                    alpha, beta = prediction
                    prediction = alpha / (alpha + beta)
                    fixed_loss = self.lossfunc(alpha, beta, target)
                else:
                    prediction, random_loss = self.nli(embedding, participant)
                    fixed_loss = self.lossfunc(prediction, target)
                all_predictions = torch.cat((all_predictions, prediction))

                random_loss = (
                    random_loss
                    if isinstance(random_loss, float)
                    else random_loss.item()
                )

                # Add total loss to trace
                loss = fixed_loss + random_loss
                loss_trace.append(loss.item())
                fixed_loss_trace.append(fixed_loss.item())
                random_loss_trace.append(random_loss)

                # If categorical, calculate accuracy (and Spearman's coefficient)
                if isinstance(self.nli, CategoricalRandomIntercepts) or isinstance(
                    self.nli, CategoricalRandomSlopes
                ):
                    acc = accuracy(prediction, target)
                    best = accuracy_best(items)
                    metric_trace.append(acc)
                    best_trace.append(acc / best)

                # If unit, calculate absolute error
                else:
                    error = absolute_error(prediction, target)
                    best = absolute_error(modal_response, target)
                    metric_trace.append(error)
                    best_trace.append(1 - (error - best) / best)

            # Calculate Spearman's correlation coefficient between
            # 1. Best possible (i.e. modal) responses and true responses
            # 2. Predicted responses and true responses
            """
            spearman_df = pd.DataFrame()
            spearman_df["true"] = pd.Series(all_targets.cpu().detach().numpy())
            spearman_df["predicted"] = pd.Series(
                all_predictions.cpu().detach().numpy()
            )
            spearman_df["best"] = pd.Series(
                all_modal_responses.cpu().detach().numpy()
            )
            spearman_predicted = (
                spearman_df[["true", "predicted"]]
                .corr(method="spearman")
                .iloc[0, 1]
            )
            spearman_best = (
                spearman_df[["true", "best"]].corr(method="spearman").iloc[0, 1]
            )
            """

            # Calculate and return mean of metrics across all batches
            loss_mean = np.round(np.mean(loss_trace), 4)
            fixed_loss_mean = np.round(np.mean(fixed_loss_trace), 4)
            random_loss_mean = np.round(np.mean(random_loss_trace), 4)
            metric_mean = np.round(np.mean(metric_trace), 4)
            best_mean = np.round(np.mean(best_trace), 4)
            """
            spearman_predicted = np.round(spearman_predicted, 4)
            spearman_best = np.round(spearman_best, 4)
            """
            spearman_predicted = 0
            spearman_best = 1

            # Macroaverage
            best_mean = (all_modal_responses == all_targets).cpu().numpy().mean()
            worst_mean = naive_acc
            metric_mean = accuracy(all_predictions, all_targets)

            # An undefined Spearman means that all the predicted values are
            # the same. This is unlikely to occur across an entire test fold,
            # but not impossible. As noted in the paper, an undefined Spearman
            # correlation essentially represents *0* correlation.
            if np.isnan(spearman_predicted):
                spearman_predicted = 0.0

            return (
                loss_mean,
                fixed_loss_mean,
                random_loss_mean,
                metric_mean,
                best_mean,
                spearman_predicted,
                spearman_best,
                worst_mean,
            )


# Unit eval
class UnitEval(NaturalLanguageInferenceEval):
    LOSS_CLASS = BetaLogProbLoss
    TARGET_TYPE = torch.FloatTensor


# Categorical eval
class CategoricalEval(NaturalLanguageInferenceEval):
    LOSS_CLASS = CrossEntropyLoss
    TARGET_TYPE = torch.LongTensor
