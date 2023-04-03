import torch
import logging
import numpy as np
import pandas as pd

from torch.optim import Adam
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from ..modules.nli_random_intercepts import (
    UnitRandomIntercepts,
    CategoricalRandomIntercepts,
)
from scripts.setup_logging import setup_logging
from ..modules.nli_random_slopes import UnitRandomSlopes, CategoricalRandomSlopes
from scripts.eval_utils import (
    accuracy,
    absolute_error,
    accuracy_best,
    absolute_error_best,
)
from torch.distributions import Beta

LOG = setup_logging()


class NaturalLanguageInferenceTrainer:
    # TODO: This could probably be made a bit neater by abstracting
    # the metric (i.e. accuracy/absolute error) into the subclasses

    def __init__(
        self,
        n_participants: int,
        n_items: int,
        embedding_dim: int = 768,
        n_predictor_layers: int = 2,
        hidden_dim: int = 128,
        setting: str = "extended",
        use_item_variance: bool = False,
        use_sampling: bool = False,
        n_samples: int = 100,
        device="cpu",
    ):
        self.embedding_dim = embedding_dim
        self.n_predictor_layers = n_predictor_layers
        self.hidden_dim = hidden_dim
        self.n_participants = n_participants
        self.n_items = n_items
        self.setting = setting
        self.use_item_variance = use_item_variance
        self.device = torch.device(device)
        self.nli = self.MODEL_CLASS(
            embedding_dim,
            n_predictor_layers,
            hidden_dim,
            self.OUTPUT_DIM,
            n_participants,
            n_items,
            setting,
            use_item_variance,
            use_sampling,
            n_samples,
            device,
        ).to(device)
        self.data_type = (
            "categorical"
            if self.MODEL_CLASS is CategoricalRandomIntercepts
            or self.MODEL_CLASS is CategoricalRandomSlopes
            else "unit"
        )

    def fit(
        self,
        train_data: pd.DataFrame,
        batch_size: int = 32,
        n_epochs: int = 10,
        lr: float = 1e-2,
        verbosity: int = 10,
    ):

        optimizer = Adam(self.nli.parameters(), lr=lr)
        lossfunc = self.LOSS_CLASS()

        self.nli.train()

        n_batches = int(np.ceil(train_data.shape[0] / batch_size))

        LOG.info(
            f"Training for a max of {n_epochs} epochs, with "
            f"{n_batches} batches per epoch (batch size={batch_size})"
        )

        # For tracking a moving average of the loss across epochs for
        # early stopping
        all_loss_trace = []
        iters_without_improvement = 0
        prev_epoch_mean_loss = np.inf
        for epoch in range(n_epochs):

            train_data = train_data.sample(frac=1).reset_index(drop=True)

            train_data.loc[:, "batch_idx"] = np.repeat(
                np.arange(n_batches), batch_size
            )[: train_data.shape[0]]

            loss_trace = []
            random_loss_trace = []
            fixed_loss_trace = []
            metric_trace = []
            best_trace = []
            epoch_loss_trace = []

            for batch, items in train_data.groupby("batch_idx"):
                self.nli.zero_grad()

                # Only in the extended setting do we need participant information
                if self.setting == "extended":
                    participant = torch.LongTensor(items.participant.values).to(
                        self.device
                    )
                else:
                    participant = None

                item = torch.LongTensor(items.item.values).to(self.device)

                # Get targets, modal responses, and item embeddings
                target = self.TARGET_TYPE(items.target.values).to(self.device)
                modal_response = self.TARGET_TYPE(items.modal_response.values).to(
                    self.device
                )
                embedding = self.nli.embed(items).to(self.device)

                # Get model prediction and random loss (converting the latter as appropriate)
                prediction, random_loss = self.nli(embedding, participant, item)
                random_loss = (
                    random_loss
                    if isinstance(random_loss, float)
                    else random_loss.item()
                )

                # If unit case, get final prediction by calculating mode of predicted beta distribution
                if self.data_type == "unit":
                    alpha, beta = prediction
                    prediction = self.TARGET_TYPE(beta_mode(alpha, beta)).to(
                        self.device
                    )
                    fixed_loss = lossfunc(alpha, beta, target)
                # Otherwise, model returns predicted class directly
                else:
                    fixed_loss = lossfunc(prediction, target)

                loss = fixed_loss + random_loss

                # Shouldn't this include the random loss? -B.K.
                # loss_trace.append(loss.item()-random_loss.item())
                fixed_loss_trace.append(fixed_loss.item())
                random_loss_trace.append(random_loss)
                loss_trace.append(loss.item())
                all_loss_trace.append(loss.item())
                epoch_loss_trace.append(loss.item())

                if self.data_type == "categorical":
                    acc = accuracy(prediction, target)
                    best = accuracy_best(items)
                    metric_trace.append(acc)
                    best_trace.append(acc / best)

                elif self.data_type == "unit":
                    error = absolute_error(prediction, target)
                    best = absolute_error(modal_response, target)
                    metric_trace.append(error)
                    best_trace.append(1 - (error - best) / best)

                if not (batch % verbosity):
                    LOG.info(f"epoch:               {int(epoch)}")
                    LOG.info(f"batch:               {int(batch)}")
                    LOG.info(f"mean loss:           {np.round(np.mean(loss_trace), 4)}")
                    LOG.info(
                        f"mean fixed loss:     {np.round(np.mean(fixed_loss_trace), 4)}"
                    )
                    # This should obviously remain constant
                    LOG.info(
                        f"mean random loss:    {np.round(np.mean(random_loss_trace), 4)}"
                    )
                    if self.data_type == "categorical":
                        LOG.info(
                            f"mean acc.:           {np.round(np.mean(metric_trace), 4)}"
                        )
                    elif self.data_type == "unit":
                        LOG.info(
                            f"mean error:          {np.round(np.mean(metric_trace), 4)}"
                        )

                    LOG.info(f"Prop. best possible: {np.round(np.mean(best_trace), 4)}")
                    LOG.info("")

                    LOG.info("")

                    loss_trace = []
                    fixed_loss_trace = []
                    random_loss_trace = []
                    metric_trace = []
                    best_trace = []

                loss.backward()
                optimizer.step()

            cur_epoch_mean_loss = np.mean(epoch_loss_trace)
            print(f"prev epoch mean loss: {prev_epoch_mean_loss}")
            print(f"cur epoch mean loss: {cur_epoch_mean_loss}")
            if prev_epoch_mean_loss - cur_epoch_mean_loss > 0.01:
                prev_epoch_mean_loss = cur_epoch_mean_loss
            else:
                LOG.info(
                    "No performance improvement over previous epoch. " "Stopping early."
                )
                return self.nli

        return self.nli.eval()


# Custom loss functions
class BetaLogProbLoss(torch.nn.Module):
    def __init__(self):
        super(BetaLogProbLoss, self).__init__()

    def forward(self, alphas, betas, targets):
        # Added this to adjust targets to constrain them within soft interval (0, 1) - B.K.
        eps = torch.tensor(0.000001)
        zero_idx = (targets == 0).nonzero()
        one_idx = (targets == 1).nonzero()
        targets_adj = targets.clone()
        targets_adj[zero_idx] += eps
        targets_adj[one_idx] -= eps
        # Return log probability of beta distribution (using adjusted targets)
        return -torch.mean(Beta(alphas, betas).log_prob(targets_adj))


def beta_mode(alpha, beta):
    """Compute the mode of the beta distribution"""
    modes = []
    for a, b in zip(alpha, beta):
        if a > 1 and b > 1:
            modes.append((a - 1) / (a + b - 2))
        elif a == 1 and b == 1:
            # This can technically be any value in (0,1)
            modes.append(0.5)
        elif a < 1 and b < 1:
            # Can be either 0 or 1. We pick 0.
            modes.append(0)
        elif a <= 1 and b > 1:
            # Always 0
            modes.append(0)
        elif a > 1 and b <= 1:
            # Always 1
            modes.append(1)
        else:
            raise ValueError("Unable to compute beta mode!")
    return modes


# Unit models
class UnitTrainer(NaturalLanguageInferenceTrainer):
    LOSS_CLASS = BCEWithLogitsLoss
    TARGET_TYPE = torch.FloatTensor
    OUTPUT_DIM = 1


class UnitRandomInterceptsTrainer(UnitTrainer):
    MODEL_CLASS = UnitRandomIntercepts
    LOSS_CLASS = BetaLogProbLoss


class UnitRandomSlopesTrainer(UnitTrainer):
    MODEL_CLASS = UnitRandomSlopes
    LOSS_CLASS = BetaLogProbLoss


# Categorical models
class CategoricalTrainer(NaturalLanguageInferenceTrainer):
    LOSS_CLASS = CrossEntropyLoss
    TARGET_TYPE = torch.LongTensor
    OUTPUT_DIM = 3


class CategoricalRandomInterceptsTrainer(CategoricalTrainer):
    MODEL_CLASS = CategoricalRandomIntercepts


class CategoricalRandomSlopesTrainer(CategoricalTrainer):
    MODEL_CLASS = CategoricalRandomSlopes
