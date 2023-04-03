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
from torch.distributions import Beta

LOG = setup_logging()


class NaturalLanguageInferencePredict:
    def __init__(
        self, model: NaturalLanguageInference, subtask: str = "a", device="cpu"
    ):
        self.nli = model
        self.subtask = subtask
        self.lossfunc = self.LOSS_CLASS()
        self.device = device

    def predict(self, data: pd.DataFrame, batch_size: int = 32):

        self.nli.eval()

        with torch.no_grad():
            n_batches = np.ceil(data.shape[0] / batch_size)
            data = data.reset_index(drop=True)
            data.loc[:, "batch_idx"] = np.repeat(np.arange(n_batches), batch_size)[
                : data.shape[0]
            ]

            # Tensor for accumulating predictions
            all_pred_mean = torch.FloatTensor().to(self.device)
            all_pred_precision = torch.FloatTensor().to(self.device)

            # Make predictions for each batch in data
            for batch, items in data.groupby("batch_idx"):
                LOG.info("predicting batch [%s/%s]" % (int(batch), int(n_batches)))

                if self.subtask == "a":
                    participant = torch.LongTensor(items.participant.values)
                else:
                    participant = None

                item = torch.LongTensor(items.item.values).to(self.device)

                # Embed items
                embedding = self.nli.embed(items)

                # Calculate model prediction
                if isinstance(self.nli, UnitRandomIntercepts) or isinstance(
                    self.nli, UnitRandomSlopes
                ):
                    prediction, _ = self.nli(embedding, participant, item)
                    alpha, beta = prediction
                    pred_mean = alpha / (alpha + beta)
                    pred_precision = alpha + beta
                    all_pred_mean = torch.cat((all_pred_mean, pred_mean))
                    all_pred_precision = torch.cat((all_pred_precision, pred_precision))
                else:
                    pred_mean, _ = self.nli(embedding, participant, item)
                    all_pred_mean = torch.cat((all_pred_mean, pred_mean))

            return all_pred_mean, all_pred_precision


# Unit predict
class UnitPredict(NaturalLanguageInferencePredict):
    LOSS_CLASS = BetaLogProbLoss
    TARGET_TYPE = torch.FloatTensor


# Categorical predict
class CategoricalPredict(NaturalLanguageInferencePredict):
    LOSS_CLASS = CrossEntropyLoss
    TARGET_TYPE = torch.LongTensor
