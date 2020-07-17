import torch
import logging
import numpy as np
import pandas as pd

from torch.optim import Adam
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from ..modules.nli_base import NaturalLanguageInference
from ..modules.nli_random_intercepts import (
    UnitRandomIntercepts,
    CategoricalRandomIntercepts
)
from scripts.setup_logging import setup_logging
from ..modules.nli_random_slopes import (
    UnitRandomSlopes,
    CategoricalRandomSlopes
)
from ..trainers.nli_trainer import (
    BetaLogProbLoss,
    beta_mode
)
from scripts.eval_utils import (
    accuracy,
    absolute_error,
    accuracy_best,
)
from torch.distributions import Beta

LOG = setup_logging()

class NaturalLanguageInferenceEval:
    
    def __init__(self, model: NaturalLanguageInference, 
                 subtask: str = 'a', device='cpu'):
        self.nli = model
        self.subtask = subtask
        self.lossfunc = self.LOSS_CLASS()
        self.device = device

    
    def eval(self, test_data: pd.DataFrame, batch_size: int = 32):

        self.nli.eval()
        
        with torch.no_grad():
          n_batches = np.ceil(test_data.shape[0]/batch_size)
          test_data.loc[:,'batch_idx'] = np.repeat(np.arange(n_batches), batch_size)[:test_data.shape[0]]

          loss_trace = []
          fixed_loss_trace = []
          random_loss_trace = []
          metric_trace = []
          best_trace = []
          spearman_trace = []

          # Calculate metrics for each batch in test set
          for batch, items in test_data.groupby('batch_idx'):
            LOG.info('evaluating batch [%s/%s]' % (int(batch), int(n_batches)))

            if self.subtask == 'a':
              participant = torch.LongTensor(items.participant.values)
            else:
              participant = None
      
            # Get target values of appropriate type
            if isinstance(self.nli, CategoricalRandomIntercepts) or isinstance(self.nli, CategoricalRandomSlopes):
              target = torch.LongTensor(items.target.values).to(self.device)
              modal_response = torch.LongTensor(items.modal_response.values).to(self.device)
            else:
              target = torch.FloatTensor(items.target.values).to(self.device)
              modal_response = torch.FloatTensor(items.modal_response.values).to(self.device)

            # Embed items    
            embedding = self.nli.embed(items)

            # Calculate model prediction and compute fixed & random loss
            if isinstance(self.nli, UnitRandomIntercepts) or \
               isinstance(self.nli, UnitRandomSlopes):
              prediction, random_loss = self.nli(embedding, participant)
              alpha, beta = prediction
              prediction = self.TARGET_TYPE(beta_mode(alpha, beta)).to(self.device)
              fixed_loss = self.lossfunc(alpha, beta, target)       
            else:
              prediction, random_loss = self.nli(embedding, participant)
              fixed_loss = self.lossfunc(prediction, target)

            random_loss = random_loss if isinstance(random_loss, float) else random_loss.item()

            # Add total loss to trace
            loss = fixed_loss + random_loss
            loss_trace.append(loss.item())
            fixed_loss_trace.append(fixed_loss.item())
            random_loss_trace.append(random_loss)

            # If categorical, calculate accuracy (and Spearman's coefficient)
            if isinstance(self.nli, CategoricalRandomIntercepts) or \
              isinstance(self.nli, CategoricalRandomSlopes):
              acc = accuracy(prediction, target)
              best = accuracy_best(items)
              metric_trace.append(acc)
              best_trace.append(acc/best)

              # Calculate Spearman
              spearman_df = pd.DataFrame()
              spearman_df['true'] = pd.Series(target.cpu().detach().numpy())
              spearman_df['predicted'] = pd.Series(prediction.argmax(1).cpu().detach().numpy())

                      
            # If unit, calculate absolute error
            else:
              error = absolute_error(prediction, target)
              best = absolute_error(modal_response, target)
              metric_trace.append(error)
              best_trace.append(1 - (error-best)/best)

            spearman = spearman_df.corr().iloc[0,1]
            spearman_trace.append(spearman)

          # Calculate and return mean of metrics across all batches
          loss_mean = np.round(np.mean(loss_trace), 4)
          fixed_loss_mean = np.round(np.mean(fixed_loss_trace), 4)
          random_loss_mean = np.round(np.mean(random_loss_trace), 4)
          metric_mean = np.round(np.mean(metric_trace), 4)
          best_mean = np.round(np.mean(best_trace), 4)
          spearman_mean = np.round(np.mean(spearman_trace), 4)

          return loss_mean, fixed_loss_mean, random_loss_mean, metric_mean, best_mean, spearman_mean


# Unit eval
class UnitEval(NaturalLanguageInferenceEval):
    LOSS_CLASS = BetaLogProbLoss
    TARGET_TYPE = torch.FloatTensor


# Categorical eval
class CategoricalEval(NaturalLanguageInferenceEval):
    LOSS_CLASS = CrossEntropyLoss
    TARGET_TYPE = torch.LongTensor
