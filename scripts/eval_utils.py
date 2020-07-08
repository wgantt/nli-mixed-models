import torch
import logging
import numpy as np
import pandas as pd

from itertools import product
from pandas.api.types import CategoricalDtype
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from .setup_logging import setup_logging

LOG = setup_logging()

def eval_model(data: pd.DataFrame, model, data_type, batch_size: int = 32):
  """Evaluates a trained model on a test set, using a given batch size. data_type
     should be either 'categorical' or 'unit'.
  """
  # TODO: currently this isn't encapsulated in a class - not sure if it would
  # be better to do so. Also, there's probably a better way of accessing the data
  # type (i.e. categorical vs. unit), but would require a little more restructuring
  # of the class interface.
  # TODO: in the paper, it distinguishes between "subtask (a)" and "subtask (b)",
  # depending on whether we're evaluating using accuracy/absolute error, or whether
  # we're using log likelihood of the responses given estimated theta/Sigma. However,
  # currently I'm just calculating/logging both of these regardless.

  model.eval()

  n_batches = np.ceil(data.shape[0]/batch_size)
  data['batch_idx'] = np.repeat(np.arange(n_batches), batch_size)[:data.shape[0]]

  lossfunc = CrossEntropyLoss() if data_type == 'categorical' else BCEWithLogitsLoss()

  loss_trace = []
  metric_trace = []
  best_trace = []

  for batch, items in data.groupby('batch_idx'):
    LOG.info('evaluating batch [%s/%s]' % (int(batch), int(n_batches)))

    participant = torch.LongTensor(items.participant.values)
    target = torch.LongTensor(items.target.values) if data_type == 'categorical' else \
             torch.FloatTensor(items.target.values)
    embedding = model.embed(items)

    prediction, random_loss = model(embedding, participant)
    fixed_loss = lossfunc(prediction, target)
    loss = fixed_loss + random_loss

    loss_trace.append(loss.item())
                
    if data_type == 'categorical':
      acc = accuracy(prediction, target)
      best = accuracy_best(items)
      metric_trace.append(acc)
      best_trace.append(acc/best)
                    
    elif data_type == 'unit':
      error = absolute_error(prediction, target)
      best = absolute_error_best(items)
      metric_trace.append(error)
      best_trace.append(1 - (error-best)/best)

  loss_mean = np.round(np.mean(loss_trace), 2)
  metric_mean = np.round(np.mean(metric_trace), 2)
  best_mean = np.round(np.mean(best_trace), 2)

  return loss_mean, metric_mean, best_mean


def accuracy(prediction, target):
  """The fraction of predictions which match the target class"""
  return (prediction.argmax(1) == target).data.cpu().numpy().mean()

def absolute_error(prediction, target, lossfunc = BCEWithLogitsLoss()):
  """The absolute error using some loss function (BCEWithLogitsLoss by default)"""
  return lossfunc(prediction, target).item()

def accuracy_best(items):
  """The best possible accuracy on the given items"""
  return (items.modal_response == items.target).mean()

def absolute_error_best(items):
  """The best possible error (cross-entropy loss) on the given items"""
  return -(items.target.values * np.log(items.modal_response.values) +\
          (1-items.target.values) * np.log(1-items.modal_response.values)).mean()


