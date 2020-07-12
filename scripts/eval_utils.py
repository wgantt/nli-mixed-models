import torch
import logging
import numpy as np
import pandas as pd

from itertools import product
from pandas.api.types import CategoricalDtype
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from .setup_logging import setup_logging


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


