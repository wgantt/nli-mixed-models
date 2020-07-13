import torch
import logging
import numpy as np
import pandas as pd

from torch.optim import Adam
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from ..modules.nli_random_intercepts import (
    UnitRandomInterceptsNormal,
    UnitRandomInterceptsBeta,
    CategoricalRandomIntercepts
)
from scripts.setup_logging import setup_logging
from ..modules.nli_random_slopes import (
    UnitRandomSlopes,
    CategoricalRandomSlopes
)
from scripts.eval_utils import (
    accuracy,
    absolute_error,
    accuracy_best,
    absolute_error_best
)
from torch.distributions import Beta

LOG = setup_logging()

# Ideally, these should be parameters of the trainer itself, but didn't
# want to take the time to do the refactoring. -W.G.
MOVING_AVERAGE_WINDOW_SIZE = 100
TOLERANCE = 100

class NaturalLanguageInferenceTrainer:
    # TODO: This could probably be made a bit neater by abstracting
    # the metric (i.e. accuracy/absolute error) into the subclasses
    
    def __init__(self, n_participants: int, 
                 embedding_dim: int = 768, 
                 n_predictor_layers: int = 2,
                 setting: str = 'extended',
                 device='cpu'):
        self.embedding_dim = embedding_dim
        self.n_predictor_layers = n_predictor_layers
        self.n_participants = n_participants
        self.setting = setting
        self.device = torch.device(device)
        self.nli = self.MODEL_CLASS(embedding_dim, 
                                    n_predictor_layers,
                                    self.OUTPUT_DIM, 
                                    n_participants,
                                    setting,
                                    device)
        self.data_type = 'categorical' if \
            self.MODEL_CLASS is CategoricalRandomIntercepts or \
            self.MODEL_CLASS is CategoricalRandomSlopes \
            else 'unit'
    
    def fit(self, train_data: pd.DataFrame, batch_size: int = 32, 
            n_epochs: int = 10, lr: float = 1e-2, verbosity: int = 10):
        
        optimizer = Adam(self.nli.parameters(),
                         lr=lr)
        lossfunc = self.LOSS_CLASS()
        
        self.nli.train()
        
        n_batches = int(np.ceil(train_data.shape[0]/batch_size))
        
        LOG.info(f'Training for a max of {n_epochs} epochs, with '\
                 f'{n_batches} batches per epoch (batch size={batch_size})')
        
        # For tracking a moving average of the loss across epochs for
        # early stopping
        all_loss_trace = []       
        iters_without_improvement = 0
        for epoch in range(n_epochs):

            train_data = train_data.sample(frac=1).reset_index(drop=True)

            train_data.loc[:,'batch_idx'] = np.repeat(np.arange(n_batches), batch_size)[:train_data.shape[0]]
            
            loss_trace = []
            random_loss_trace = []
            fixed_loss_trace = []
            metric_trace = []
            best_trace = []
            
            for batch, items in train_data.groupby('batch_idx'):
                self.nli.zero_grad()
                
                # Only in the extended setting do we need participant information
                if self.setting == 'extended':
                    participant = torch.LongTensor(items.participant.values).to(self.device)
                else:
                    participant = None

                target = self.TARGET_TYPE(items.target.values)
                
                embedding = self.nli.embed(items)
                
                if self.MODEL_CLASS == UnitRandomInterceptsBeta or \
                   self.MODEL_CLASS == UnitRandomSlopes:
                    alpha, beta, prediction, random_loss = self.nli(embedding, participant)
                    fixed_loss = lossfunc(alpha, beta, target)
                else:
                    prediction, random_loss = self.nli(embedding, participant)
                    fixed_loss = lossfunc(prediction, target)
                
                loss = fixed_loss + random_loss
                
                # Shouldn't this include the random loss? -B.K.
                # loss_trace.append(loss.item()-random_loss.item())
                fixed_loss_trace.append(fixed_loss.item())
                random_loss_trace.append(random_loss.item())
                loss_trace.append(loss.item())
                all_loss_trace.append(loss.item())
                
                if self.data_type == 'categorical':
                    acc = accuracy(prediction, target)
                    best = accuracy_best(items)
                    metric_trace.append(acc)
                    best_trace.append(acc/best)
                    
                elif self.data_type == 'unit':
                    error = absolute_error(prediction, target)
                    best = absolute_error_best(items)
                    metric_trace.append(error)
                    best_trace.append(1 - (error-best)/best)
                
                if not (batch % verbosity):
                    LOG.info(f'epoch:               {int(epoch)}')
                    LOG.info(f'batch:               {int(batch)}')
                    LOG.info(f'mean loss:           {np.round(np.mean(loss_trace), 2)}')
                    LOG.info(f'mean fixed loss:     {np.round(np.mean(fixed_loss_trace), 2)}')
                    # This should obviously remain constant
                    LOG.info(f'mean random loss:    {np.round(np.mean(random_loss_trace), 2)}')
                    if self.data_type == 'categorical':
                        LOG.info(f'mean acc.:           {np.round(np.mean(metric_trace), 2)}')
                    elif self.data_type == 'unit':
                        LOG.info(f'mean error:          {np.round(np.mean(metric_trace), 2)}')
                    LOG.info(f'prop. best possible: {np.round(np.mean(best_trace), 2)}')
                    LOG.info('')
                    
                    LOG.info('')

                    loss_trace = []
                    fixed_loss_trace = []
                    random_loss_trace = []
                    metric_trace = []
                    best_trace = []

                # Evaluate moving average loss
                if len(all_loss_trace) == MOVING_AVERAGE_WINDOW_SIZE:
                    prev_moving_average_loss = np.mean(all_loss_trace)
                elif len(all_loss_trace) > MOVING_AVERAGE_WINDOW_SIZE:
                    # Compute the moving average of the loss
                    cur_moving_average_loss = \
                        np.mean(all_loss_trace[-MOVING_AVERAGE_WINDOW_SIZE:])
                    # Determine whether there was any improvement or not
                    if cur_moving_average_loss >= prev_moving_average_loss:
                        iters_without_improvement += 1
                    else:
                        prev_moving_average_loss = cur_moving_average_loss
                        iters_without_improvement = 0
                    # Early stopping condition
                    if iters_without_improvement == TOLERANCE:
                        LOG.info(f'Reached {TOLERANCE} minibatches without '\
                                 f'improvement. Stopping early')
                        return self.nli.eval()
                
                loss.backward()
                optimizer.step()
                
        return self.nli.eval()


# Custom loss functions
class BetaLogProbLoss(torch.nn.Module):
    def __init__(self):
        super(BetaLogProbLoss,self).__init__()

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


# Unit models
class UnitTrainer(NaturalLanguageInferenceTrainer):
    LOSS_CLASS = BCEWithLogitsLoss
    TARGET_TYPE = torch.FloatTensor
    OUTPUT_DIM = 1

class UnitRandomInterceptsNormalTrainer(UnitTrainer):
    MODEL_CLASS = UnitRandomInterceptsNormal

class UnitRandomInterceptsBetaTrainer(UnitTrainer):
    MODEL_CLASS = UnitRandomInterceptsBeta
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
