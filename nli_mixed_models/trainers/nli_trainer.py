import torch
import numpy as np
import pandas as pd

from torch.optim import Adam
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from ..modules.nli_random_intercepts import (
    UnitRandomInterceptsNormal,
    UnitRandomInterceptsBeta,
    CategoricalRandomIntercepts
)
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

class NaturalLanguageInferenceTrainer:
    # TODO: This could probably be made a bit neater by abstracting
    # the metric (i.e. accuracy/absolute error) into the subclasses
    
    def __init__(self, n_participants: int, 
                 embedding_dim: int = 768, 
                 n_predictor_layers: int = 2,
                 setting: str = 'extended',
                 device=torch.device('cpu')):
        self.embedding_dim = embedding_dim
        self.n_predictor_layers = n_predictor_layers
        self.n_participants = n_participants
        self.setting = setting
        self.device = device
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
    
    def fit(self, data: pd.DataFrame, batch_size: int = 32, 
            n_epochs: int = 10, lr: float = 1e-2, verbosity: int = 10):
        
        optimizer = Adam(self.nli.parameters(),
                         lr=lr)
        lossfunc = self.LOSS_CLASS()
        
        self.nli.train()
        
        n_batches = np.ceil(data.shape[0]/batch_size)
        
        for epoch in range(n_epochs):

            data = data.sample(frac=1).reset_index(drop=True)

            data['batch_idx'] = np.repeat(np.arange(n_batches), batch_size)[:data.shape[0]]
            
            loss_trace = []
            metric_trace = []
            best_trace = []
            
            for batch, items in data.groupby('batch_idx'):
                self.nli.zero_grad()
                
                # Only in the extended setting do we need participant information
                if self.setting == 'extended':
                    participant = torch.LongTensor(items.participant.values).to(self.device)
                else:
                    participant = None

                target = self.TARGET_TYPE(items.target.values)
                
                embedding = self.nli.embed(items)
                
                prediction, random_loss = self.nli(embedding, participant)
            
                fixed_loss = lossfunc(prediction, target)
                
                loss = fixed_loss + random_loss
                
                # Shouldn't this include the random loss? -B.K.
                # loss_trace.append(loss.item()-random_loss.item())
                loss_trace.append(loss.item())
                
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
                    print('epoch:              ', int(epoch))
                    print('batch:              ', int(batch))
                    print('mean loss:          ', np.round(np.mean(loss_trace), 2))
                    if self.data_type == 'categorical':
                        print('mean acc.:          ', np.round(np.mean(metric_trace), 2))
                    elif self.data_type == 'unit':
                        print('mean error:          ', np.round(np.mean(metric_trace), 2))
                    print('prop. best possible:', np.round(np.mean(best_trace), 2))
                    print()
                    
                    print()

                    loss_trace = []
                    metric_trace = []
                    best_trace = []
                
                loss.backward()
                optimizer.step()
                
        return self.nli.eval()

class UnitTrainer(NaturalLanguageInferenceTrainer):
    LOSS_CLASS = BCEWithLogitsLoss
    TARGET_TYPE = torch.FloatTensor
    OUTPUT_DIM = 1

class UnitRandomInterceptsNormalTrainer(UnitTrainer):
    MODEL_CLASS = UnitRandomInterceptsNormal

class UnitRandomInterceptsBetaTrainer(UnitTrainer):
    MODEL_CLASS = UnitRandomInterceptsBeta

class UnitRandomSlopesTrainer(UnitTrainer):
    MODEL_CLASS = UnitRandomSlopes

class CategoricalTrainer(NaturalLanguageInferenceTrainer):
    LOSS_CLASS = CrossEntropyLoss
    TARGET_TYPE = torch.LongTensor
    OUTPUT_DIM = 3

class CategoricalRandomInterceptsTrainer(CategoricalTrainer):
    MODEL_CLASS = CategoricalRandomIntercepts

class CategoricalRandomSlopesTrainer(CategoricalTrainer):
    MODEL_CLASS = CategoricalRandomSlopes