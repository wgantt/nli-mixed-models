import torch
import numpy as np
import pandas as pd

from torch.optim import Adam
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from ..modules.nli import (
    UnitRandomIntercepts,
    UnitRandomSlopes,
    CategoricalRandomIntercepts,
    CategoricalRandomSlopes
)

class NaturalLanguageInferenceTrainer:
    
    def __init__(self, n_participants: int, 
                 embedding_dim: int = 768, 
                 n_predictor_layers: int = 2,
                 tied_covariance: bool = False,
                 device=torch.device('cpu')):
        self.embedding_dim = embedding_dim
        self.n_predictor_layers = n_predictor_layers
        self.n_participants = n_participants
        self.device = device
        self.tied_covariance = tied_covariance
        self.nli = self.MODEL_CLASS(embedding_dim, 
                                    n_predictor_layers,
                                    self.OUTPUT_DIM, 
                                    n_participants,
                                    tied_covariance,
                                    device)
    
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
            acc_trace = []
            best_trace = []
            
            for batch, items in data.groupby('batch_idx'):
                self.nli.zero_grad()
                
                participant = torch.LongTensor(items.participant.values).to(self.device)
                target = self.TARGET_TYPE(items.target.values)
                
                embedding = self.nli.embed(items)
                
                prediction, random_loss = self.nli(embedding, participant)
            
                fixed_loss = lossfunc(prediction, target)
                
                loss = fixed_loss + random_loss
                
                loss_trace.append(loss.item()-random_loss.item())
                
                if self.MODEL_CLASS is CategoricalRandomIntercepts or \
                   self.MODEL_CLASS is CategoricalRandomSlopes:
                    acc = (prediction.argmax(1) == target).data.cpu().numpy().mean()
                    best = (items.modal_response==items.target).mean()
                    
                    acc_trace.append(acc)
                    best_trace.append(acc/best)
                    
                elif self.MODEL_CLASS is UnitRandomIntercepts or \
                     self.MODEL_CLASS is UnitRandomSlopes:
                    acc = loss_trace[-1]
                    best = -(items.target.values * np.log(items.modal_response.values) +\
                             (1-items.target.values) * np.log(1-items.modal_response.values)).mean()
                    
                    acc_trace.append(acc)
                    best_trace.append(1 - (acc-best)/best)
                
                if not (batch % verbosity):
                    print('epoch:              ', int(epoch))
                    print('batch:              ', int(batch))
                    print('mean loss:          ', np.round(np.mean(loss_trace), 2))
                    print('mean acc.:          ', np.round(np.mean(acc_trace), 2))
                    print('prop. best possible:', np.round(np.mean(best_trace), 2))
                    print()
                    
                    print()

                    loss_trace = []
                    acc_trace = []
                    best_trace = []
                
                loss.backward()
                optimizer.step()
                
        return self.nli.eval()

class UnitTrainer(NaturalLanguageInferenceTrainer):
    LOSS_CLASS = BCEWithLogitsLoss
    TARGET_TYPE = torch.FloatTensor
    OUTPUT_DIM = 1

class UnitRandomInterceptsTrainer(UnitTrainer):
    MODEL_CLASS = UnitRandomIntercepts

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