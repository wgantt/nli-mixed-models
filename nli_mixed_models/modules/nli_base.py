import torch
import pandas as pd

from typing import Tuple
from torch import cat, flatten
from torch.nn import Module, ModuleList, Linear, ReLU, Sequential, Dropout
from torch.distributions.multivariate_normal import MultivariateNormal
from fairseq.data.data_utils import collate_tokens

class NaturalLanguageInference(Module):
    
    def __init__(self, embedding_dim: int, n_predictor_layers: int, hidden_dim: int,
                 output_dim: int, n_participants: int, setting: str,
                 use_sampling: bool, n_samples: int,
                 device=torch.device('cpu')):
        super().__init__()
        
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        self.roberta.eval()
        self.embedding_dim = embedding_dim
        self.n_predictor_layers = n_predictor_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_participants = n_participants
        self.setting = setting
        self.use_sampling = use_sampling
        self.n_samples = n_samples
        self.device = device


    def forward(self, embeddings, participant=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Abstract method to be overridden by subclasses."""
        prediction, random_loss = None, None
        return prediction, random_loss


    def embed(self, items: pd.DataFrame) -> torch.Tensor:
        """Creates text+hypothesis embeddings for each item using RoBERTa."""
        texts, hypotheses = items.sentence.values, items.hypothesis.values
                
        token_ids = collate_tokens([self.roberta.encode(t, h) 
                                    for t, h in zip(texts, hypotheses)], 
                                   pad_idx=1)

        with torch.no_grad():
            embedding = self.roberta.extract_features(token_ids)
            
        return embedding