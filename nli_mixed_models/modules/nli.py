import torch
import pandas as pd

from typing import Tuple
from torch.nn import Module, ModuleList, Linear, ReLU, Sequential, Dropout
from torch.distributions.multivariate_normal import MultivariateNormal
from fairseq.data.data_utils import collate_tokens
print(torch)

class NaturalLanguageInference(Module):
    
    def __init__(self, embedding_dim: int, n_predictor_layers: int,  
                 output_dim: int, n_participants: int,
                 use_random_slopes=False, tied_covariance=False, device=torch.device('cpu')):
        super().__init__()
        
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        self.roberta.eval()
        self.embedding_dim = embedding_dim
        self.n_predictor_layers = n_predictor_layers
        self.output_dim = output_dim
        self.n_participants = n_participants
        self.tied_covariance = tied_covariance
        self.use_random_slopes = use_random_slopes
        self.device = device
        
        self._initialize_random_effects()
        
        # TODO: comment
        if self.use_random_slopes:
            self.predictor = self._initialize_predictor_for_random_slopes(self.n_participants)
        else:
            self.predictor = self._initialize_predictor()
     
    def _initialize_predictor(self):
        seq = []
        
        prev_size = self.embedding_dim
        
        for l in range(self.n_predictor_layers):
            curr_size = int(prev_size/2)
                
            seq += [Linear(prev_size,
                           curr_size),
                    ReLU(),
                    Dropout(0.5)]

            prev_size = curr_size

        seq += [Linear(prev_size,
                       self.output_dim)]
        
        return Sequential(*seq)

    def _initialize_predictor_for_random_slopes(self, n_participants):
        # Separate MLP for each annotator. We assume the annotator IDs
        # are zero-indexed and range up to n_participants.
        heads = []
        for _ in range(n_participants):
            seq = []
            prev_size = self.embedding_dim
        
            for l in range(self.n_predictor_layers):
                curr_size = int(prev_size/2)

                seq += [Linear(prev_size,
                               curr_size),
                        ReLU(),
                        Dropout(0.5)]

                prev_size = curr_size

            seq += [Linear(prev_size,
                           self.output_dim)]
            
            heads.append(Sequential(*seq))
            
        return ModuleList(heads)
        
    def forward(self, embeddings, participant=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_random_slopes:
            # Not sure this is right, but given that the *entire* prediction head
            # is random, I don't know what would count as the fixed component.
            # Maybe just the embedding?
            fixed = None
            
            # In the random slopes setting, 'predictor' is actually a Tensor of 
            # n_participants MLPs, one for each participant. Couldn't figure out
            # a way to vectorize this, unfortunately. This should probably go in
            # the _random_effects method of a CategoricalNaturalLanguageInferenceRandomSlopes
            # class.
            random = torch.stack([self.predictor[p](e.mean(0)) for p, e in zip(participant, embeddings)], dim=0)
            random_loss = self._random_loss(self._random_effects(participant))
            
        else:
            fixed = self.predictor(embeddings.mean(1))
        
            if participant is None:
                random = None
                random_loss = 0.
            else:
                random = self._random_effects(participant)
                random_loss = self._random_loss(random)
        
        prediction = self._link_function(fixed, random, participant)
        
        return prediction, random_loss

    def embed(self, items: pd.DataFrame) -> torch.Tensor:
        texts, hypotheses = items.sentence.values, items.hypothesis.values
                
        token_ids = collate_tokens([self.roberta.encode(t, h) 
                                    for t, h in zip(texts, hypotheses)], 
                                   pad_idx=1)

        with torch.no_grad():
            embedding = self.roberta.extract_features(token_ids)
            
        return embedding

class CategoricalNaturalLanguageInference(NaturalLanguageInference):
    
    def _initialize_random_effects(self):
        self.random_effects = torch.randn(self.n_participants, self.output_dim)
    
    # Not sure why "participant" is a parameter here
    def _random_effects(self, participant):
        # No mean subtraction for random slopes
        if self.use_random_slopes:
            return self.random_effects
        else:
            return self.random_effects - self.random_effects.mean(0)[None,:]
    
    def _link_function(self, fixed, random, participant):
        # Random slopes + random intercepts case
        if fixed is None:
            return random
        # Fixed effects (standard setting)
        elif random is None:
            return fixed
        # Random intercepts alone (extended setting)
        else:
            return fixed + random[participant]
    
    def _random_loss(self, random):
        # TODO: Handle random slopes for tied covariance? May not be worth it.
        if self.tied_covariance:
            return torch.mean(torch.square(random/random.std(0)[None,:]))
        else:
            if self.use_random_slopes:
                mean = random.mean(0)
                # This is currently incorrect
                cov = torch.matmul(torch.transpose(random, 1, 0), random) / (self.n_participants - 1)
            # Random intercepts only: mean is zero
            else:
                mean = torch.zeros(self.n_participants, self.output_dim)
                cov = torch.matmul(torch.transpose(random, 1, 0), random) / (self.n_participants - 1)
            return torch.mean(MultivariateNormal(mean, cov).log_prob(random)[None,:])

class UnitNaturalLanguageInference(NaturalLanguageInference):
    
    # TODO: Convert to using beta distribution
    def _initialize_random_effects(self):
        self.random_effects = torch.randn(self.n_participants, 2)
        
    def _random_effects(self, participant):
        random_scale = torch.square(self.random_effects[:,0])
        random_shift = self.random_effects[:,1] - self.random_effects[:,1].mean(0)
        
        return random_scale, random_shift
    
    def _link_function(self, fixed, random, participant):
        if random is None:
            return torch.square(self.random_effects[:,0]).mean()*fixed.squeeze(1)
        else:
            random_scale, random_shift = random
            return (random_scale[participant][:,None]*fixed +\
                    random_shift[participant][:,None]).squeeze(1)
    
    def _random_loss(self, random):
        random_scale, random_shift = random
        
        random_scale_loss = torch.mean(torch.square(random_scale/random_scale.mean(0)))
        random_shift_loss = torch.mean(torch.square(random_shift/random_shift.std(0)))
        
        return (random_scale_loss + random_shift_loss)/2