import torch
import pandas as pd

from typing import Tuple
from torch import cat, flatten
from torch.nn import Module, ModuleList, Linear, ReLU, Sequential, Dropout
from torch.distributions.multivariate_normal import MultivariateNormal
from fairseq.data.data_utils import collate_tokens

from .nli_base import NaturalLanguageInference

class RandomInterceptsModel(NaturalLanguageInference):

    def __init__(self, embedding_dim: int, n_predictor_layers: int,  
                 output_dim: int, n_participants: int,
                 tied_covariance=False, device=torch.device('cpu')):
        super().__init__(embedding_dim, n_predictor_layers, output_dim,
                         n_participants, tied_covariance, device)

        self.predictor = self._initialize_predictor()
        self._initialize_random_effects()

    
    def _initialize_predictor(self, reduction_factor=2):
        """Creates an MLP predictor with n predictor layers, ReLU activation,
           and a 0.5 dropout layer.

           TODO: come up with a better name than 'reduction_factor', make it
           a model parameter.
        """
        seq = []
        
        prev_size = self.embedding_dim
        
        for l in range(self.n_predictor_layers):
            curr_size = int(prev_size/reduction_factor)
                
            seq += [Linear(prev_size,
                           curr_size),
                    ReLU(),
                    Dropout(0.5)]

            prev_size = curr_size

        seq += [Linear(prev_size,
                       self.output_dim)]
        
        return Sequential(*seq)


    def forward(self, embeddings, participant=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Do a forward pass on the model. Returns a tuple (prediction, random_loss), where
           random_loss is a loss associated with the prior over the random components.
        """
        # Single MLP for all annotators
        fixed = self.predictor(embeddings.mean(1))
        
        # The "standard" setting, where we do not have access to annotator
        # information and thus do not have random effects.
        if participant is None:
            random = None
            random_loss = 0.
        # The "extended" setting, where we have annotator random effects.
        else:
            random = self._random_effects()
            random_loss = self._random_loss(random)
        
        prediction = self._link_function(fixed, random, participant)
        return prediction, random_loss



class CategoricalRandomIntercepts(RandomInterceptsModel):

    def _initialize_random_effects(self):
        """Initializes random effects - random intercept terms generated from a standard normal."""
        self.random_effects = torch.randn(self.n_participants, self.output_dim, requires_grad=True)
    

    def _random_effects(self):
        """Returns the mean-subtracted random effects of the model."""
        # Even in the random slopes case, I don't think it matters
        # whether we mean subtract or not here, since this is just used
        # to compute loss (and not to actually scale the fixed term, as
        # in UNLI).
        return self.random_effects - self.random_effects.mean(0)[None,:]
    

    def _link_function(self, fixed, random, participant):
        """Computes the link function for a given model configuration."""
        return fixed + random[participant] if random is not None else fixed
    

    def _random_loss(self, random):
        """Compute loss over random effect priors."""
        # TODO: decide whether to compute loss over all annotators at each iteration,
        # or only over a subset. Currently random loss is computed over ALL annotators
        # at each iteration. TBD whether this makes a difference or not. Regardless,
        # the covariance should always be estimated from all annotators.
        if self.tied_covariance:
            return torch.mean(torch.square(random/random.std(0)[None,:]))
        else:
            mean = torch.zeros(self.output_dim)
            cov = torch.matmul(torch.transpose(random, 1, 0), random) / (self.n_participants - 1)
            return -torch.mean(MultivariateNormal(mean, cov).log_prob(random)[None,:])



class UnitRandomIntercepts(RandomInterceptsModel):
    
    # TODO: Ben Kane - Convert to using beta distribution
    def _initialize_random_effects(self):
        self.random_effects = torch.randn(self.n_participants, 2)
        

    def _random_effects(self):
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