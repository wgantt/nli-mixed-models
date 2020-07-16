import torch
import pandas as pd

from typing import Tuple
from torch import cat, flatten, sigmoid
from torch.nn import Module, ModuleList, Linear, ReLU, Sequential, Dropout, Parameter
from fairseq.data.data_utils import collate_tokens

from .nli_base import NaturalLanguageInference

class RandomInterceptsModel(NaturalLanguageInference):

    def __init__(self, embedding_dim: int, n_predictor_layers: int, hidden_dim: int,
                 output_dim: int, n_participants: int, setting: str,
                 use_sampling: bool, n_samples: int,
                 device=torch.device('cpu')):
        super().__init__(embedding_dim, n_predictor_layers, hidden_dim, output_dim,
                         n_participants, setting, use_sampling, n_samples, device)

        self.predictor = self._initialize_predictor()
        if self.setting == 'extended':
            self.random_effects = Parameter(self._initialize_random_effects())

    
    def _initialize_predictor(self):
        """Creates an MLP predictor with n predictor layers, ReLU activation,
           and a 0.5 dropout layer.
        """
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


    def forward(self, embeddings, participant=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Do a forward pass on the model. Returns a tuple (prediction, random_loss), where
           random_loss is a loss associated with the prior over the random components.
        """
        # Single MLP for all annotators
        fixed = self.predictor(embeddings.mean(1))
        
        # The "standard" setting, where we do not have access to annotator
        # information and thus do not have random effects.
        if self.setting == 'standard':
            random = None
            random_loss = 0.
        # The "extended" setting, where we have annotator random effects.
        else:
            random = self._random_effects()
            random_loss = self._random_loss(random)
        
        prediction = self._link_function(fixed, random, participant)
        return prediction, random_loss



class CategoricalRandomIntercepts(RandomInterceptsModel):

    def __init__(self, embedding_dim: int, n_predictor_layers: int, hidden_dim: int,
                 output_dim: int, n_participants: int, setting: str,
                 use_sampling: bool, n_samples: int,
                 device=torch.device('cpu')):
        super().__init__(embedding_dim, n_predictor_layers, hidden_dim, output_dim,
                         n_participants, setting, use_sampling,
                         n_samples, device=device)
        
        # The number of random effects per annotator is equivalent to the output dimension.
        self.mean = torch.zeros(self.output_dim)
        self.cov  = torch.zeros(((self.output_dim), (self.output_dim)))

    def _initialize_random_effects(self):
        """Initializes random effects - random intercept terms generated from a standard normal."""
        return torch.randn(self.n_participants, self.output_dim)
    

    def _random_effects(self):
        """Returns the mean-subtracted random effects of the model."""
        # Even in the random slopes case, I don't think it matters
        # whether we mean subtract or not here, since this is just used
        # to compute loss (and not to actually scale the fixed term, as
        # in UNLI).
        return self.random_effects - self.random_effects.mean(0)[None,:]
    

    def _link_function(self, fixed, random, participant):
        """Computes the link function for a given model configuration."""
        # Standard setting: no participants, so no random effects.
        if self.setting == 'standard' or random is None:
            return fixed
        # Extended setting subtask (b): assume mean annotator, so use mean
        # of random effects for prediction (should be zero in this case).
        elif participant is None:
            return fixed + random.mean(0)[None,:]
        # Extended setting subtask (a).
        else:
            return fixed + random[participant]


    def _random_loss(self, random):
        """Compute loss over random effect priors."""
        # TODO: decide whether to compute loss over all annotators at each iteration,
        # or only over a subset. Currently random loss is computed over ALL annotators
        # at each iteration. TBD whether this makes a difference or not. Regardless,
        # the covariance should always be estimated from all annotators.
        self.mean = torch.zeros(self.output_dim)
        self.cov = torch.matmul(torch.transpose(random, 1, 0), random) / (self.n_participants - 1)
        invcov = torch.inverse(self.cov)
        return torch.matmul(torch.matmul(random.unsqueeze(1), invcov), \
                            torch.transpose(random.unsqueeze(1), 1, 2)).mean(0)



class UnitRandomIntercepts(RandomInterceptsModel):

    def __init__(self, embedding_dim: int, n_predictor_layers: int, hidden_dim: int,
                 output_dim: int, n_participants: int, setting: str,
                 use_sampling: bool, n_samples: int,
                 device=torch.device('cpu')):
        super().__init__(embedding_dim, n_predictor_layers, hidden_dim, output_dim,
                         n_participants, setting, use_sampling,
                         n_samples, device=device)

        # The squashing function to bound continuous outputs to [0,1]
        self.squashing_function = sigmoid

        # A fixed shift term for calculating nu when parametrizing the beta distribution
        self.nu_shift = Parameter(torch.tensor([0.]))

        # There are two random effects per annotator (shift and variance)
        self.mean = torch.zeros(2)
        self.cov  = torch.zeros((2, 2))

        # Define a fixed shift term if standard setting
        if self.setting == 'standard':
            self.standard_shift = Parameter(torch.tensor([0.]))


    def _initialize_random_effects(self):
        # For the beta distribution, the random effects are:
        #   1) A random shifting term (positive or negative)
        #   2) The variance of the beta distribution itself (nonnegative)
        # The non-negativity of the variance is enforced in the link function
        return torch.randn(self.n_participants, 2)


    def _random_effects(self):
        return self.random_effects - self.random_effects.mean(0)[None,:]


    def _link_function(self, fixed, random, participant):
        # Standard setting
        if self.setting == 'standard' or random is None:

            # Fixed shift term is initialized to 0 in the standard setting,
            # but obviously will vary somewhat during training
            mean = self.squashing_function(fixed + self.standard_shift).squeeze(1)

            # Mu and nu used to calculate the parameters for the beta distribution
            mu = mean
            nu = torch.exp(self.nu_shift)

        # Extended setting subtask (b): assume mean annotator, so use mean
        # of random effects for prediction.
        elif participant is None:
            random_shift, random_variance = random[:,0], random[:,1]

            # This is the mean of the beta distribution
            mean = self.squashing_function(fixed + random_shift.mean(0)).squeeze(1)

            # Mu and nu used to calculate the parameters for the beta distribution
            mu = mean
            nu = torch.exp(self.nu_shift)

        # Extended setting subtask (a).
        else:
            random_shift, random_variance = random[:,0], random[:,1]

            # This is the mean of the beta distribution
            mean = self.squashing_function(fixed + random_shift[participant][:,None]).squeeze(1)

            # Mu and nu used to calculate the parameters for the beta distribution
            mu = mean
            nu = torch.exp(self.nu_shift + random_variance[participant])

        # Parameter estimates for the beta distribution. These are
        # estimated from the mean and variance.
        alpha = mu * nu
        beta = (1 - mu) * nu

        # The prediction is just the expected value for the beta
        # distribution whose parameters we've just estimated.
        return alpha, beta


    def _random_loss(self, random):

        # Parameter estimates for the prior
        self.mean = random.mean(0)
        self.cov = torch.matmul(torch.transpose(random, 1, 0), random) / (self.n_participants - 1)
        invcov = torch.inverse(self.cov)
        return torch.matmul(torch.matmul(random.unsqueeze(1), invcov), \
                            torch.transpose(random.unsqueeze(1), 1, 2)).mean(0)

