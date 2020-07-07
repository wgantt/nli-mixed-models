import torch
import pandas as pd

from typing import Tuple
from torch import cat, flatten
from torch.nn import Module, ModuleList, Linear, ReLU, Sequential, Dropout, Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from fairseq.data.data_utils import collate_tokens

from .nli_base import NaturalLanguageInference

class RandomSlopesModel(NaturalLanguageInference):

    def __init__(self, embedding_dim: int, n_predictor_layers: int,  
                 output_dim: int, n_participants: int, setting: str,
                 device=torch.device('cpu')):
        super().__init__(embedding_dim, n_predictor_layers, output_dim,
                         n_participants, setting, device)
        
        self.predictor_base, self.predictor_heads = self._initialize_predictor(self.n_participants)
        self.random_effects = Parameter(self._initialize_random_effects())


    def _initialize_predictor(self, n_participants, hidden_dim=128):
        """Creates MLP predictors that has annotator-specific
           final layers. Uses ReLU activation and a 0.5 dropout layer.

           TODO: make hidden_dim a model parameter. Note that it must be
           less than n_participants to avoid having a singular covariance
           matrix when computing the prior over the predictor head weights. 
        """

        # Shared base
        predictor_base = Sequential(Linear(self.embedding_dim, hidden_dim),
                                    ReLU(),
                                    Dropout(0.5))
    
        # Annotator-specific final layers
        predictor_heads = ModuleList([Linear(hidden_dim,
                       self.output_dim) for _ in range(n_participants)])

        return predictor_base, predictor_heads


    def _extract_random_slopes_params(self):
        """Assuming a random slopes model, extract and flatten the parameters
           the final linear layers in each participant MLP.
        """
        # Iterate over annotator-specific heads
        random_effects = []
        for i in range(self.n_participants):
            weight, bias = self.predictor_heads[i].weight, self.predictor_heads[i].bias.unsqueeze(1)
            flattened = flatten(cat([weight, bias], dim=1))
            random_effects.append(flattened)

        # Return (n_participants, flattened_head_dim)-shaped tensor
        return torch.stack(random_effects)

    
    def forward(self, embeddings, participant=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Do a forward pass on the model. Returns a tuple (prediction, random_loss), where
           random_loss is a loss associated with the prior over the random components
        """
        # Shared base MLP with annotator-specific final linear layers. Not sure whether
        # there's a clever way to vectorize this.
        predictions = []
        for p, e in zip(participant, embeddings):
            predictions.append(self.predictor_heads[p](self.predictor_base(e.mean(0))))

        # 'fixed' is obviously something of a misnomer here, given that it's
        # just computed from the separate annotator MLPs. 
        predictions = torch.stack(predictions, dim=0)

        # There is, however, still a random loss, which is just the prior over
        # the (flattened) weights and biases of the MLPs.
        random_loss = self._random_loss(self._random_effects())

        # Return the prediction and the random loss
        predictions = self._link_function(predictions)
        return predictions, random_loss



class CategoricalRandomSlopes(RandomSlopesModel):

    def _initialize_random_effects(self):
        """Initializes random effects as the MLP parameters."""

        # shape = n_participants x len(flattened MLP weights + biases)
        return self._extract_random_slopes_params()
    

    def _random_effects(self):
        """Returns the mean-subtracted random effects of the model."""

        # Even in the random slopes case, I don't think it matters
        # whether we mean subtract or not here, since this is just used
        # to compute loss (and not to actually scale the fixed term, as
        # in UNLI).
        return self.random_effects - self.random_effects.mean(0)[None,:]
    

    def _link_function(self, predictions):
        """Computes the link function for a given model configuration."""

        # 'predictions' contains the outputs from the individual annotator MLPs,
        # which have the random slopes and intercepts embedeed within them,
        # so there are no separate 'random' and 'fixed' components.
        return predictions
    

    def _random_loss(self, random):
        """Compute loss over random effect priors."""

        # TODO: decide whether to compute loss over all annotators at each iteration,
        # or only over a subset. Currently random loss is computed over ALL annotators
        # at each iteration. TBD whether this makes a difference or not. Regardless,
        # the covariance should always be estimated from all annotators.
        mean = torch.zeros(random.shape[1])
        cov = torch.matmul(torch.transpose(random, 1, 0), random) / (self.n_participants - 1)
        invcov = torch.inverse(cov)
        return torch.matmul(torch.matmul(random.unsqueeze(1), invcov), \
                            torch.transpose(random.unsqueeze(1), 1, 2)).mean(0)



class UnitRandomSlopes(RandomSlopesModel):
    
    def _initialize_random_effects(self):
        return self._extract_random_slopes_params()
        

    def _random_effects(self):
        return self.random_effects - self.random_effects.mean(0)[None,:]
    

    def _link_function(self, predictions):
        return predictions.squeeze(1)
    

    def _random_loss(self, random):
        mean = torch.zeros(random.shape[1])
        cov = torch.matmul(torch.transpose(random, 1, 0), random) / (self.n_participants - 1)
        invcov = torch.inverse(cov)
        return torch.matmul(torch.matmul(random.unsqueeze(1), invcov), \
                    torch.transpose(random.unsqueeze(1), 1, 2)).mean(0)