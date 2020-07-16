import torch
import pandas as pd

from typing import Tuple
from torch import cat, flatten, sigmoid
from torch.nn import Module, ModuleList, Linear, ReLU, Sequential, Dropout, Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from fairseq.data.data_utils import collate_tokens

from .nli_base import NaturalLanguageInference

class RandomSlopesModel(NaturalLanguageInference):

    def __init__(self, embedding_dim: int, n_predictor_layers: int, hidden_dim: int,
                 output_dim: int, n_participants: int, setting: str,
                 use_sampling: bool, n_samples: int,
                 device=torch.device('cpu')):
        super().__init__(embedding_dim, n_predictor_layers, hidden_dim, output_dim,
                         n_participants, setting, use_sampling, n_samples, device)

        self.predictor_base, self.predictor_heads = self._initialize_predictor(self.n_participants)


    def _initialize_predictor(self, n_participants):
        """Creates MLP predictors that has annotator-specific
           final layers. Uses ReLU activation and a 0.5 dropout layer.
           NOTE: hidden_dim must be less than n_participants to avoid
           having a singular covariance matrix when computing the prior
           over the predictor head weights. 
           TODO: the above should probably be checked during initialization,
           and an informative error returned if the condition fails.
        """

        # Shared base
        predictor_base = Sequential(Linear(self.embedding_dim, self.hidden_dim),
                                    ReLU(),
                                    Dropout(0.5))
    
        # Annotator-specific final layers
        predictor_heads = ModuleList([Linear(self.hidden_dim,
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


    def _create_mean_predictor(self):
      """Creates a new predictor using the mean parameters across the predictor heads."""
      weights = []
      biases = []
      # Collect weights/biases from each predictor head and create tensors
      for i in range(self.n_participants):
        weights.append(self.predictor_heads[i].weight)
        biases.append(self.predictor_heads[i].bias)
      weights = torch.stack(weights)
      biases = torch.stack(biases)
      # Create new linear predictor and set weights/biases to means
      predictor_heads_mean = Linear(self.hidden_dim, self.output_dim)
      predictor_heads_mean.weight = Parameter(weights.mean(0))
      predictor_heads_mean.bias = Parameter(biases.mean(0))
      return predictor_heads_mean

    
    def forward(self, embeddings, participant=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Do a forward pass on the model. Returns a tuple (prediction, random_loss), where
           random_loss is a loss associated with the prior over the random components
        """
        # Shared base MLP with annotator-specific final linear layers. Not sure whether
        # there's a clever way to vectorize this.
        predictions = []
        # Extended setting subtask (b): assume a mean annotator, so create a new predictor
        # head using the mean parameters across the predictor heads.
        # NOTE: only used in eval mode - cannot be used for training since autograd will not work.
        if participant is None:
          predictor_heads_mean = self._create_mean_predictor()
          for e in embeddings:
            predictions.append(predictor_heads_mean(self.predictor_base(e.mean(0))))
        # Extended setting subtask (a).
        else:
          for p, e in zip(participant, embeddings):
            predictions.append(self.predictor_heads[p](self.predictor_base(e.mean(0))))

        # 'fixed' is obviously something of a misnomer here, given that it's
        # just computed from the separate annotator MLPs. 
        predictions = torch.stack(predictions, dim=0)

        # There is, however, still a random loss, which is just the prior over
        # the (flattened) weights and biases of the MLPs.
        random_loss = self._random_loss(self._random_effects())

        # Return the prediction and the random loss
        predictions = self._link_function(predictions, participant)
        return predictions, random_loss



class CategoricalRandomSlopes(RandomSlopesModel):

    def __init__(self, embedding_dim: int, n_predictor_layers: int, hidden_dim: int, 
                 output_dim: int, n_participants: int, setting: str,
                 use_sampling: bool, n_samples: int,
                 device=torch.device('cpu')):
        super().__init__(embedding_dim, n_predictor_layers, hidden_dim, output_dim,
                         n_participants, setting, use_sampling,
                         n_samples, device)

        # The number of random effects per annotator is equal to the hidden dimension
        # times the output dimension (weights) plus the hidden dimension (bias).
        n_random_effects = ((self.hidden_dim + 1) * self.output_dim)
        self.mean = torch.zeros(n_random_effects)
        self.cov  = torch.zeros((n_random_effects, n_random_effects))

        self.random_effects = self._initialize_random_effects()

    def _initialize_random_effects(self):
        """Initializes random effects as the MLP parameters."""

        # shape = n_participants x len(flattened MLP weights + biases)
        return self._random_effects()
    

    def _random_effects(self):
        """Returns the mean-subtracted random effects of the model.

        Even in the random slopes case, I don't think it matters
        whether we mean subtract or not here, since this is just used
        to compute loss (and not to actually scale the fixed term, as
        in UNLI). In any case, the weights have to be extracted anew
        at each iteration, since they will have been updated.
        """
        self.random_effects = self._extract_random_slopes_params()
        return self.random_effects - self.random_effects.mean(0)[None,:]
    

    def _link_function(self, predictions, participant):
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
        self.mean = torch.zeros(random.shape[1])
        self.cov = torch.matmul(torch.transpose(random, 1, 0), random) / (self.n_participants - 1)
        # mean: torch.Size([387])
        # cov:  torch.Size([387, 387])
        invcov = torch.inverse(self.cov)
        return torch.matmul(torch.matmul(random.unsqueeze(1), invcov), \
                            torch.transpose(random.unsqueeze(1), 1, 2)).mean(0)



class UnitRandomSlopes(RandomSlopesModel):

    def __init__(self, embedding_dim: int, n_predictor_layers: int, hidden_dim: int,
                 output_dim: int, n_participants: int, setting: str,
                 use_sampling: bool, n_samples: int,
                 device=torch.device('cpu')):
        super().__init__(embedding_dim, n_predictor_layers, hidden_dim, output_dim,
                         n_participants, setting, use_sampling,
                         n_samples, device)

        # The number of random effects per annotator is equal to the hidden dimension
        # times the output dimension (weights) plus the hidden dimension (bias), plus
        # random variance.
        n_random_effects = ((self.hidden_dim + 1) * self.output_dim) + 1
        self.mean = torch.zeros(n_random_effects)
        self.cov  = torch.zeros((n_random_effects, n_random_effects))

        self.squashing_function = sigmoid
        self._initialize_random_effects()

        # A fixed shift term for calculating nu when parametrizing the beta distribution
        self.nu_shift = Parameter(torch.tensor([0.]))
        

    def _initialize_random_effects(self):
        # The random effects in the unit random slopes model consist not
        # only of the annotator MLP weights, but also of a random variance
        # term, just as with the unit random intercepts model.
        self.weights = self._extract_random_slopes_params()
        self.weights -= self.weights.mean(0)[None,:]
        variance = Parameter(torch.randn(self.n_participants))
        self.variance = (variance - variance.mean()).to(self.device)
        return self.weights, variance


    def _random_effects(self):
        # Weights must be extracted anew each time from the regression heads
        self.weights = self._extract_random_slopes_params()
        self.weights -= self.weights.mean(0)[None,:]
        return self.weights, self.variance - self.variance.mean()
    

    def _link_function(self, predictions, participant):
        # Same link function as for unit random intercepts, except that we
        # feed the outputs of the annotator-specific regression heads to the
        # squashing function.
        mean = self.squashing_function(predictions).squeeze(1)

        # Mu and nu used to calculate the parameters for the beta distribution
        # Extended setting subtask (b): use mean variance across participants.
        if participant is None:
          mu = mean
          nu = torch.exp(self.nu_shift)
        # Extended setting subtask (a).
        else:
          mu = mean
          nu = torch.exp(self.nu_shift + self.variance[participant])

        # Parameter estimates for the beta distribution. These are
        # estimated from the mean and variance.
        alpha = mu * nu
        beta = (1 - mu) * nu
        return alpha, beta
    

    def _random_loss(self, random):
        # Joint multivariate normal log prob loss over *all* random
        # effects (weights + variance)
        weights, variance = random
        combined = torch.cat([weights, variance.unsqueeze(1)], dim=1)
        self.mean = combined.mean(0)
        self.cov = torch.matmul(torch.transpose(combined, 1, 0), combined) / (self.n_participants - 1)
        invcov = torch.inverse(self.cov)
        return torch.matmul(torch.matmul(combined.unsqueeze(1), invcov), \
                    torch.transpose(combined.unsqueeze(1), 1, 2)).mean(0)

    

