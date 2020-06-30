import torch
import pandas as pd

from typing import Tuple
from torch import cat, flatten
from torch.nn import Module, ModuleList, Linear, ReLU, Sequential, Dropout
from torch.distributions.multivariate_normal import MultivariateNormal
from fairseq.data.data_utils import collate_tokens

class NaturalLanguageInference(Module):
    
    def __init__(self, embedding_dim: int, n_predictor_layers: int,  
                 output_dim: int, n_participants: int,
                 tied_covariance=False, use_random_slopes=False, device=torch.device('cpu')):
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
        
        # TODO: comment
        if self.use_random_slopes:
            self.predictor = self._initialize_predictor_for_random_slopes(self.n_participants)
        else:
            self.predictor = self._initialize_predictor()

        self._initialize_random_effects()
     

    def _initialize_predictor(self):
        """Creates an MLP predictor with n predictor layers, ReLU activation,
           and a 0.5 dropout layer."""
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
        """Creates a seperate MLP predictor for each annotator (assuming annotator
           IDs are zero-indexed and range up to n_participants). Uses ReLU activation
           and a 0.5 dropout layer."""
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


    def _extract_random_slopes_params(self):
        """Assuming a random slopes model, extract and flatten the parameters of the
           linear layers in each participant MLP."""
        # Iterate over annotator MLPs
        random_effects = []
        for i in range(self.n_participants):
            flattened = None
            for layer in self.predictor[i]:
                # Collect all weights and biases from the MLP and flatten them
                if isinstance(layer, Linear):
                    if flattened == None:
                        flattened = flatten(cat([layer.weight, layer.bias.unsqueeze(1)], dim=1))
                    else:
                        flattened = cat([flattened, flatten(cat([layer.weight, layer.bias.unsqueeze(1)], dim=1))])
            random_effects.append(flattened)

        # Return (n_participants, flattened_mlp_dim)-shaped tensor
        return torch.stack(random_effects)


    def forward(self, embeddings, participant=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Do a forward pass on the model. Returns a tuple (prediction, random_loss), where
           random_loss is a loss associated with the prior over the random components."""
        # Random slopes
        if self.use_random_slopes:
            # In the random slopes setting, 'predictor' is actually a Tensor of 
            # n_participants MLPs, one for each participant. Couldn't figure out
            # a way to vectorize this, unfortunately.
            fixed = torch.stack([self.predictor[p](e.mean(0)) for p, e in zip(participant, embeddings)], dim=0)

            # The annotator MLPs contain the random slopes and intercepts, which are
            # used to generate the predictions above. So no separate term here to use
            # in the link function.
            random = None

            # There is, however, still a random loss, which is just the prior over
            # the (flattened) weights and biases of the MLPs.
            random_loss = self._random_loss(self._random_effects())
            
        # Standard setting + random intercepts
        else:
            # In the random intercepts setting, we have a single MLP
            # for all annotators.
            fixed = self.predictor(embeddings.mean(1))
        
            # The "standard" setting, where we do not have access to annotator
            # information and thus do not have random effects.
            if participant is None:
                random = None
                random_loss = 0.
            # The "extended" setting, where we have annotator random effects
            else:
                random = self._random_effects()
                random_loss = self._random_loss(random)
        
        prediction = self._link_function(fixed, random, participant)
        
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



class CategoricalNaturalLanguageInference(NaturalLanguageInference):
    
    def _initialize_random_effects(self):
        """Initializes random effects - random intercept terms in the case of the random intercepts
           model, and the MLP parameters in the case of random slopes."""
        # For random slopes, the random slope and intercept terms are
        # just the weights and biases of the MLPs.
        if self.use_random_slopes:
            # shape = n_participants x len(flattened MLP weights + biases)
            self.random_effects = self._extract_random_slopes_params()
        # For random intercepts, the intercepts terms are generated
        # separately from a standard normal
        else:
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
        # Standard setting + random slopes. In random slopes, 'fixed' contains
        # the outputs from the individual annotator MLPs, which have the random
        # slopes and intercepts embedeed within them, so there's no separate
        # random component.
        if random is None or self.use_random_slopes:
            return fixed
        # Extended setting
        else:
            return fixed + random[participant]
    

    def _random_loss(self, random):
        """Compute loss over random effect priors."""
        # TODO: decide whether to compute loss over all annotators at each iteration,
        # or only over a subset. Currently random loss is computed over ALL annotators
        # at each iteration. TBD whether this makes a difference or not. Regardless,
        # the covariance should always be estimated from all annotators.
        if self.tied_covariance:
            return torch.mean(torch.square(random/random.std(0)[None,:]))
        else:
            # The way things are currently implemented, the mean of 'random'
            # will be zero in both the random intercepts and random slopes
            # cases, since we subtract off the true mean before calling this method.
            mean = torch.zeros(self.n_participants, self.output_dim)
            print('BEFORE FREEZE')
            test = torch.transpose(random, 1, 0)
            print('transp:\n > shape: %s\n > value: %s' % (test.shape, test))
            print('\nrandom:\n > shape: %s\n > value: %s' % (random.shape, random))
            print('Seems to freeze here: %s' % torch.matmul(test, random))
            cov = torch.matmul(torch.transpose(random, 1, 0), random) / (self.n_participants - 1)
            print('AFTER FREEZE')
            return -torch.mean(MultivariateNormal(mean, cov).log_prob(random)[None,:])



class UnitNaturalLanguageInference(NaturalLanguageInference):
    
    # TODO: Ben Kane - Convert to using beta distribution
    def _initialize_random_effects(self):
        if self.use_random_slopes:
            self.random_effects = self._extract_random_slopes_params()
        else:
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