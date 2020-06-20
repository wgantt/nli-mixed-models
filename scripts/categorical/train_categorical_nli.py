import argparse, json, torch

from nli_mixed_models.trainers.nli_trainer import CategoricalNaturalLanguageInferenceTrainer
from scripts.setup_logging import setup_logging
from scripts.training_utils import (
    parameter_grid,
    load_veridicality
    )

LOG = setup_logging()

def main(args):

    with open(args.parameters) as f:
        params = json.load(f)

    print(parameter_grid(params['hyper']))
    # LOG.info('Loading MegaVeridicality Data...')
    # ver = load_veridicality()
    # LOG.info('...Complete.')

    # LOG.info('Initializing Categorical NLI model...')
    # n_participants = ver.participant.unique().shape[0]
    # cnli_trainer = CategoricalNaturalLanguageInferenceTrainer(n_participants=n_participants, use_random_slopes)
    # LOG.info('...Complete')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a categorical NLI mixed effects model')
    parser.add_argument('--parameters',
                        type=str,
                        default='train_categorical_nli.json',
                        help='Path to a JSON containing parameters to sweep')
    parser.add_argument('--checkpoints',
                        type=str,
                        default='checkpoints/',
                        help='Where to save the models to')
    args = parser.parse_args()
    main(args)
