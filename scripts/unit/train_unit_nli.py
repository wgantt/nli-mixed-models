import argparse, json, torch

from nli_mixed_models.trainers.nli_trainer import (
    UnitRandomInterceptsNormalTrainer,
    UnitRandomInterceptsBetaTrainer,
    UnitRandomSlopesTrainer
    )
from scripts.setup_logging import setup_logging
from scripts.training_utils import (
    parameter_grid,
    load_neg_raising
    )

LOG = setup_logging()

def main(args):

    # Load parameters from the supplied parameters file
    with open(args.parameters) as f:
        params = json.load(f)

    for hyperparams in parameter_grid(params["hyper"]):
        for trainparams in parameter_grid(params["training"]):

            # Load MegaVeridicality data
            LOG.info('Loading MegaNegRaising data...')
            neg = load_neg_raising()
            hyperparams['n_participants'] = neg.participant.unique().shape[0]
            LOG.info('...Complete.')

            # Initialize the model
            LOG.info('Initializing unit NLI model with the following hyperparameters:')
            LOG.info(json.dumps(hyperparams, indent=4))
            if params["use_random_slopes"]:
                unli_trainer = UnitRandomSlopesTrainer(**hyperparams)
            elif params["use_beta_distribution"]:
                unli_trainer = UnitRandomInterceptsBetaTrainer(**hyperparams)
            else:
                unli_trainer = UnitRandomInterceptsNormalTrainer(**hyperparams)
            LOG.info('...Complete')

            # Run the model
            LOG.info('Beginning training...')
            unit_model = unli_trainer.fit(data=neg[neg.verb.isin(['know', 'think'])], **trainparams)
            LOG.info('Finished training.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a unit NLI mixed effects model')
    parser.add_argument('--parameters',
                        type=str,
                        default='scripts/unit/train_unit_nli.json',
                        help='Path to a JSON containing parameters to sweep')
    # Currently will have no effect
    parser.add_argument('--checkpoints',
                        type=str,
                        default='checkpoints/',
                        help='Where to save the models to')
    args = parser.parse_args()
    main(args)
