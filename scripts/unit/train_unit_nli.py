import argparse, json, torch

from nli_mixed_models.trainers.nli_trainer import (
    UnitRandomInterceptsNormalTrainer,
    UnitRandomInterceptsBetaTrainer,
    UnitRandomSlopesTrainer
    )
from scripts.setup_logging import setup_logging
from scripts.training_utils import (
    parameter_grid,
    load_neg_raising,
    generate_splits
    )

LOG = setup_logging()

def main(args):

    # Load parameters from the supplied parameters file
    with open(args.parameters) as f:
        params = json.load(f)

    for hyperparams in parameter_grid(params['hyper']):
        for trainparams in parameter_grid(params['training']):

            # Load MegaVeridicality data
            LOG.info('Loading MegaNegRaising data...')
            neg = load_neg_raising()
            hyperparams['n_participants'] = neg.participant.unique().shape[0]
            LOG.info('...Complete.')

            # Initialize the model
            LOG.info('Initializing unit NLI model with the following hyperparameters:')
            LOG.info(json.dumps(hyperparams, indent=4))
            if hypparams['use_random_slopes']:
                unli_trainer = UnitRandomSlopesTrainer(**hyperparams)
            elif hyperparams['use_beta_distribution']:
                unli_trainer = UnitRandomInterceptsBetaTrainer(**hyperparams)
            else:
                unli_trainer = UnitRandomInterceptsNormalTrainer(**hyperparams)
            LOG.info('...Complete')

            # Run the model
            # Using 'know' and 'think' just for debugging purposes; actual training
            # will run on all verbs
            if args.debug:
                LOG.info('Beginning training in debug mode...')
                unit_model = unli_trainer.fit(data=neg[neg.verb.isin(['know', 'think'])],\
                                                **trainparams)
            else:
                # Assign folds to each veridicality example
                k_folds = trainparams['k_folds']
                split_type = trainparams['split_type']
                neg = generate_splits(neg, split_type, k_folds=k_folds, datatype='n')
                LOG.info(f'Beginning training with {k_folds}-fold cross-validation')

                # Perform k-fold cross-validation
                for i in range(k_folds):

                    # Select the folds
                    train_folds = [j for j in range(k_folds) if j != i]
                    train_data = neg[neg.fold.isin(train_folds)]
                    test_data = neg[~neg.fold.isin(train_folds)]

                    # Fit the model on the train folds
                    cat_model = cnli_trainer.fit(data=train_data, **trainparams)

                    # Evaluate the model on the test fold
                    # TODO: write eval function

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
    parser.add_argument('--debug',
                        action='store_true',
                        help='If provided, runs training on a small subset of'\
                        'the data, instead of doing full k-fold cross-validation.')
    args = parser.parse_args()
    main(args)
