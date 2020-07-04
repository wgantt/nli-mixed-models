import argparse, json, torch

from nli_mixed_models.trainers.nli_trainer import (
    CategoricalRandomInterceptsTrainer,
    CategoricalRandomSlopesTrainer
    )
from scripts.setup_logging import setup_logging
from scripts.training_utils import (
    parameter_grid,
    load_veridicality,
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
            LOG.info('Loading MegaVeridicality data...')
            ver = load_veridicality()
            hyperparams['n_participants'] = ver.participant.unique().shape[0]
            LOG.info('...Complete.')

            # Initialize the model
            LOG.info('Initializing categorical NLI model with the following hyperparameters:')
            LOG.info(json.dumps(hyperparams, indent=4))
            if trainparams['use_random_slopes']:
                del hyperparams['use_random_slopes']
                cnli_trainer = CategoricalRandomSlopesTrainer(**hyperparams)
            else:
                cnli_trainer = CategoricalRandomInterceptsTrainer(**hyperparams)
            LOG.info('...Complete')

            # Run the model
            # Using 'know' and 'think' just for testing purposes; actual training should
            # probably run on all verbs
            if args.debug:
                LOG.info('Beginning training in debug mode...')
                cat_model = cnli_trainer.fit(data=ver[ver.verb.isin(['know', 'think'])],\
                                             **trainparams)
            else:
                # Assign folds to each veridicality example
                k_folds = trainparams['k_folds']
                split_type = trainparams['split_type']
                ver = generate_splits(ver, split_type, k_folds=k_folds, datatype='v')
                LOG.info(f'Beginning training with {k_folds}-fold cross-validation')

                # Perform k-fold cross-validation
                for i in range(k_folds):

                    # Select the folds
                    train_folds = [j for j in range(k_folds) if j != i]
                    train_data = ver[ver.fold.isin(train_folds)]
                    test_data = ver[~ver.fold.isin(train_folds)]
                    print(len(train_data), len(test_data))

                    # Fit the model on the train folds
                    # cat_model = cnli_trainer.fit(data=train_data, **trainparams)

                    # Evaluate the model on the test fold
                    # TODO: write eval function

            LOG.info('Finished training.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a categorical NLI mixed effects model')
    parser.add_argument('--parameters',
                        type=str,
                        default='scripts/categorical/train_categorical_nli.json',
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
