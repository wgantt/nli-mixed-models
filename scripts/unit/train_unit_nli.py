import argparse, json, torch
import numpy as np

from nli_mixed_models.trainers.nli_trainer import (
    UnitRandomInterceptsNormalTrainer,
    UnitRandomInterceptsBetaTrainer,
    UnitRandomSlopesTrainer
    )
from scripts.setup_logging import setup_logging
from scripts.training_utils import (
    parameter_grid,
    load_neg_raising,
    generate_splits,
    save_model_with_args
    )
from scripts.eval_utils import (
    eval_model
    )

LOG = setup_logging()

def main(args):

    # Load parameters from the supplied parameters file
    with open(args.parameters) as f:
        params = json.load(f)

    settings = params['settings']
    checkpoints = params['checkpoints']
    save_checkpoints = checkpoints['save_ckpts']
    checkpoint_dir = checkpoints['ckpt_dir'] 
    checkpoint_file_name = checkpoints['ckpt_file_name']

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

            # Only the random intercepts models have been modified to handle
            # the standard setting in addition to the extended one.
            if hyperparams['setting'] == 'extended' and settings['use_random_slopes']:
                unli_trainer = UnitRandomSlopesTrainer(**hyperparams)
            elif settings['use_beta_distribution']:
                unli_trainer = UnitRandomInterceptsBetaTrainer(**hyperparams)
            else:
                unli_trainer = UnitRandomInterceptsNormalTrainer(**hyperparams)

            LOG.info('...Complete')

            # Run the model
            # Using 'know' and 'think' just for debugging purposes;
            # actual training runs on all verbs
            if args.debug:
                LOG.info('Beginning training in debug mode...')
                unit_model = unli_trainer.fit(data=neg[neg.verb.isin(['know', 'think'])],\
                                                **trainparams)
                LOG.info('Finished training.')
            else:
                # Assign folds to each veridicality example
                k_folds = settings['k_folds']
                split_type = settings['split_type']
                setting = hyperparams['setting']
                neg = generate_splits(neg, split_type, k_folds=k_folds, datatype='n')
                LOG.info(f'Beginning training with {k_folds}-fold cross-validation in the {setting} setting, partitioning based on {split_type}.')

                # Perform k-fold cross-validation
                loss_all = []
                error_all = []
                best_all = []
                for i in range(k_folds):

                    # Select the folds
                    test_fold = i
                    train_folds = [j for j in range(k_folds) if j != test_fold]
                    LOG.info(f'Beginning training with test fold={test_fold}.')

                    train_data = neg[neg.fold.isin(train_folds)]
                    test_data = neg[neg.fold == test_fold]

                    # Fit the model on the train folds
                    unit_model = unli_trainer.fit(train_data=train_data, **trainparams)
                    LOG.info('Finished training.')

                    # Save the model
                    if save_checkpoints:
                        LOG.info('Saving model...')
                        save_model_with_args(params, unit_model, hyperparams, checkpoint_dir, checkpoint_file_name + '-fold-' + str(i))
                        LOG.info('Model saved.')

                    # Evaluate the model on the test fold
                    loss_mean, error_mean, best_mean = eval_model(test_data, unit_model, 'unit', trainparams['batch_size'])
                    loss_all.append(loss_mean)
                    error_all.append(error_mean)
                    best_all.append(best_mean)
                    LOG.info(f'Test results for fold {i}')
                    LOG.info(f'Mean loss:           {loss_mean}')
                    LOG.info(f'Mean error:       {error_mean}')
                    LOG.info(f'Prop. best possible: {best_mean}')

                LOG.info('Finished k-fold cross evaluation.')
                LOG.info(f'Mean error:           {np.round(np.mean(loss_all), 2)}')
                LOG.info(f'Mean error:       {np.round(np.mean(error_all), 2)}')
                LOG.info(f'Prop. best possible: {np.round(np.mean(best_all), 2)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a unit NLI mixed effects model')
    parser.add_argument('--parameters',
                        type=str,
                        default='scripts/unit/train_unit_nli.json',
                        help='Path to a JSON containing parameters to sweep')
    parser.add_argument('--debug',
                        action='store_true',
                        help='If provided, runs training on a small subset of'\
                        'the data, instead of doing full k-fold cross-validation.')
    args = parser.parse_args()
    main(args)
