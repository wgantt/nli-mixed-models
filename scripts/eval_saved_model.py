import os
import argparse
import torch
import json
import numpy as np
import pandas as pd

from scripts.training_utils import load_model_with_missing_hyperparams
from scripts.setup_logging import setup_logging

from nli_mixed_models.modules.nli_random_intercepts import (
    UnitRandomIntercepts,
    CategoricalRandomIntercepts
)
from nli_mixed_models.modules.nli_random_slopes import (
    UnitRandomSlopes,
    CategoricalRandomSlopes
)
from nli_mixed_models.eval.nli_eval import (
    UnitEval,
    CategoricalEval
)

CAT_OUTPUT_DIM = 3
UNIT_OUTPUT_DIM = 1
NUM_FOLDS = 5
BATCH_SIZE = 128

LOG = setup_logging()

def load_old_style_model(ckpt_path, device):
	"""Loads a model that was saved in a now-obsolete format

	Previously, our models did not have the mean and covariance of the
	random effects saved as instance variables. This function loads
	such a model, adding those instance variables along with it.
	"""

	# Determine the type of model
	if 'categorical' in ckpt_path:
		if 'random_intercepts' in ckpt_path:
			model_cls = CategoricalRandomIntercepts
		else:
			model_cls = CategoricalRandomSlopes
	elif 'unit' in ckpt_path:
		if 'random_intercepts' in ckpt_path:
			model_cls = UnitRandomIntercepts
		else:
			model_cls = UnitRandomSlopes
	else:
		raise ValueError(f'Unknown model type {model_type}!')
	
	# Determine the model output dimension
	output_dim = CAT_OUTPUT_DIM if 'categorical' in ckpt_path else UNIT_OUTPUT_DIM

	# Required model constructor parameters that are missing for some
	# reason from the models I saved.
	missing_hyperparams = {
		'hidden_dim': 128,
		'output_dim': output_dim,
		'use_sampling': True,
		'n_samples': 10000
	}

	# Load the model with the missing hyperparameters
	model, hyper_params = load_model_with_missing_hyperparams(model_cls, ckpt_path, missing_hyperparams)

	# Estimate the mean and the covariance from the saved random effects
	random = model._random_effects()
	model = model.to(torch.device(device))
	model.mean = random.mean(0).to(torch.device(device))
	model.cov = (torch.matmul(torch.transpose(random, 1, 0), random) / (model.n_participants - 1)).to(torch.device(device))
	# model.cov = model.cov.to(device)
	return model, hyper_params

def main(args):
	ckpt_path = args.model_ckpt_path
	model_type = 'categorical' if 'categorical' in ckpt_path else 'unit'
	data_file_name = ckpt_path + '/' + '-'.join([model_type, 'partition', args.partition, 'data']) + '.csv'
	data = pd.read_csv(data_file_name)

	fixed_loss_all = []
	metric_all = []
	best_all = []
	for test_fold in range(NUM_FOLDS):
		ckpt_file_name = ckpt_path + '/' + '-'.join([model_type, 'fold', str(test_fold), 'partition', args.partition]) + '.pt'
		model, hyperparams = load_old_style_model(ckpt_file_name, args.device)
		LOG.info(f'Evaluating {model.__class__.__name__} model on the {args.partition} partition with the following hyperparameters:')
		LOG.info(json.dumps(hyperparams, indent=4))
		test_data = data[data.fold == test_fold]
		if model_type == 'categorical':
			nli_eval = CategoricalEval(model, 'b', device=args.device)
		else:
			nli_eval = UnitEval(model, 'b', device=args.device)
		loss_mean, fixed_loss_mean, random_loss_mean, metric_mean, best_mean, spearman_mean = \
			nli_eval.eval(test_data, BATCH_SIZE)
		fixed_loss_all.append(fixed_loss_mean)
		metric_all.append(metric_mean)
		best_all.append(best_mean)
		LOG.info(f'Test results for fold {test_fold}, subtask b')
		LOG.info(f'Mean fixed loss:     {fixed_loss_mean}')
		LOG.info(f'Mean metric:         {metric_mean}')
		LOG.info(f'Prop. best possible: {best_mean}')
		# LOG.info(f'Mean Spearman:       {spearman_mean}')

	LOG.info('Finished k-fold cross evaluation.')
	LOG.info('Subtask b results:')
	LOG.info(f'Mean fixed loss:           {np.round(np.mean(fixed_loss_all), 4)}')
	LOG.info(f'Mean metric:       {np.round(np.mean(metric_all), 4)}')
	LOG.info(f'Prop. best possible: {np.round(np.mean(best_all), 4)}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model_ckpt_path', help='the directory containing the '\
							'model checkpoints, as well as the CSV of splits')
	parser.add_argument('partition', help='the type of partition the model was trained on.')
	parser.add_argument('--device', default='cpu', help='the device on which to run the script')
	args = parser.parse_args()
	main(args)
