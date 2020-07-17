import os
import argparse
import pandas as pd

from scripts.training_utils import load_model_with_args
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

LOG = setup_logging()

def resave_old_style_model(model_ckpt_path, model_type):
	"""Re-saves an old format model

	Previously, our models did not have the mean and covariance of the
	random effects saved as instance variables. This function loads
	such a model, adds those instance variables, and resaves it.
	"""
	if model_type == 'icat':
		model_cls = CategoricalRandomIntercepts
	elif model_type == 'scat':
		model_cls = CategoricalRandomSlopes
	elif model_type == 'iunit':
		model_cls = UnitRandomIntercepts
	elif model_type == 'sunit':
		model_cls = UnitRandomSlopes
	else:
		raise ValueError(f'Unknown model type {model_type}!')

	model, hyper_params = load_model_with_args(model_cls, ckpt_path)
	print(model.__class__)


def main(args):
	resave_old_style_model()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model_ckpt_dir', help='the directory containing the '\
							'model checkpoints, as well as the CSV of splits')
	parser.add_argument('model_type', help='the type of model to be evaluated: '\
							'icat, iunit, scat, sunit'
	parser.add_argument('model_ckpt_prefix', help='the prefix shared by the '\
							'names of all checkpoint files in the checkpoint '\
							'directory')
	parser.add_argument('data_file', help='the name of the CSV file within the '\
							'model_ckpt_dir that contains the train/test splits')
	args = parser.parse_args()
	main(args)