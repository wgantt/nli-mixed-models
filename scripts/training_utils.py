import numpy as np
import pandas as pd

from itertools import product
from pandas.api.types import CategoricalDtype

URL_PREFIX = 'http://megaattitude.io/projects/'
VERIDICALITY_URL = URL_PREFIX + 'mega-veridicality/mega-veridicality-v2/mega-veridicality-v2.csv'
NEG_RAISING_URL = URL_PREFIX + 'mega-negraising/mega-negraising-v1/mega-negraising-v1.tsv'
ACCEPTABILITY_URL = URL_PREFIX + 'mega-acceptability/mega-acceptability-v2/mega-acceptability-v2.tsv'

"""
TODO: Add functions for saving models

Also: It would be faster to load the data from the data/ directory
      instead of pulling it off the web, so we should probably fix that.
"""

def parameter_grid(param_dict):
	ks = list(param_dict.keys())
	vlists = []
	for k, v in param_dict.items():
		if isinstance(v, dict):
			vlists.append(parameter_grid(v))
		elif isinstance(v, list):
			vlists.append(v)
		else:
			errmsg = ("param_dict must be a dictionary contining lists or "
					  "recursively other param_dicts")
			raise ValueError(errmsg)
	for configuration in product(*vlists):
		yield dict(zip(ks, configuration))

def load_veridicality():

	def make_hypothesis(frame):
		"""Function for generating veridicality hypotheses
		"""
		if frame in ['that_S', 'for_NP_to_VP']:
			return 'That thing happened.'
		elif frame in ['to_VPeventive', 'NP_to_VPeventive']:
			return 'That person did that thing.'
		elif frame in ['to_VPstative', 'NP_to_VPstative']:
			return 'That person had that thing.'

	# Read the CSV
	df = pd.read_csv(VERIDICALITY_URL)

	# Remove non-native English speakers
	df = df[df.nativeenglish]

	"""
	MegaVeridicality contains judgments to the same items presented under two
	different prompts:

	  1. Conditional prompt: If someone _ed that a particular thing happened,
		 did that thing happen?
	  2. Unconditional prompt: Someone _ed that a particular thing happened.
		 Did that thing happen?

	We remove responses to conditional items.
	"""
	df = df[~df.conditional]

	# Finally, we remove NA responses, which arise from MTurk errors.
	df = df[~df.veridicality.isnull()]

	# Add a column for the veridicality hypothesis.
	df['hypothesis'] = df.frame.map(make_hypothesis)

	# Convert responses to integers: no = 0, maybe = 1, yes = 2. This is
	# required for the model.
	df['veridicality'] = df.veridicality.astype(CategoricalDtype(['no', 'maybe', 'yes']))
	df['target'] = df.veridicality.cat.codes

	"""
	We similarly convert the participant indices to contiguous integers. 
	This step is necessary since we removed some participants, meaning the
	participant identifiers are not necessarily contiguous. This condfsion
	is necessary for the random effects component of the model.
	"""
	df['participant'] = df.participant.astype('category').cat.codes

	# Lastly, we compute the modal response for each verb-frame pair. This
	# will allow us to determine how well the model does in comparison to
	# the best possible model.
	df['modal_response'] = df.groupby(['verb', 'frame']).target.transform(lambda x: int(np.round(np.mean(x))))

	return df


def load_neg_raising():

	def convert_subject(subject, tense, verb, frame, verbform, hypothesis):
		"""
		Converts a subject to first person when the neg-raising sentence being
		judged has a first person subject.
		"""
		if subject == 'first':
			hypothesis = hypothesis.replace('Someone', 'I')
			
			if tense == 'present':
				hypothesis = hypothesis.replace('I is', "I'm")
				
				if 'be' not in frame:
					hypothesis = hypothesis.replace(verbform, verb)
			
			return hypothesis
		else:
			return hypothesis.replace('Someone', 'That person')

	# Read the TSV
	neg = pd.read_csv(NEG_RAISING_URL, sep='\t')

	# Load MegaAcceptability, which contains necessary information for
	# hypothesis generation
	acc = load_acceptability()

	# Because we just care about the mapping from verb, frame, and tense
	# to sentence, we drop all the other columns, de-dupe, and rename the
	# sentence column to hypothesis.
	sentence_map = acc[['verb', 'verbform', 'frame', 'tense', 'sentence']].drop_duplicates().reset_index(drop=True)
	sentence_map = sentence_map.rename(columns={'sentence': 'hypothesis'})

	# We then add the negation in to make the neg-raising hypotheses and then
	# add them to the neg-raising data.
	sentence_map['hypothesis'] =\
		sentence_map.hypothesis.str.replace('something happened.', "that thing didn't happen.")
	sentence_map['hypothesis'] =\
		sentence_map.hypothesis.str.replace('to do something.', "not to do that thing.")
	sentence_map['hypothesis'] =\
		sentence_map.hypothesis.str.replace('to have something.', "not to have that thing.")
	neg = pd.merge(neg, sentence_map)

	# The last thing we need to do is convert the subject to first person
	# when the neg-raising sentence being judged has a first person subject.
	neg['hypothesis'] = neg[['subject', 'tense', 'verb', 'frame', 'verbform', 'hypothesis']].apply(lambda x: convert_subject(*x), axis=1)

	# Keep only first-person, present subjects
	neg[(neg.subject=='first')&(neg.tense=='present')]

	# Map participant numbers to contiguous integers
	neg['participant'] = neg.participant.astype('category').cat.codes

	# The target values are just the neg-raising scores
	neg['target'] = neg.negraising

	# We will be use a binary cross entrop loss in the models, and the best
	# possible response for this loss is the mean
	neg['modal_response'] = neg.groupby(['verb', 'frame', 'tense', 'subject']).negraising.transform(np.mean)

	return neg

def load_acceptability():

	def get_idx(sentence, tense, template, verblemma):
		"""
		Identifies the location of a verb in a sentence.
		"""
		tokens = sentence.split()
		lemmasplit = verblemma.split('_')
		idx = np.where([w=='V' for w in template.split()])[0][0]
		
		if template == 'S, I V':
			if len(lemmasplit) > 1:
				return [len(tokens)-3, len(tokens)-2]
			else:
				return [len(tokens)-2]
			
		elif tense == 'past_progressive':
			
			if len(lemmasplit) > 1:
				return [idx+1, idx+2]
			else:
				return [idx+1]
			
		else:
			if len(lemmasplit) > 1:
				return [idx, idx+1]
			else:
				return [idx]

	def get_verb_form(sentence, idx):
		"""
		Given a sentence and the index of the verb within the sentence,
		returns the verb.
		"""
		tokens = np.array(sentence.split())
		return ' '.join([c.replace('.', '') for t in tokens[idx] for c in t.split('_')])
	
	# Read the TSV
	acc = pd.read_csv(ACCEPTABILITY_URL, sep='\t')

	# For each verb + sentence, identify the position of the verb within the
	# sentence, as well as the actual form of the verb used in the sentence
	acc['verbidx'] = acc[['sentence', 'tense', 'frame', 'verb']].apply(lambda x: get_idx(*x), axis=1)
	acc['verbform'] = acc[['sentence', 'verbidx']].apply(lambda x: get_verb_form(*x), axis=1)

	return acc

def generate_random_splits(df, k_folds=5):
	"""Generates purely random k-fold splits of a dataframe"""

	# Assign each row a fold number, corresponding to its index mod k_folds.
	# This ensures an even number of items in each fold
	df['fold'] = df.apply(lambda row: row.name % k_folds, axis=1)

	# Randomly shuffle the fold assignments
	df['fold'] = df['fold'].sample(frac=1, replace=False)

	return df

def generate_predicate_splits(df, k_folds=5):
	"""Generates predicate-based splits

	All items featuring a particular predicate will be in the same split.
	The DataFrame is assumed to have a 'verb' column containing the
	predicate lemma
	"""
	df['fold'] = df.verb.astype('category').cat.codes
	df['fold'] = df.apply(lambda row: row.fold % k_folds, axis=1)
	return df

def generate_syntax_splits(df, datatype, k_folds=5):
	"""Generates syntax-based splits

	All items sharing a particular syntactic structure will be in the same
	split. Note: There are far fewer syntax categories than categories for
	the other split methods.
	"""

	# Veridicality: partition on frame, voice, and polarity
	if datatype == 'v':
		# Create a new column that contains all the relevant
		# syntactic properties for the split
		df['fold'] = df.apply(lambda row: '-'.join([row.frame, row.voice, row.polarity]), axis=1)

	# Neg-raising: partition on frame, tense, and subject
	elif datatype == 'n':
		# Do the same for neg-raising (just different properties)
		df['fold'] = df.apply(lambda row: '-'.join([row.frame, row.tense, row.subject]), axis=1)
	else:
		raise ValueError(f'Unknown dataset type {datatype}!')

	# Convert these values to integers
	df['fold'] = df['fold'].astype('category').cat.codes

	# Assign each category to a fold
	df['fold'] = df.apply(lambda row: df['fold'] % k_folds)

	return df


def generate_annotator_splits(df, k_folds=5):
	"""Generate annotator folds

	All items for a particular annotator are contained within the same fold.
	The dataframe is assumed to have a 'participant' column with contiguous
	integer participant IDs. One difficulty here is that, for veridicality,
	there are a few annotators who annotated many times more items than others,
	So the splits can end up being rather uneven.
	"""
	df['fold'] = df.apply(lambda row: row.participant % k_folds, axis=1)
	return df

def generate_splits(df, split_type, k_folds=5, datatype=None):
	if split_type == 'random':
	    df = generate_random_splits(df, k_folds=k_folds)
	elif split_type == 'predicate':
	    df = generate_predicate_splits(df, k_folds=k_folds)
	elif split_type == 'annotator':
	    df = generate_annotator_splits(df, k_folds=k_folds)
	elif split_type == 'syntax':
	    df = generate_syntax_splits(df, datatype=datatype, k_folds=k_folds)
	else:
	    raise ValueError(f'Unknown split type {split_type}')
	return df
