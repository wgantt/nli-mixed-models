import os, uuid, torch, inspect
import numpy as np
import pandas as pd

from .setup_logging import setup_logging
from itertools import product
from pandas.api.types import CategoricalDtype

URL_PREFIX = "http://megaattitude.io/projects/"


# The versions of the neg-raising and veridicality data at these URLS
# have changed since we obtained our initial results. We therefore use
# the local copies in this repo instead, which are fixed. 
# VERIDICALITY_URL = (
#     URL_PREFIX + "mega-veridicality/mega-veridicality-v2/mega-veridicality-v2.csv"
# )
# NEG_RAISING_URL = (
#     URL_PREFIX + "mega-negraising/mega-negraising-v1/mega-negraising-v1.tsv"
# )

VERIDICALITY_DATA = "data/mega-veridicality-v2.csv"
NEG_RAISING_DATA = "data/mega-negraising-v1.tsv"
DIFFICULTY_DATA = "data/mega-intensionality-nonfinite-difficulty.tsv"
INTENSIONALITY_DATA = "data/mega-intensionality-v1.tsv"
NAME_DATA = "data/names.txt"

ACCEPTABILITY_URL = (
    URL_PREFIX + "mega-acceptability/mega-acceptability-v2/mega-acceptability-v2.tsv"
)
RANDOM_SEED = 42

LOG = setup_logging()

# set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def parameter_grid(param_dict):
    ks = list(param_dict.keys())
    vlists = []
    for k, v in param_dict.items():
        if isinstance(v, dict):
            vlists.append(parameter_grid(v))
        elif isinstance(v, list):
            vlists.append(v)
        else:
            errmsg = (
                "param_dict must be a dictionary contining lists or "
                "recursively other param_dicts"
            )
            raise ValueError(errmsg)
    for configuration in product(*vlists):
        yield dict(zip(ks, configuration))


# Functions below for saving and loading models arae taken directly from
# torch-combinatorial


def save_model(data_dict, ckpt_dir, file_name):
    # Not sure why Aaron and Gene were using random
    # hashing to name their model files. Absent more information,
    # more descriptive file names seem preferable. -- W.G.
    #
    # rand_hash = uuid.uuid4().hex
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, file_name)
    torch.save(data_dict, ckpt_path)
    return ckpt_path


def save_model_with_args(params, model, initargs, ckpt_dir, file_name):
    filtered_args = {}
    for p in inspect.signature(model.__class__.__init__).parameters:
        if p in initargs:
            filtered_args[p] = initargs[p]
    ckpt_dict = dict(
        params, **{"state_dict": model.state_dict(), "curr_hyper": filtered_args}
    )
    return save_model(ckpt_dict, ckpt_dir, file_name)


def load_model_with_args(cls, ckpt_path):
    ckpt_dict = torch.load(ckpt_path, map_location="cuda:0")
    hyper_params = ckpt_dict["curr_hyper"]
    model = cls(**hyper_params)
    model.load_state_dict(ckpt_dict["state_dict"])
    return model, hyper_params


def load_model_with_missing_hyperparams(cls, ckpt_path, missing_hyperparams):
    ckpt_dict = torch.load(ckpt_path, map_location="cuda:0")
    existing_hyperparams = ckpt_dict["curr_hyper"]
    all_hyperparams = {**existing_hyperparams, **missing_hyperparams}
    all_hyperparams["device"] = "cuda:0"
    model = cls(**all_hyperparams)
    model.load_state_dict(ckpt_dict["state_dict"])
    return model, all_hyperparams


def mode(x):
    values, counts = np.unique(x, return_counts=True)
    return values[counts.argmax()]


def load_veridicality():
    def make_hypothesis(frame):
        """Function for generating veridicality hypotheses
		"""
        if frame in ["that_S", "for_NP_to_VP"]:
            return "That thing happened."
        elif frame in ["to_VPeventive", "NP_to_VPeventive"]:
            return "That person did that thing."
        elif frame in ["to_VPstative", "NP_to_VPstative"]:
            return "That person had that thing."

    def mode(x):
        values, counts = np.unique(x, return_counts=True)
        return values[counts.argmax()]

    # Read the CSV
    df = pd.read_csv(VERIDICALITY_DATA)

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
    df["hypothesis"] = df.frame.map(make_hypothesis)

    # Convert responses to integers: no = 0, maybe = 1, yes = 2. This is
    # required for the model.
    df["veridicality"] = df.veridicality.astype(
        CategoricalDtype(["no", "maybe", "yes"])
    )
    df["target"] = df.veridicality.cat.codes

    """
	We similarly convert the participant indices to contiguous integers. 
	This step is necessary since we removed some participants, meaning the
	participant identifiers are not necessarily contiguous. This condfsion
	is necessary for the random effects component of the model.
	"""
    df["participant"] = df.participant.astype("category").cat.codes
    
    df["item"] = (df.sentence+df.hypothesis).astype("category").cat.codes

    # Calculate accuracy of zero-rule classifier (i.e. predicting majority class)
    mode_prediction = torch.ones(df.target.shape) * mode(df.target)
    mode_target = torch.tensor(df.target.values)
    mode_accuracy = (mode_prediction == mode_target).numpy().mean()
    LOG.info("Majority class accuracy (MV):")
    LOG.info(f"{mode_accuracy}")

    # Lastly, we compute the modal response for each verb-frame pair. This
    # will allow us to determine how well the model does in comparison to
    # the best possible model.
    df["modal_response"] = df.groupby(["verb", "frame"]).target.transform(
        lambda x: mode(x)
    )

    return df


def load_neg_raising():
    def convert_subject(subject, tense, verb, frame, verbform, hypothesis):
        """
		Converts a subject to first person when the neg-raising sentence being
		judged has a first person subject.
		"""
        if subject == "first":
            hypothesis = hypothesis.replace("Someone", "I")

            if tense == "present":
                hypothesis = hypothesis.replace("I is", "I'm")

                if "be" not in frame:
                    hypothesis = hypothesis.replace(verbform, verb)

            return hypothesis
        else:
            return hypothesis.replace("Someone", "That person")

    # Read the TSV
    neg = pd.read_csv(NEG_RAISING_DATA, sep="\t")

    # Load MegaAcceptability, which contains necessary information for
    # hypothesis generation
    acc = load_acceptability()

    # Because we just care about the mapping from verb, frame, and tense
    # to sentence, we drop all the other columns, de-dupe, and rename the
    # sentence column to hypothesis.
    sentence_map = (
        acc[["verb", "verbform", "frame", "tense", "sentence"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    sentence_map = sentence_map.rename(columns={"sentence": "hypothesis"})

    # We then add the negation in to make the neg-raising hypotheses and then
    # add them to the neg-raising data.
    sentence_map["hypothesis"] = sentence_map.hypothesis.str.replace(
        "something happened.", "that thing didn't happen."
    )
    sentence_map["hypothesis"] = sentence_map.hypothesis.str.replace(
        "to do something.", "not to do that thing."
    )
    sentence_map["hypothesis"] = sentence_map.hypothesis.str.replace(
        "to have something.", "not to have that thing."
    )
    neg = pd.merge(neg, sentence_map)

    # The last thing we need to do is convert the subject to first person
    # when the neg-raising sentence being judged has a first person subject.
    neg["hypothesis"] = neg[
        ["subject", "tense", "verb", "frame", "verbform", "hypothesis"]
    ].apply(lambda x: convert_subject(*x), axis=1)

    # Map participant numbers to contiguous integers
    neg["participant"] = neg.participant.astype("category").cat.codes

    # Create indices for unique items
    neg["item"] = (neg.sentence+neg.hypothesis).astype("category").cat.codes

    # The target values are just the neg-raising scores
    neg["target"] = neg.negraising

    # We will be use a binary cross entrop loss in the models, and the best
    # possible response for this loss is the mean
    neg["modal_response"] = neg.groupby(
        ["verb", "frame", "tense", "subject"]
    ).negraising.transform(np.median)

    return neg


def load_unit_data(dataset):

    # Read the TSV
    data = pd.read_csv(dataset, sep="\t")

    data = data.rename(columns={'target':'consequent_target','antecedent_text':'sentence','consequent_text':'hypothesis','response':'target'})

    # Map participant numbers to contiguous integers
    data["participant"] = data.participant.astype("category").cat.codes

    # Create indices for unique items
    data["item"] = (data.sentence+data.hypothesis).astype("category").cat.codes

    # We will be use a binary cross entrop loss in the models, and the best
    # possible response for this loss is the mean
    data["modal_response"] = data.groupby(['sentence', 'hypothesis']).target.transform(np.median)

    return data


def load_difficulty(templatized=True):
    names = load_names()
    def fill_templates(sentence, hypothesis):
        replacements = np.random.choice(names, 2)
        sentence = sentence.replace('A', replacements[0])
        hypothesis = hypothesis.replace('A', replacements[0])
        sentence = sentence.replace('B', replacements[1])
        hypothesis = hypothesis.replace('B', replacements[1])
        sentence = sentence.replace('C', 'something')
        hypothesis = hypothesis.replace('C', 'that thing')
        return sentence, hypothesis

    # Read the TSV
    diff = pd.read_csv(DIFFICULTY_DATA, sep="\t")

    diff = diff.rename(columns={'target':'consequent_target','antecedent_text':'sentence','consequent_text':'hypothesis'})

    # Map participant numbers to contiguous integers
    diff["participant"] = diff.participant.astype("category").cat.codes

    # Create indices for unique items
    diff["item"] = (diff.sentence+diff.hypothesis).astype("category").cat.codes

    # The target values are just the difficulty scores
    diff["target"] = diff.response_difficulty

    # Fill templates
    if not templatized:
        diff[["sentence","hypothesis"]] = diff[["sentence", "hypothesis"]].apply(lambda x: fill_templates(*x), axis=1, result_type='expand')

    # We will be use a binary cross entrop loss in the models, and the best
    # possible response for this loss is the mean
    diff["modal_response"] = diff.groupby(
        ["antecedent_verb", "antecedent_polarity", "consequent_target", "control", "antecedent_frame",
         "consequent_verb", "consequent_embedded_tense"]
    ).response_difficulty.transform(np.median)

    return diff


def load_intensionality(templatized=True):
    names = load_names()
    def fill_templates(sentence, hypothesis):
        replacements = np.random.choice(names, 2)
        sentence = sentence.replace('A', replacements[0])
        hypothesis = hypothesis.replace('A', replacements[0])
        sentence = sentence.replace('B', replacements[1])
        hypothesis = hypothesis.replace('B', replacements[1])
        sentence = sentence.replace('C', 'something')
        hypothesis = hypothesis.replace('C', 'that thing')
        return sentence, hypothesis

    # Read both the nonfinite difficulty TSV and the finite MI TSV
    diff = pd.read_csv(DIFFICULTY_DATA, sep="\t")
    mi = pd.read_csv(INTENSIONALITY_DATA, sep="\t")

    # Concatenate the two
    diff = diff.drop(columns=['response_difficulty'])
    diff = diff.rename(columns={'response_likelihood':'response'})
    mi = pd.concat([mi, diff])
    mi["control"] = mi["control"].fillna('na')

    mi = mi.rename(columns={'target':'consequent_target','antecedent_text':'sentence','consequent_text':'hypothesis'})

    # Map participant numbers to contiguous integers
    mi["participant"] = mi.participant.astype("category").cat.codes

    # Create indices for unique items
    mi["item"] = (mi.sentence+mi.hypothesis).astype("category").cat.codes

    # The target values are just the difficulty scores
    mi["target"] = mi.response

    # Fill templates
    if not templatized:
        mi[["sentence","hypothesis"]] = mi[["sentence", "hypothesis"]].apply(lambda x: fill_templates(*x), axis=1, result_type='expand')

    # We will be use a binary cross entrop loss in the models, and the best
    # possible response for this loss is the mean
    mi["modal_response"] = mi.groupby(
        ["antecedent_verb", "antecedent_polarity", "consequent_target", "control", "antecedent_frame",
         "consequent_verb", "consequent_embedded_tense"]
    ).response.transform(np.median)

    return mi


def load_acceptability():
    def get_idx(sentence, tense, template, verblemma):
        """
		Identifies the location of a verb in a sentence.
		"""
        tokens = sentence.split()
        lemmasplit = verblemma.split("_")
        idx = np.where([w == "V" for w in template.split()])[0][0]

        if template == "S, I V":
            if len(lemmasplit) > 1:
                return [len(tokens) - 3, len(tokens) - 2]
            else:
                return [len(tokens) - 2]

        elif tense == "past_progressive":

            if len(lemmasplit) > 1:
                return [idx + 1, idx + 2]
            else:
                return [idx + 1]

        else:
            if len(lemmasplit) > 1:
                return [idx, idx + 1]
            else:
                return [idx]

    def get_verb_form(sentence, idx):
        """
		Given a sentence and the index of the verb within the sentence,
		returns the verb.
		"""
        tokens = np.array(sentence.split())
        return " ".join([c.replace(".", "") for t in tokens[idx] for c in t.split("_")])

    # Read the TSV
    acc = pd.read_csv(ACCEPTABILITY_URL, sep="\t")

    # For each verb + sentence, identify the position of the verb within the
    # sentence, as well as the actual form of the verb used in the sentence
    acc["verbidx"] = acc[["sentence", "tense", "frame", "verb"]].apply(
        lambda x: get_idx(*x), axis=1
    )
    acc["verbform"] = acc[["sentence", "verbidx"]].apply(
        lambda x: get_verb_form(*x), axis=1
    )

    return acc


def load_names(n=100):
    """Loads a list of the n most common first names (for use in filling
       templatized sentences)"""
    with open(NAME_DATA) as f:
        lines = [line.rstrip() for line in f]
    names = lines[0:n] if n < len(lines) else lines
    return names


def generate_random_splits(df, k_folds=5):
    """Generates purely random k-fold splits of a dataframe"""

    # Sort the data by participant before generating the folds. This
    # will guarantee that each participant appears in each fold.
    df.sort_values(by="participant", inplace=True, ignore_index=True)

    # Assign each row a fold number, corresponding to its index mod k_folds.
    # This ensures an even number of items in each fold
    df["fold"] = df.apply(lambda row: row.name % k_folds, axis=1)

    # Randomly shuffle the fold assignments
    df["fold"] = df["fold"].sample(frac=1, replace=False, random_state=RANDOM_SEED)

    # Verify
    _assert_each_value_in_each_fold(df, "participant")

    # Show how many items are in each fold
    LOG.info("Items per fold:")
    LOG.info(f"{df.groupby('fold')['participant'].count()}")

    # Show how many annotators are in each fold
    LOG.info("Unique annotators per fold:")
    LOG.info(f"{df.groupby('fold')['participant'].nunique()}")

    return df


def generate_predicate_splits(df, k_folds=5):
    """Generates predicate-based splits

	All items featuring a particular predicate will be in the same split.
	The DataFrame is assumed to have a 'verb' column containing the
	predicate lemma.
	"""

    # Assign each verb type to its own fold
    df["fold"] = df.verb.astype("category").cat.codes
    df["fold"] = df.apply(lambda row: row.fold % k_folds, axis=1)

    # Verify that each verb appears in exactly one fold
    _assert_unique_to_fold(df, "verb")

    # Show how many items are in each fold
    print("Items per fold:")
    print(df.groupby("fold")["participant"].count())

    # Show how many annotators are in each fold. Interestingly,
    # we don't seem to need to do anything fancy to make it so that
    # nearly all annotators appear in all folds (for k=5)
    print("Unique annotators per fold:")
    print(df.groupby("fold")["participant"].nunique())

    # return df
    return df


def generate_syntax_splits(df, datatype, k_folds=5):
    """Generates syntax-based splits

	All items sharing a particular syntactic structure will be in the same
	split. Note: There are far fewer syntax categories than categories for
	the other split methods.
	"""

    # Veridicality: partition on frame, voice, and polarity
    if datatype == "v":
        # Create a new column that contains all the relevant
        # syntactic properties for the split
        df["syntax"] = df.apply(
            lambda row: "-".join([row.frame, row.voice, row.polarity]), axis=1
        )

    # Neg-raising: partition on frame, tense, and subject
    elif datatype == "n":
        # Do the same for neg-raising (just different properties)
        df["syntax"] = df.apply(
            lambda row: "-".join([row.frame, row.tense, row.subject]), axis=1
        )
    else:
        raise ValueError(f"Unknown dataset type {datatype}!")

    # Assign each syntax combination to a fold. This works fine for
    # neg-raising, but for veridicality, there's a seemingly irresolvable
    # problem in which one fold ends up with 357 annotators, whereas
    # all others have the full 507, assuming k=5.
    df["syntax"] = df.syntax.astype("category").cat.codes
    df["fold"] = df.apply(lambda row: row.syntax % k_folds, axis=1)

    # Verify that each syntax structure appears in exactly one fold
    _assert_unique_to_fold(df, "syntax")

    # Show the number of unique annotators per fold
    print("Unique annotators per fold:")
    print(df.groupby("fold")["participant"].nunique())

    return df


def generate_annotator_splits(df, k_folds=5):
    """Generate annotator folds

	All items for a particular annotator are contained within the same fold.
	The dataframe is assumed to have a 'participant' column with contiguous
	integer participant IDs. One difficulty here is that, for veridicality,
	there are a few annotators who annotated many times more items than others,
	So the splits can end up being rather uneven.
	"""
    df["fold"] = df.apply(lambda row: row.participant % k_folds, axis=1)

    # Verify that each annotator appears in exactly one fold
    _assert_unique_to_fold(df, "participant")

    # Show the number of items per fold
    print("Items per fold:")
    print(df.groupby("fold")["participant"].count())

    # Show the number of annotators per fold
    print("Unique annotators per fold:")
    print(df.groupby("fold")["participant"].nunique())

    return df


def generate_splits(df, split_type, k_folds=5, datatype=None):
    if split_type == "random":
        df = generate_random_splits(df, k_folds=k_folds)
    elif split_type == "predicate":
        df = generate_predicate_splits(df, k_folds=k_folds)
    elif split_type == "annotator":
        df = generate_annotator_splits(df, k_folds=k_folds)
    elif split_type == "syntax":
        df = generate_syntax_splits(df, datatype=datatype, k_folds=k_folds)
    else:
        raise ValueError(f"Unknown split type {split_type}")
    return df


def _assert_each_value_in_each_fold(df, col):
    """Verifies that each value of a certain column appears in each fold

	Assumes fold column 'fold' and participant column 'participant'.
	"""

    # All participants and all folds
    all_values = set(df[col])
    all_folds = set(df.fold)

    for num_values_in_fold in df.groupby("fold")[col].nunique():
        assert num_values_in_fold == len(
            all_values
        ), f"some values are not in all folds"


def _assert_unique_to_fold(df, groupby):
    assert (
        df.groupby(groupby)["fold"].nunique().max() == 1
    ), f'distinct "{groupby}" values span multiple folds'


if __name__ == "__main__":
    test = load_difficulty(templatized=False)
    print(len(test))
    print(test.columns)
    print(test.head())
    print(test.tail())
    print(test[['sentence','hypothesis']].head())

    # print(pd.value_counts(test['participant']))

    # test2 = generate_random_splits(test)
    # print(test2)