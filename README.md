# Natural Language Inference with Mixed Effects

This project implements reusable annotator random effects components for natural language inference models as presented in the following paper:

Natural Language Inference with Mixed Effects
William Gantt, Benjamin Kane, and Aaron Steven White
*SEM 2020
https://aclanthology.org/2020.starsem-1.9v2.pdf

## Development

To begin development, run:

```
git clone git://github.com/aaronstevenwhite/nli-mixed-models.git
cd nli-mixed-models
pip install --user --no-cache-dir -r ./requirements.txt
python setup.py develop
```

If you add any dependencies as you're working, please be sure to add them to `requirements.txt`. I have not been working inside a Docker container, but anyone who does should please update the Dockerfile as appropriate. All training scripts (which can be found in the `scripts` directory should be run from the root directory as follows:

```python -m scripts.{categorical,unit}.{script_name} --parameters [path/to/parameters/file]```

If no file path is given as an argument to `--parameters`, the script uses the default JSON file in the appropriate directory (either `scripts/categorical` or `scripts/unit`). In any case, these can be consulted as templates for any custom parameters files you create.

The structure of this package is largely modeled off that of [torch-combinatorial](https://github.com/aaronstevenwhite/torch-combinatorial). Guidance for how to structure any additions to this package should thus be sought there. The core components of the code were drawn from the "Natural Language Inference with Mixed Effects Models" notebook, which can be found [here](https://github.com/aaronstevenwhite/nli-mixed-models/blob/master/scripts/Natural%20Language%20Inference%20with%20Mixed%20Effects%20Models.ipynb).

Throughout the code, you may see reference to "subtask (a)" and "subtask (b)". These refer to different ways of performing inference at test time:
- `subtask (a)`: For a given test example, if we have learned random effects parameters for the annotator associated with that test example, then we use those parameters. Otherwise, we use the mean parameter value across annotators.
- `subtask (b)`: For a given test example, we *always* use the mean random effects parameter values across annotators &mdash; even if we have learned specific parameters for the relevant annotator.
The results presented in the paper are for subtask (a) and these are the results you should look at if trying to reproduce the results from the paper.

Lastly, if these setup instructions are inadequate, please update them as you see fit.
