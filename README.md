# Natural Language Inference with Mixed Effects

This project implements reusable annotator random effects components for natural language inference models.

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

The only way I could get this to work was to add `__init__.py` files to all directories in scripts. If anyone is able to fix that, please do. 

The structure of this package is largely modeled off that of [torch-combinatorial](https://github.com/aaronstevenwhite/torch-combinatorial). Guidance for how to structure any additions to this package should thus be sought there. The core components of the code are drawn from the "Natural Language Inference with Mixed Effects Models" notebook, which can be found [here](https://github.com/aaronstevenwhite/nli-mixed-models/blob/master/scripts/Natural%20Language%20Inference%20with%20Mixed%20Effects%20Models.ipynb).

Lastly, if these setup instructions are inadequate, please update them as you see fit.
