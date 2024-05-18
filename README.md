# Researching neural network qualities in tabular data domain (TDD)

This repository contains code for:
* visualizing neural network losslandscapes in TDD
* the presence of the grokking effect in TDD
* neural networks ability to pruning in TDD
* experiments for finding optimal hyperparameter search algorithm in TDD

## Relevant papers
Starting points
- https://arxiv.org/abs/2104.10201
- https://openreview.net/forum?id=1k4rJYEwda-

# Overview of the code
The following "guidelines" are not strict at all, but they may be helpful:
- `bin` for experimental code (i.e. where the research is happening) and for pipelines that produce artifacts
- `lib` for code that is unlikely to change
- `dev` private directory, do not publish
- `research` for research ```*.ipynb``` files

## Set up the environment

### Software

Preliminaries:
- Install [conda](https://docs.conda.io/en/latest/miniconda.html)

```bash
# clone the repo somehow
export PROJECT_DIR=$(pwd)/tabular-hpo  # path to the repository root
cd $PROJECT_DIR

conda create -n template python=3.10.9
conda activate template

pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

conda env config vars set PYTHONPATH=${PROJECT_DIR}:
conda env config vars set PROJECT_DIR=${PROJECT_DIR}

conda deactivate
conda activate template
```

### Data

```
cd $PROJECT_DIR
wget https://zenodo.org/record/7612634/files/data.tar?download=1 -O data.tar
```

### Example
Once everything is ready, this should work:

```
python bin/go.py exp/mlp/california/0-tuning
```
