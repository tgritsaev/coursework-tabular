# Project: hpo in tabular data

We investigate the hyperparameter optimization landscape in tabular data. 

We are mainly interested in studing the variance and efficiency of various black-box hyperparameter tuning algorithms, such as TPE, Random Search and TuRBO. Our main focus are tabular problems of medium to large scale and differentiable neural networks.

## Relevant papers
Starting points
- https://arxiv.org/abs/2104.10201
- https://openreview.net/forum?id=1k4rJYEwda-

## Unstructured Ideas

  The models being benchmarked (baseline deep, baseline shallow, SoTA deep)

  - mlp
  - xgboost
  - mlp-plr

  Tuning configuration:
  - base (as we always do)
  - plus extensive optimizer tuning
  - plus additional plr tuning (sigmas)
  - plus a-la cocktails (architectural details) /maybe not/

  Wanna report:
  - tuning variance
    sources of variance
    - random seeds of the tuning algorithm *must*
    - seeds of the algorithms (initializations)
  - Tuning objectives
    - default
    - robust to initialization (is somewhat better?)

  Tuning (the variables)

  - number of tuning iterations (plots for all datasets) ~range(0,1000,100)~
  - the optimization algorithm
    - grid search (best that we can)
    - random search *must*
    - TPE (optuna default) *must*
    - turbo (trust region something) *must*


  We are mainly interested in varying the tuning seed and comparing the hyperparam optimization algorithms
  on more tabular DL problems

  Additional research:
  - investigate early stopping
  - some good initial initialization of parameters
  - try the google vizer based transformer for hyperoptimization
  - combine optimization algorithms (ensemble)


## Plan

Get the baselines up and running:
- Verify the main claims that HPO is superior to random search
  - Run default configs with RandomSearch and optuna's TPE, do this for 1000? iterations, and 5 different seeds (FFN and XGBoost)
  - Get this done and talk  

# Overview of the code
The following "guidelines" are not strict at all, but they may be helpful:
- `bin` for experimental code (i.e. where the research is happening) and for pipelines that produce artifacts
- `lib` for code that is unlikely to change
- `dev` private directory, do not publish

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
wget TODO -O data.tar
tar -xvf data.tar
```

### Example
Once everything is ready, this should work:

```
python bin/go.py exp/mlp/california/0-tuning
```
