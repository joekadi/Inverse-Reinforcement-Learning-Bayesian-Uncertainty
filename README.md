# Evaluating Uncertainty Estimation Methods For Deep Neural Network’s In Inverse Reinforcement Learning
This is part of my Computing Science Level 5 MSci Project @ Glasgow University where I compare the quality of uncertainty estimates from multiple Deep Neural Network uncertainty calibration techniques on the Inverse Reinforcement Learning problem. Namely: Monte-Carlo Dropout, Stochastic Weight Averaging Gaussian and Caruna's Ensemble Selector.

# Setup Instructions

In every overwrite change sys.path[0] = "/Users/MSci-Project/pyTorch/" to path of cloned directory
This will ensure all relative and absolute imports find correc paths

To create conda virtualenv from requirements:

`$ conda create --name <env> --file requirements.txt`

# Running Instructions

All objectworld params customisable in objectworld.create_objectworld(). Default params:

`mdp_params = {'n': 16, 'placement_prob': 0.05, 'c1': 2.0, 'c2': 2.0, 'continuous': False, 'determinism': 1.0, 'discount': 0.9, 'seed': 0, 'r_tree': None}`
  
All gridworld params customisable in gridworld.create_gridworld(). Default params:

`mdp_params = {'n': 8, 'b': 1, 'determinism': 1.0, 'discount': 0.99, 'seed': 0}`  
  
To initialise benchmark (objectworld or gridworld):

`$ python initialise_problem.py <worldtype> <number_of_paths>`

This saves all variables required to construct benchmark problem in "./param_list/"

## Regular IRL

To train a model:

`cd ./train_models/regular/`

`$ python <training_script_name>.py <dropout_p_value> <number_of_paths>`

Trained model saved in $TRAINED_MODELS_PATH$

For ensembles, eval also carried out in this script and results saved to $RESULTS_PATH$

To evaluate a trained model:

`$ cd "./eval_models/"`

`$ python <eval_script_name>.py <dropout_p_value> <number_of_paths>`

Results saved in$ RESULTS_PATH$

## Noisy IRL

`cd ./train_models/$TYPE_OF_NOISE$/`

index_of_noisy_states is 0 for states 1-32, 1 for states 1-64 and 2 for states 1-128

`$ python <training_script_name>.py <index_of_noisy_states> <dropout_p_value> <number_of_paths>`

Trained model saved in $TRAINED_MODELS_PATH$

For ensembles, eval also carried out in this script and results saved to $RESULTS_PATH$

To evaluate a trained model:

`$ cd "./eval_models/$TYPE_OF_NOISE$"`

`$ python <eval_script_name>.py <index_of_noisy_states> <dropout_p_value> <number_of_paths>`

Results saved in $RESULTS_PATH$

## Hyperparameter tuning

ClearML (https://clear.ml) is used to tune each models hyper parameters. Methods supported: GridSearch, RandomSearch, OptimizerBOHB and OptimizerOptuna.

To hyper parameter tune:

`pip install clearml`

In each model training script change task log line to

`task = Task.init(project_name='<your_project_name>', task_name='<task_name>')`

After running training script, the task name will be logged as a task in your clearML dashboard. TASK_ID is then obtainable from your clearML dashboard.

Configure hyper parameter search  in pyTorch/train_models/hyperparamsearch.py then run:

`$ cd "pyTorch/train_models/"`
`$ python hyperparamsearch.py <TASK_ID>`

## Analysing and visualising IRL performance:

`$ cd "/pyTorch/results"`

`$ python irl_visualise.py`

All result figures PNG's and CSVs can be seen in "pyTorch/results/regular/"


## Analysing and visualising uncertainty calibrations:

`$ cd "/pyTorch/results"`

`$ python uncertainty_visualise.py`

All result figures PNG's and CSVs can be seen in "pyTorch/results/$TYPE_OF_NOISE$"

All figures used in the final report can be seen in "./final_figures/"
