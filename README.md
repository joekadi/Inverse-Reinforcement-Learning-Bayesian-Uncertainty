# Evaluating Uncertainty Estimation Methods For Deep Neural Network’s In Inverse Reinforcement Learning
This is part of my Computing Science Level 5 MSci Project @ Glasgow University where I compare the quality of epistemic uncertainty estimates from multiple Deep Neural Network reguliarsation techniques on the Inverse Reinforcement Learning problem. 

## Setup Instructions

In every overwrite change sys.path[0] = "/Users/MSci-Project/pyTorch/" to path of cloned directory
This will ensure all relative and absolute imports find correc paths

To create conda virtualenv from requirements:

`$ conda create --name <env> --file requirements.txt`

## Running Instructions

To initialise benchmark (objectworld or gridworld):

`$ python initialise_problem.py <worldtype> <number_of_paths>`

This saves all variables required to construct benchmark problem in "./param_list/"

# Regular IRL

To train a model:

`cd ./train_models/regular/`

`$ python <training_script_name>.py <dropout_p_value> <number_of_paths>`

Trained model saved in $TRAINED_MODELS_PATH$

For ensembles, eval also carried out in this script and results saved to $RESULTS_PATH$

To evaluate a trained model:

`$ cd "./eval_models/"`

`$ python <eval_script_name>.py <dropout_p_value> <number_of_paths>`

Results saved in$ RESULTS_PATH$

# Noisy IRL

`cd ./train_models/$TYPE_OF_NOISE$/`

index_of_noisy_states is 0 for states 1-32, 1 for states 1-64 and 2 for states 1-128

`$ python <training_script_name>.py <index_of_noisy_states> <dropout_p_value> <number_of_paths>`

Trained model saved in $TRAINED_MODELS_PATH$

For ensembles, eval also carried out in this script and results saved to $RESULTS_PATH$

To evaluate a trained model:

`$ cd "./eval_models/$TYPE_OF_NOISE$"`

`$ python <eval_script_name>.py <index_of_noisy_states> <dropout_p_value> <number_of_paths>`

Results saved in $RESULTS_PATH$







