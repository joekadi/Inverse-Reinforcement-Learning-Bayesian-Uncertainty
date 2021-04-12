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

This saves all variables required to construct benchmark problem in "./param_list/"*num paths*_NNIRL_param_list.pkl"


