
import torch 
import torch.nn as nn
from torch.nn.functional import softplus
import torch.nn.functional as Functional
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import gradcheck
from torch.autograd import Variable
import torch
from torch.utils.data import random_split
import os
from sklearn.preprocessing import MinMaxScaler
import sys
from NLLFunction import *
from gridworld import *
from objectworld import *
from linearvalueiteration import *
import pprint
import numpy as np
import pandas as pd
import time
import math as math
import random
import torchvision
import torchvision.transforms as transforms
from clearml import Task
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from torch.utils.tensorboard import SummaryWriter
import pickle
import pytorch_lightning as pl
from scipy.io import savemat
import scipy.io

if len(sys.argv) > 1:
    index_states_to_remove = int(str(sys.argv[1]))
    print('\n... got which noisey states from cmd line ...\n')
else:
    raise Exception('no index_states_to_remove value supplied')
    
if len(sys.argv) > 2:
    dropout_val = float(str(sys.argv[2]))
    print('\n... got dropout value from cmd line ...\n')
else:
    raise Exception('no dropout value supplied')

if len(sys.argv) > 3:
    num_paths = int(str(sys.argv[3]))
    print('\n... got number of paths value from cmd line ...\n')
else:
    raise Exception('no num paths value supplied')

if index_states_to_remove < 0 or index_states_to_remove > 3:
    raise Exception("Index of states to remove from paths must be within range 0 - 3")


# Load variables from main
open_file = open(str(num_paths) + "_NNIRL_param_list.pkl", "rb")
NNIRL_param_list = pickle.load(open_file)
open_file.close()
threshold = NNIRL_param_list[0]
optim_type = NNIRL_param_list[1]
net = NNIRL_param_list[2]
initD = NNIRL_param_list[3]
mu_sa = NNIRL_param_list[4]
muE = NNIRL_param_list[5]
mdp_data = NNIRL_param_list[6]
truep = NNIRL_param_list[7] 
NLL_EVD_plots = NNIRL_param_list[8]
#example_samples = NNIRL_param_list[9]
mdp_params = NNIRL_param_list[10] 
r = NNIRL_param_list[11] 
mdp_solution = NNIRL_param_list[12] 
feature_data = NNIRL_param_list[13] 
trueNLL = NNIRL_param_list[14]
normalise = NNIRL_param_list[15]
user_input = NNIRL_param_list[16]
worldtype = NNIRL_param_list[17]    

torch.manual_seed(mdp_params['seed'])
np.random.seed(seed=mdp_params['seed'])
random.seed(mdp_params['seed'])


# Initalise loss function
NLL = NLLFunction()
# Assign loss function constants
NLL.F = feature_data['splittable']
NLL.muE = muE
NLL.mu_sa = mu_sa
NLL.initD = initD
NLL.mdp_data = mdp_data

filename = 'ow_0.0_'+str(num_paths)+'_results_'+str(index_states_to_remove)+'.pkl'
open_file = open('/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/pyTorch/noisey_paths/results/ensembles/' + filename, "rb")
params = pickle.load(open_file)
open_file.close()


y_mc_std_relu = params[0]
y_mc_relu = params[1]


# Extract full reward function
y_mc_relu_reward = torch.from_numpy(y_mc_relu)
y_mc_relu_reward = y_mc_relu_reward.reshape(len(y_mc_relu_reward), 1)
y_mc_relu_reward = y_mc_relu_reward.repeat((1, 5))

#Solve with learned reward functions
y_mc_relu_v, y_mc_relu_q, y_mc_relu_logp, y_mc_relu_P = linearvalueiteration(mdp_data, y_mc_relu_reward)

NF_results = [y_mc_relu, y_mc_std_relu, y_mc_relu_reward, y_mc_relu_v, y_mc_relu_P, y_mc_relu_q, NLL.calculate_EVD(truep, y_mc_relu_reward), 10, 5,000]

#Save results
print('\n... saving results ...\n')

# Create path for trained models
RESULTS_PATH = "./noisey_paths/results/ensembles/formatted/"
for path in [RESULTS_PATH]:
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

file_name = RESULTS_PATH+str(worldtype)+'_'+str(dropout_val)+'_'+ str(num_paths)+ '_results_'+str(index_states_to_remove)+'.pkl'
open_file = open(file_name, "wb")
pickle.dump(NF_results, open_file)
open_file.close()


