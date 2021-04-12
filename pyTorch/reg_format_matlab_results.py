
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
    num_paths = int(str(sys.argv[1]))
    print('\n... got number of paths value from cmd line ...\n')
else:
    raise Exception('No paths supplied')

    
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

filename = 'ow_'+str(num_paths)+'_results.mat'
mat = scipy.io.loadmat('/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/pyTorch/regular/results/gpirl/'+filename)

y_mc_relu_reward = mat['result_struct'][0][0][0]
y_mc_relu = y_mc_relu_reward[:,0]
y_mc_std_relu = mat['result_struct'][0][0][1]

y_mc_relu_reward = torch.from_numpy(y_mc_relu_reward)

y_mc_relu_v, y_mc_relu_q, y_mc_relu_logp, y_mc_relu_P = linearvalueiteration(mdp_data, y_mc_relu_reward)

NP_results = [y_mc_relu, y_mc_std_relu, y_mc_relu_reward, y_mc_relu_v, y_mc_relu_P, y_mc_relu_q, NLL.calculate_EVD(truep, y_mc_relu_reward), 10, 5,000]

#Save results
print('\n... saving results ...\n')

# Create path for trained models
RESULTS_PATH = "./regular/results/gpirl/"
for path in [RESULTS_PATH]:
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

file_name = RESULTS_PATH+str(worldtype)+'_'+ str(num_paths)+ '_results.pkl'
open_file = open(file_name, "wb")
pickle.dump(NP_results, open_file)
open_file.close()


