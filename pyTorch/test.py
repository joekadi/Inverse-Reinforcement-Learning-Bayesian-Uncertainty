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
from operator import add
from gridworld import gridworlddrawuncertainty
from utils import epdue, plot_epdue, ence, plot_ence, se
from scipy.stats import variation
from tabulate import tabulate
import csv
from swag_np_train import LitModel
from torch.optim.swa_utils import AveragedModel, SWALR

open_file = open("/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/pyTorch/128_NNIRL_param_list.pkl", "rb")
NNIRL_param_list = pickle.load(open_file)
open_file.close()

mdp_params = NNIRL_param_list[10] 
mdp_solution = NNIRL_param_list[12] 
feature_data = NNIRL_param_list[13] 
mdp_data = NNIRL_param_list[6]
truep = mdp_solution['p']

torch.manual_seed(mdp_params['seed'])
np.random.seed(seed=mdp_params['seed'])
random.seed(mdp_params['seed'])
'''
open_file = open("/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/pyTorch/noisey_features/results/swag/ow_0.0_128_results_1.pkl", "rb")
open_results = pickle.load(open_file)
open_file.close()

policy = open_results[4]

stds = open_results[1]


swa_ence = ence(truep, policy, stds)
'''


#os.system('python swag_nf_train.py ' + str(1) + ' ' + str(0.0) + ' ' + str(128))


print('\nmodel done training\n')



#Load trained model
configuration_dict = {'number_of_epochs': 3, 'base_lr': 0.05, 'p': 0.0, 'no_hidden_layers': 3, 'no_neurons_in_hidden_layers': len(feature_data['splittable'][0])*2 } #set config params for clearml
model = LitModel(len(feature_data['splittable'][0]), 'relu', configuration_dict)
irl_model = AveragedModel(model)
irl_model.load_state_dict(torch.load("/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/pyTorch/noisey_features/models/test/swag/ow_0.0_128_NF_model_1.pth"
))

num_preds = 100 # Number of samples
start_time = time.time()

irl_model = irl_model.train()
# Make predicitons w/ trained models
print('\n... Making predictions w/ trained models ...\n')
for i in range(len(feature_data['splittable'])):
    Yt_hat_relu = np.array([torch.matmul(feature_data['splittable'],irl_model(feature_data['splittable'][i].view(-1)).reshape(len(feature_data['splittable'][0]),1)).data.cpu().numpy() for _ in range(num_preds)]).squeeze()
run_time = (time.time() - start_time)


# Extract mean and std of predictions
y_mc_relu = Yt_hat_relu.mean(axis=0)





y_mc_std_relu = Yt_hat_relu.std(axis=0)


# Extract full reward function
y_mc_relu_reward = torch.from_numpy(y_mc_relu)
y_mc_relu_reward = y_mc_relu_reward.reshape(len(y_mc_relu_reward), 1)
y_mc_relu_reward = y_mc_relu_reward.repeat((1, 5))

#Solve with learned reward functions
y_mc_relu_v, y_mc_relu_q, y_mc_relu_logp, y_mc_relu_P = linearvalueiteration(mdp_data, y_mc_relu_reward)


swag_ence = ence(truep, y_mc_relu_P, y_mc_std_relu)

print('swa ence:', swa_ence)
print('swag ence:', swag_ence)




