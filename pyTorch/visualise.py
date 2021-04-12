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

tensorboard_writer = SummaryWriter('./tensorboard_logs')
torch.set_printoptions(precision=5, sci_mode=False, threshold=1000)
torch.set_default_tensor_type(torch.DoubleTensor)

num_states_to_remove = [np.arange(0, 64, 1), np.arange(0, 128, 1), np.arange(128, 256, 1), np.arange(192,256,1)]
states_to_remove = ['States 1-64 Removed From Paths', 'States 1-128 Removed From Paths', 'States 128-256 Removed From Paths', 'States 192-256 Removed From Paths']

# Load variables from main
open_file = open("NNIRL_param_list.pkl", "rb")
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
example_samples = NNIRL_param_list[9]
mdp_params = NNIRL_param_list[10] 
r = NNIRL_param_list[11] 
mdp_solution = NNIRL_param_list[12] 
feature_data = NNIRL_param_list[13] 
trueNLL = NNIRL_param_list[14]
normalise = NNIRL_param_list[15]
user_input = NNIRL_param_list[16]
worldtype = NNIRL_param_list[17]

optimal_paths = 128
p_value_to_inspect = 0.4

# Compute visible examples.
Eo = torch.zeros(int(mdp_data['states']),1)
for i in range(len(example_samples[0])):
    for t in range(len(example_samples[0][0])):
        Eo[[example_samples][0][i][t][0]] = 1
g = torch.ones(int(mdp_data['states']),1)*0.5+Eo*0.5

#Get directory for results
RESULTS_PATH = "./regular/results/"

#Create path for graphs 
GRAPHS_PATH = RESULTS_PATH + "/graphs/"
for path in [GRAPHS_PATH]:
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

#Init results dict - real
results = {
        'ow': {
                8: {0.0: None, 0.2: None,  0.4: None, 0.6: None,0.8: None}, 
                16: {0.0: None, 0.2: None,  0.4: None, 0.6: None,0.8: None}, 
                32: {0.0: None, 0.2: None,  0.4: None, 0.6: None,0.8: None}, 
                64:  {0.0: None, 0.2: None,  0.4: None, 0.6: None,0.8: None}, 
                128: {0.0: None, 0.2: None, 0.4: None,0.6: None,0.8: None}
                },
                
        'gw': {
                12: {0.0: None, 0.2: None, 0.4: None,  0.6: None,0.8: None}, 
                48: {0.0: None, 0.2: None, 0.4: None,  0.6: None,0.8: None}, 
                128: {0.0: None, 0.2: None, 0.4: None, 0.6: None, 0.8: None},
                256: {0.0: None, 0.2: None, 0.4: None, 0.6: None,0.8: None}
                }
    }


# Get lists of variants
# Worlds = results.keys()
worlds = ['ow']
paths = list(results['ow'].keys())
dropout_vals = results['ow'][paths[0]].keys()

# Populate results dictionary
for world in worlds:
    for no_paths in paths:
        for dropout_val in dropout_vals:
            file_name = RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results.pkl'
            open_file = open(file_name, "rb")
            open_results = pickle.load(open_file)
            open_file.close()
            y_mc_std_relu_resized = torch.from_numpy(open_results[1])
            y_mc_std_relu_resized = y_mc_std_relu_resized.reshape(len(y_mc_std_relu_resized), 1)
            y_mc_std_relu_resized = y_mc_std_relu_resized.repeat((1, 5))
            result_values = {'y_mc_relu': open_results[0], 
                                'y_mc_std_relu': open_results[1],
                                'y_mc_relu_reward': open_results[2],
                                'y_mc_relu_v': open_results[3],
                                'y_mc_relu_P': open_results[4],
                                'y_mc_relu_q': open_results[5],
                                'evd': open_results[6],
                                'run_time': open_results[7],
                                'num_preds': open_results[8],
                                'y_mc_std_relu_resized': y_mc_std_relu_resized}
            
            results[str(world)][no_paths][dropout_val] = result_values

#Print EVD's
evds = []
for path in paths:
    for p_value in dropout_vals:
        evd = results['ow'][path][p_value]['evd']
        print('Paths= ' + str(path) + ' P= ' + str(p_value) + ' EVD = ' + str(evd.item()))

#Get optimal IRL results
irl_result_relu = { 
    'r': results['ow'][optimal_paths][p_value_to_inspect]['y_mc_relu_reward'],
    'v': results['ow'][optimal_paths][p_value_to_inspect]['y_mc_relu_v'],
    'p': results['ow'][optimal_paths][p_value_to_inspect]['y_mc_relu_P'],
    'q': results['ow'][optimal_paths][p_value_to_inspect]['y_mc_relu_q'],
    'r_itr': [results['ow'][optimal_paths][p_value_to_inspect]['y_mc_relu_reward']],
    'model_r_itr': [results['ow'][optimal_paths][p_value_to_inspect]['y_mc_relu_reward']],
    'p_itr': [results['ow'][optimal_paths][p_value_to_inspect]['y_mc_relu_P']],
    'model_p_itr':[results['ow'][optimal_paths][p_value_to_inspect]['y_mc_relu_P']],
    #'time': run_time,
    #'uncertainty': y_mc_std_relu_resized, #leave out so no uncertainty plotted
    'truth_figure_title': 'Truth R & P',
    'pred_reward_figure_title': 'Pred R & P w/ ReLU non-linearities',
    'uncertainty_figure_title': 'Uncertainty w/ ReLU non-linearities'
}

# Ground truth dict for predicitons with ReLU non-linearities
test_result_relu = { 
    'irl_result': irl_result_relu,
    'true_r': r,
    'example_samples': [example_samples],
    'mdp_data': mdp_data,
    'mdp_params': mdp_params,
    'mdp_solution': mdp_solution,
    'feature_data': feature_data
}

#Plot true reward and policy
fig1, ax1 = plt.subplots(1)
objectworlddraw(test_result_relu['true_r'],test_result_relu['mdp_solution']['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig1, ax1)
ax1.set_title('True Reward & Policy')
fig1.savefig(GRAPHS_PATH + "true_reward.png")

#Plot optimal reward and policy
fig1, ax1 = plt.subplots(1)
objectworlddraw(test_result_relu['irl_result']['r'],test_result_relu['mdp_solution']['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig1, ax1)
ax1.set_title('Optimal Est Reward & Policy w/ Dropout = 0.4 & Paths = 128')
fig1.savefig(GRAPHS_PATH + "optimal_est_reward.png")

# Plot reward func for each dropout value
for p_value in dropout_vals:
    fig2, ax1 = plt.subplots(1)
    irl_result_relu = { 
        'r': results['ow'][optimal_paths][p_value]['y_mc_relu_reward'],
        'v': results['ow'][optimal_paths][p_value]['y_mc_relu_v'],
        'p': results['ow'][optimal_paths][p_value]['y_mc_relu_P'],
        'q': results['ow'][optimal_paths][p_value]['y_mc_relu_q'],
        'r_itr': [results['ow'][optimal_paths][p_value]['y_mc_relu_reward']],
        'model_r_itr': [results['ow'][optimal_paths][p_value]['y_mc_relu_reward']],
        'p_itr': [results['ow'][optimal_paths][p_value]['y_mc_relu_P']],
        'model_p_itr':[results['ow'][optimal_paths][p_value]['y_mc_relu_P']],
        #'time': run_time,
        #'uncertainty': y_mc_std_relu_resized, #leave out so no uncertainty plotted
        'truth_figure_title': 'Truth R & P',
        'pred_reward_figure_title': 'Pred R & P w/ ReLU non-linearities',
        'uncertainty_figure_title': 'Uncertainty w/ ReLU non-linearities'
    }

    # Ground truth dict for predicitons with ReLU non-linearities
    test_result_relu = { 
        'irl_result': irl_result_relu,
        'true_r': r,
        'example_samples': [example_samples],
        'mdp_data': mdp_data,
        'mdp_params': mdp_params,
        'mdp_solution': mdp_solution,
        'feature_data': feature_data
    }

    objectworlddraw(test_result_relu['irl_result']['r'],test_result_relu['irl_result']['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig2, ax1)
    ax1.set_title('Dropout = ' + str(p_value) + ' & Paths = ' + str(optimal_paths))
    fig2.savefig(GRAPHS_PATH + str(p_value) +"_reward_predictions.png")





#Plot EVD (Y) vs Paths (X) line graph figure per dropout value (colour)
evd_vals = [[], [], [], [], []] #list for each dropout val
#Get all EVD values
i = 0
for p_value in dropout_vals:
    for path in paths:
        curr_evd = results['ow'][path][p_value]['evd']

        evd_vals[i].append(curr_evd)
    i += 1



# Plot lines ng
fig3, ax1 = plt.subplots()
ax1.plot(paths, evd_vals[0], alpha=0.8, label=list(dropout_vals)[0])
ax1.plot(paths, evd_vals[1], alpha=0.8, label=list(dropout_vals)[1])
ax1.plot(paths, evd_vals[2], alpha=0.8, label=list(dropout_vals)[2])
ax1.plot(paths, evd_vals[3], alpha=0.8, label=list(dropout_vals)[3])
ax1.plot(paths, evd_vals[4], alpha=0.8, label=list(dropout_vals)[4])
ax1.legend(title='Dropout', fontsize='small')
ax1.set_title('EVD vs Paths vs Dropout')
ax1.set_xlabel('Number Of Paths')
ax1.set_ylabel('EVD')

#Save figure
fig3.savefig(GRAPHS_PATH + "evd_vs_paths_vs_dropout.png")