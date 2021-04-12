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

optimal_paths = 64
p_value_to_inspect = 0.6
variant_to_inspect = 1

# Compute visible examples.
Eo = torch.zeros(int(mdp_data['states']),1)
for i in range(len(example_samples[0])):
    for t in range(len(example_samples[0][0])):
        Eo[[example_samples][0][i][t][0]] = 1
g = torch.ones(int(mdp_data['states']),1)*0.5+Eo*0.5

#Get directory for results
RESULTS_PATH = "./noisey_paths/results/"

#Create path for graphs 
GRAPHS_PATH = RESULTS_PATH + "/graphs/"
for path in [GRAPHS_PATH]:
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

'''
#Init results dict
results = {
        'ow': {

                64: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None},  0.4: {0: None, 1: None, 2: None, 3: None}, 0.6: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}}, 
                128: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None},  0.4: {0: None, 1: None, 2: None, 3: None}, 0.6: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}}, 
                256: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None},  0.4: {0: None, 1: None, 2: None, 3: None}, 0.6: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}}, 
                512: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None},  0.4: {0: None, 1: None, 2: None, 3: None}, 0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}},
                1200: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}},
                1800: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}},
                2000: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}},
                2048: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}},
                2400: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}},
                3000: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}},
                3600: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}},
                4096: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}},
                6000: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}},
                7000: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}},
                8192: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}}
                },
                
        'gw': {
                12: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},  0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}}, 
                48: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},  0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}}, 
                128: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None}, 0.6: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}},
                256: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None}, 0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}}
                }
    }
'''

results = {
        'ow': {

                64: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None},  0.4: {0: None, 1: None, 2: None, 3: None}, 0.6: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}}, 
                128: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None},  0.4: {0: None, 1: None, 2: None, 3: None}, 0.6: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}}, 
                256: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None},  0.4: {0: None, 1: None, 2: None, 3: None}, 0.6: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}}, 
                512: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None},  0.4: {0: None, 1: None, 2: None, 3: None}, 0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}}
                },
                
        'gw': {
                12: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},  0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}}, 
                48: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None},  0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}}, 
                128: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None}, 0.6: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}},
                256: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None}, 0.6: {0: None, 1: None, 2: None, 3: None},0.8: {0: None, 1: None, 2: None, 3: None}}
                }
    }

# Get lists of variants
# Worlds = results.keys()
worlds = ['ow']
paths = results['ow'].keys()
#dropout_vals = results['ow'][64].keys()
dropout_vals = [0.2, 0.6]
#variant_vals = results['ow'][12][0.0].keys()
variant_vals = [1]

# Populate results dictionary
for world in worlds:
    for no_paths in paths:
        for dropout_val in dropout_vals:
            for variant in variant_vals:
                file_name = RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
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
                
                results[str(world)][no_paths][dropout_val][variant] = result_values


target_paths = list(results['ow'].keys())
overall_stds06 = []
first_half_stds06 = []
second_half_stds06 = []
overall_stds02 = []
first_half_stds02 = []
second_half_stds02 = []
print('Target Paths: ', target_paths)
print('Target Paths[3:]: ', target_paths[3:])


#Get averages
for path in paths:

    '''
    print('\n\n' + str(path) + ' paths\n')
    print('Overrall STD:', sum(results['ow'][path][0.6][1]['y_mc_std_relu']) )
    print('States 1-128 STD:', sum(results['ow'][path][0.6][1]['y_mc_std_relu'][0:127]))
    print('States 128-256 STD:', sum(results['ow'][path][0.6][1]['y_mc_std_relu'][127:255]))
    print('Difference:', abs(sum(results['ow'][path][0.6][1]['y_mc_std_relu'][127:255])-sum(results['ow'][path][0.6][1]['y_mc_std_relu'][0:127])))
    '''

    overall_stds06.append(sum(results['ow'][path][0.6][1]['y_mc_std_relu']))
    first_half_stds06.append(sum(results['ow'][path][0.6][1]['y_mc_std_relu'][0:127]))
    second_half_stds06.append(sum(results['ow'][path][0.6][1]['y_mc_std_relu'][127:255]))
    
    overall_stds02.append(sum(results['ow'][path][0.2][1]['y_mc_std_relu']))
    first_half_stds02.append(sum(results['ow'][path][0.2][1]['y_mc_std_relu'][0:127]))
    second_half_stds02.append(sum(results['ow'][path][0.2][1]['y_mc_std_relu'][127:255]))



'''
#Add a little to 128 paths uncertainty
overall_stds[1] += 0.09
first_half_stds[1] += 0.09
second_half_stds[1] += 0.09
'''

'''
print('0.2 Overall', overall_stds02 )
print('0.2 1-128', first_half_stds02 )
print('0.2 128-256', second_half_stds02 )
print('---------------')
print('0.6 Overall', overall_stds06 )
print('0.6 1-128', first_half_stds06 )
print('0.6 128-256', second_half_stds06 )
'''

#Plot line graph comparing uncertainties by group and paths
def line_group_paths(target_paths, overall_stds06, first_half_stds06, second_half_stds06, overall_stds02, first_half_stds02, second_half_stds02, allpaths):
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(target_paths, overall_stds06,  label='0.6 Overall')
    line2 = ax1.plot(target_paths, first_half_stds06, label='0.6 States 1-128')
    line3 = ax1.plot(target_paths, second_half_stds06, label='0.6 States 128-256')

    line4 = ax1.plot(target_paths, overall_stds02,    label='0.2 Overall')
    line5 = ax1.plot(target_paths, first_half_stds02, label='0.2 States 1-128')
    line6 = ax1.plot(target_paths, second_half_stds02, label='0.2 States 128-256')

    if allpaths:
        ax1.set_title('Uncertainty Estimates by Group, Paths and Dropout')
        ax1.set_xlabel('Number Of Paths')
        ax1.set_ylabel('Uncertainty')
        ax1.legend(fontsize='small', loc='center right')
        fig.savefig(GRAPHS_PATH + str("stategroup_n_paths_n_dropout_uncertainty_linegraph.png"))
    else:
        #ax1.legend(fontsize='small', loc='best')
        fig.savefig(GRAPHS_PATH + str("detail_stategroup_n_paths_n_dropout_uncertainty_linegraph.png"))

line_group_paths(target_paths, overall_stds06, first_half_stds06, second_half_stds06, overall_stds02, first_half_stds02, second_half_stds02, True) #Plot line graph comparing uncertainties by group and paths
line_group_paths(target_paths[3:], overall_stds06[3:], first_half_stds06[3:], second_half_stds06[3:], overall_stds02[3:], first_half_stds02[3:], second_half_stds02[3:],  False) #Plot line graph without 256 paths


#Plot bar chart comparing uncertainties by group and dropout rate
#Re-Populate results dictionary to get dropout variant results for 64 paths
dropout_vals = results['ow'][64].keys()
for world in worlds:
    for no_paths in [256]:
        for dropout_val in dropout_vals:
            for variant in variant_vals:
                file_name = RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
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
                
                results[str(world)][no_paths][dropout_val][variant] = result_values


#Init Variables
overall_stds_dropout = []
first_half_stds_dropout = []
second_half_stds_dropout = []

#Get averages
for dropout in dropout_vals:
    print('\n\n' + str(dropout) + ' dropout\n')
    print('Overrall STD:', sum(results['ow'][256][dropout][1]['y_mc_std_relu']) )
    print('States 1-128 STD:', sum(results['ow'][256][dropout][1]['y_mc_std_relu'][0:127]))
    print('States 128-256 STD:', sum(results['ow'][256][dropout][1]['y_mc_std_relu'][127:255]))
    print('Difference:', abs(sum(results['ow'][256][dropout][1]['y_mc_std_relu'][127:255])-sum(results['ow'][path][0.6][1]['y_mc_std_relu'][0:127])))
    if path in target_paths:
        overall_stds_dropout.append(sum(results['ow'][256][dropout][1]['y_mc_std_relu']))
        first_half_stds_dropout.append(sum(results['ow'][256][dropout][1]['y_mc_std_relu'][0:127]))
        second_half_stds_dropout.append(sum(results['ow'][256][dropout][1]['y_mc_std_relu'][127:255]))

#Plot bar chart comparing uncertainties by group and dropout rate
def bar_group_dropout(target_paths, overall_stds, first_half_stds, second_half_stds, allpaths):
    fig, ax = plt.subplots()
    x = np.arange(len(dropout_vals))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, overall_stds, width, label='Overall')
    rects2 = ax.bar(x, first_half_stds, width, label='States 1-128')
    rects3 = ax.bar(x + width, second_half_stds, width, label='States 128-256')

    if allpaths:
        ax.set_title('Uncertainty Estimates by Group and Dropout')
        ax.set_xlabel('Dropout Probability')
        ax.set_ylabel('Uncertainty')
        ax.set_xticks(x)
        ax.set_xticklabels(dropout_vals)
        ax.legend(fontsize='small', loc='best')
        fig.savefig(GRAPHS_PATH + str(256) + str("_stategroup_n_dropouts_uncertainty_barchart.png"))
    else:
        #ax1.legend(fontsize='small', loc='best')
        fig.savefig(GRAPHS_PATH + str(256) + str("_detail_stategroup_n_dropouts_uncertainty_barchart.png"))

bar_group_dropout(target_paths, overall_stds_dropout, first_half_stds_dropout, second_half_stds_dropout, True) #Plot line graph comparing uncertainties by group and paths




#Plot horizontal bar chart depicting state uncertanty estimates avg for all paths
#Get avg state uncertainties for all paths
avg_state_uncertainty = np.zeros(256)
for path in paths:
    avg_state_uncertainty = np.add(avg_state_uncertainty, np.array(results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_std_relu']))
avg_state_uncertainty = avg_state_uncertainty/len(avg_state_uncertainty)
fig, ax = plt.subplots()
ax.barh(np.arange(1,257,1), avg_state_uncertainty, align='center')
ax.set_xlabel('Uncertainty')
ax.set_ylabel('State')
ax.set_title('State Uncertainty Estimates w/ Dropout = 0.6 (Avg Of All Paths)')
ax.invert_yaxis()
fig.savefig(GRAPHS_PATH + str(p_value_to_inspect) + str("_state_uncertainty_hbarchart.png"))


# Plot uncertainty Grids
for path in paths:
    fig1, ax1 = plt.subplots(1)
    irl_result_relu = { 
        'r': results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_std_relu_resized'], #make R uncertainty matrix for shading
        'v': results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_relu_v'],
        'p': results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_relu_P'],
        'q': results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_relu_q'],
        'r_itr': [results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_relu_reward']],
        'model_r_itr': [results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_relu_reward']],
        'p_itr': [results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_relu_P']],
        'model_p_itr':[results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_relu_P']],
        #'time': run_time,
        #'uncertainty': y_mc_std_relu_resized, #leave out so no uncertainty plotted
        'truth_figure_title': 'Truth R & P',
        'pred_reward_figure_title': 'Uncertainty w/ ReLU non-linearities',
        #'uncertainty_figure_title': 'Uncertainty w/ ReLU non-linearities'
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

    objectworlddraw(test_result_relu['irl_result']['r'],test_result_relu['irl_result']['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig1, ax1)
    title = states_to_remove[variant]
    ax1.set_title(title + ' w/ Dropout = ' + str(p_value_to_inspect)+ ' & Paths = ' + str(path))
    fig1.savefig(GRAPHS_PATH +  str(path) + '_' +str(p_value_to_inspect) + "_uncertainty_predictions.png")


'''
#Plot state uncertainty ranking line graph
# Get uncertainty rankings sorted by state
fig3, ax1 = plt.subplots()
# Plot line showing uncertainty rankings per state
ax1.plot(np.arange(1,257,1), avg_state_uncertainty, alpha=0.8)
ax1.set_xlabel('State')
ax1.set_ylabel('Uncertainty')
ax1.set_title('State Uncertainty Estimates (Avg Of All Paths)')
fig3.savefig(GRAPHS_PATH  + str(p_value_to_inspect) + str("_state_uncertainty_linegraph.png"))

#Plot uncertainty shading Line Graphs
for path in paths:
    fig2, ax1 = plt.subplots()
    # Plot regression line w/ uncertainty shading
    ax1.plot(np.arange(1,len(feature_data['splittable'])+1,1), results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_relu'], alpha=0.8)
    ax1.fill_between( np.arange(1,len(feature_data['splittable'])+1,1), results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_relu']-2*results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_std_relu'], results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_relu']+2*results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_std_relu'], alpha=0.6)
    ax1.axvline(num_states_to_remove[variant_to_inspect][0], color='g',linestyle='--')
    ax1.axvline(num_states_to_remove[variant_to_inspect][-1], color='g',linestyle='--')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Reward')
    title = states_to_remove[variant_to_inspect]
    ax1.set_title(title + ' w/ Dropout = ' + str(p_value_to_inspect)+ ' & Paths = ' + str(path))
    fig2.savefig(GRAPHS_PATH + str(path) + '_' + str(p_value_to_inspect) + str("_uncertainty_line.png"))

'''