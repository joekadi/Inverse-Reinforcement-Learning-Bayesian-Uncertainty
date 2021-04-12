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

# Compute visible examples.
Eo = torch.zeros(int(mdp_data['states']),1)
for i in range(len(example_samples[0])):
    for t in range(len(example_samples[0][0])):
        Eo[[example_samples][0][i][t][0]] = 1
g = torch.ones(int(mdp_data['states']),1)*0.5+Eo*0.5

#Get directory for results
DROPOUT_RESULTS_PATH = "./regular/results/dropout/"
SWAG_RESULTS_PATH = "./regular/results/swag/"
GPIRL_RESULTS_PATH = "./regular/results/gpirl/"
MAXENT_RESULTS_PATH = "./regular/results/maxent/"


#Create path for graphs 
GRAPHS_PATH = "./regular/results/graphs/"
for path in [GRAPHS_PATH]:
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

#Init results dict
results = {
            'ow': 
            {

                64: {0.0: None, 0.5: None,  1.0: None}, 
                128: {0.0: None, 0.5: None,  1.0: None}, 
                256: {0.0: None, 0.5: None,  1.0: None}, 
                512: {0.0: None, 0.5: None,  1.0: None}, 
                1024: {0.0: None, 0.5: None,  1.0: None}
            }
        }

swag_results = {
            
                'ow': 
                {

                64: {0.0: None}, 
                128: {0.0: None}, 
                256: {0.0: None}, 
                512: {0.0: None}, 
                1024: {0.0: None}
                }
            }

gpirl_results = {
            
                'ow': 
                {

                64: {0.0: None}, 
                128: {0.0: None}, 
                256: {0.0: None}, 
                512: {0.0: None}, 
                1024: {0.0: None}
                }
            }

maxent_results = {
        
            'ow': 
            {

            64: {0.0: None}, 
            128: {0.0: None}, 
            256: {0.0: None}, 
            512: {0.0: None}, 
            1024: {0.0: None}
            }
        }          



worlds = ['ow']
paths = list(results['ow'].keys())
dropout_vals = [0.0, 0.5, 1.0]

#Read in DNN, ensemble & dropout results
for world in worlds:
    for no_paths in paths:
        for dropout_val in dropout_vals:
            file_name = DROPOUT_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results.pkl'
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
                                'y_mc_std_relu_resized': y_mc_std_relu_resized,
                                'evd_variances': open_results[9]}
            
            results[str(world)][no_paths][dropout_val]= result_values

#Read in SWAG results
for world in worlds:
    for no_paths in paths:
        for dropout_val in [0.0]:
            file_name = SWAG_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results.pkl'
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
                                'y_mc_std_relu_resized': y_mc_std_relu_resized,
                                'evd_variances': open_results[9]}
            
            swag_results[str(world)][no_paths][dropout_val] = result_values

#Read in GPIRL results
for world in worlds:
    for no_paths in paths:
        for dropout_val in [0.0]:
            file_name = GPIRL_RESULTS_PATH+str(world)+'_'+str(no_paths)+ '_results.pkl'
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
                                'y_mc_std_relu_resized': y_mc_std_relu_resized,
                                'evd_variances': open_results[9]}
            
            gpirl_results[str(world)][no_paths][0.0] = result_values
            


#Read in MaxEnt results
for world in worlds:
    for no_paths in paths:
        file_name = MAXENT_RESULTS_PATH+str(world)+'_'+ str(no_paths)+ '_results.pkl'
        open_file = open(file_name, "rb")
        open_results = pickle.load(open_file)

        y_mc_std_relu_resized = open_results[1]
        #y_mc_std_relu_resized = y_mc_std_relu_resized.reshape(len(y_mc_std_relu_resized), 1)
        #y_mc_std_relu_resized = y_mc_std_relu_resized.repeat((1, 5))
        result_values = {'y_mc_relu': open_results[0], 
                            'y_mc_std_relu': open_results[1],
                            'y_mc_relu_reward': open_results[2],
                            'y_mc_relu_v': open_results[3],
                            'y_mc_relu_P': open_results[4],
                            'y_mc_relu_q': open_results[5],
                            'evd': open_results[6],
                            'run_time': open_results[7],
                            'num_preds': open_results[8],
                            'y_mc_std_relu_resized': y_mc_std_relu_resized,
                            'evd_variances': open_results[9]}
        maxent_results[str(world)][no_paths][0.0] = result_values



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Draw final Objectworld figures
            
#Optimal dropout IRL result
dropout_irl_result_relu = { 
    'r': results['ow'][1024][1.0]['y_mc_relu_reward'],
    'v': results['ow'][1024][1.0]['y_mc_relu_v'],
    'p': results['ow'][1024][1.0]['y_mc_relu_P'],
    'q': results['ow'][1024][1.0]['y_mc_relu_q'],
    'r_itr': [results['ow'][1024][1.0]['y_mc_relu_reward']],
    'model_r_itr': [results['ow'][1024][1.0]['y_mc_relu_reward']],
    'p_itr': [results['ow'][1024][1.0]['y_mc_relu_P']],
    'model_p_itr':[results['ow'][1024][1.0]['y_mc_relu_P']],
     #'time': run_time,
    'uncertainty': results['ow'][1024][1.0]['y_mc_std_relu_resized'],
    'truth_figure_title': 'Truth R & P',
    'pred_reward_figure_title': 'Pred R & P w/ ReLU non-linearities',
    'uncertainty_figure_title': 'Uncertainty w/ ReLU non-linearities'
}

#Optimal ensemble IRL result
ensemble_irl_result_relu = { 
    'r': results['ow'][1024][0.0]['y_mc_relu_reward'],
    'v': results['ow'][1024][0.0]['y_mc_relu_v'],
    'p': results['ow'][1024][0.0]['y_mc_relu_P'],
    'q': results['ow'][1024][0.0]['y_mc_relu_q'],
    'r_itr': [results['ow'][1024][0.0]['y_mc_relu_reward']],
    'model_r_itr': [results['ow'][1024][0.0]['y_mc_relu_reward']],
    'p_itr': [results['ow'][1024][1.0]['y_mc_relu_P']],
    'model_p_itr':[results['ow'][1024][0.0]['y_mc_relu_P']],
     #'time': run_time,
    'uncertainty': results['ow'][1024][0.0]['y_mc_std_relu_resized'],
    'truth_figure_title': 'Truth R & P',
    'pred_reward_figure_title': 'Pred R & P w/ ReLU non-linearities',
    'uncertainty_figure_title': 'Uncertainty w/ ReLU non-linearities'
}

#Optimal SWAG IRL result
swag_irl_result_relu = { 
    'r': swag_results['ow'][1024][0.0]['y_mc_relu_reward'],
    'v': swag_results['ow'][1024][0.0]['y_mc_relu_v'],
    'p': swag_results['ow'][1024][0.0]['y_mc_relu_P'],
    'q': swag_results['ow'][1024][0.0]['y_mc_relu_q'],
    'r_itr': [swag_results['ow'][1024][0.0]['y_mc_relu_reward']],
    'model_r_itr': [swag_results['ow'][1024][0.0]['y_mc_relu_reward']],
    'p_itr': [swag_results['ow'][1024][0.0]['y_mc_relu_P']],
    'model_p_itr':[swag_results['ow'][1024][0.0]['y_mc_relu_P']],
     #'time': run_time,
    'uncertainty': swag_results['ow'][1024][0.0]['y_mc_std_relu_resized'],
    'truth_figure_title': 'Truth R & P',
    'pred_reward_figure_title': 'Pred R & P w/ ReLU non-linearities',
    'uncertainty_figure_title': 'Uncertainty w/ ReLU non-linearities'
}

#Optimal MaxEnt IRL result
maxent_irl_result_relu = { 
    'r': maxent_results['ow'][1024][0.0]['y_mc_relu_reward'],
    'v': maxent_results['ow'][1024][0.0]['y_mc_relu_v'],
    'p': maxent_results['ow'][1024][0.0]['y_mc_relu_P'],
    'q': maxent_results['ow'][1024][0.0]['y_mc_relu_q'],
    'r_itr': [maxent_results['ow'][1024][0.0]['y_mc_relu_reward']],
    'model_r_itr': [maxent_results['ow'][1024][0.0]['y_mc_relu_reward']],
    'p_itr': [maxent_results['ow'][1024][0.0]['y_mc_relu_P']],
    'model_p_itr':[maxent_results['ow'][1024][0.0]['y_mc_relu_P']],
     #'time': run_time,
    'uncertainty': maxent_results['ow'][1024][0.0]['y_mc_std_relu_resized'],
    'truth_figure_title': 'Truth R & P',
    'pred_reward_figure_title': 'Pred R & P w/ ReLU non-linearities',
    'uncertainty_figure_title': 'Uncertainty w/ ReLU non-linearities'
}

#Optimal GP IRL result
gp_irl_result_relu = { 
    'r': gpirl_results['ow'][1024][0.0]['y_mc_relu_reward'],
    'v': gpirl_results['ow'][1024][0.0]['y_mc_relu_v'],
    'p': gpirl_results['ow'][1024][0.0]['y_mc_relu_P'],
    'q': gpirl_results['ow'][1024][0.0]['y_mc_relu_q'],
    'r_itr': [gpirl_results['ow'][1024][0.0]['y_mc_relu_reward']],
    'model_r_itr': [gpirl_results['ow'][1024][0.0]['y_mc_relu_reward']],
    'p_itr': [gpirl_results['ow'][1024][0.0]['y_mc_relu_P']],
    'model_p_itr':[gpirl_results['ow'][1024][0.0]['y_mc_relu_P']],
     #'time': run_time,
    'uncertainty': gpirl_results['ow'][1024][0.0]['y_mc_std_relu_resized'],
    'truth_figure_title': 'Truth R & P',
    'pred_reward_figure_title': 'Pred R & P w/ ReLU non-linearities',
    'uncertainty_figure_title': 'Uncertainty w/ ReLU non-linearities'
}


#Ground truth dict
test_result_relu = { 
    'irl_result': dropout_irl_result_relu,
    'true_r': r,
    'example_samples': [example_samples],
    'mdp_data': mdp_data,
    'mdp_params': mdp_params,
    'mdp_solution': mdp_solution,
    'feature_data': feature_data
}


#Plot ground truth final figure
fig1, ax1 = plt.subplots(1)
objectworlddraw(test_result_relu['true_r'],test_result_relu['mdp_solution']['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig1, ax1)
ax1.set_title('True Reward & Policy')
fig1.savefig(GRAPHS_PATH + "ground_truth.png")

#Plot optimal dropout final figure
fig1, ax1 = plt.subplots(1)
objectworlddraw(dropout_irl_result_relu['r'],dropout_irl_result_relu['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig1, ax1)
ax1.set_title('MC Dropout')
fig1.savefig(GRAPHS_PATH + "dropout.png")

#Plot optimal ensemble final figure
fig1, ax1 = plt.subplots(1)
objectworlddraw(ensemble_irl_result_relu['r'],ensemble_irl_result_relu['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig1, ax1)
ax1.set_title('Ensemble')
fig1.savefig(GRAPHS_PATH + "ensemble.png")

#Plot optimal ensemble final figure
fig1, ax1 = plt.subplots(1)
objectworlddraw(swag_irl_result_relu['r'],swag_irl_result_relu['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig1, ax1)
ax1.set_title('SWAG')
fig1.savefig(GRAPHS_PATH + "SWAG.png")

#Plot optimal gpirl final figure
fig1, ax1 = plt.subplots(1)
objectworlddraw(gp_irl_result_relu['r'],gp_irl_result_relu['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig1, ax1)
ax1.set_title('GP')
fig1.savefig(GRAPHS_PATH + "gpirl.png")

#Plot optimal ensemble final figure
fig1, ax1 = plt.subplots(1)
objectworlddraw(maxent_irl_result_relu['r'],maxent_irl_result_relu['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig1, ax1)
ax1.set_title('MaxEnt')
fig1.savefig(GRAPHS_PATH + "MaxEnt.png")



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Plot EVDs vs Paths

#To store EVD's for each technique
evd_vals = [[], [], [], [], [], [], []] 
i = 0
for p_value in dropout_vals:
    for path in paths:
        evd_vals[i].append(results['ow'][path][p_value]['evd'].item())
    i += 1
for path in paths:
    evd_vals[3].append(swag_results['ow'][path][0.0]['evd'].item())
    evd_vals[4].append(gpirl_results['ow'][path][0.0]['evd'].item())
    evd_vals[5].append(maxent_results['ow'][path][0.0]['evd'].item())



# Plot evd lines
fig3, ax1 = plt.subplots(figsize=(5,5.5))
ax1.plot(paths, evd_vals[0], alpha=0.8, label="Ensembles", color='blue')
ax1.fill_between(paths, np.array(evd_vals[0])+np.array(results['ow'][1024][0.0]['evd_variances']), np.array(evd_vals[0])-np.array(results['ow'][1024][0.0]['evd_variances']), alpha=0.35, facecolor='blue')
ax1.plot(paths, evd_vals[1], alpha=0.8, label="DNN", color='orange')
ax1.fill_between(paths, np.array(evd_vals[1])+np.array(results['ow'][1024][0.0]['evd_variances']), np.array(evd_vals[1])-np.array(results['ow'][1024][0.0]['evd_variances']), alpha=0.35, facecolor='orange')
ax1.plot(paths, evd_vals[2], alpha=0.8, label="MC Dropout", color='green')
ax1.fill_between(paths, np.array(evd_vals[2])+np.array(results['ow'][1024][1.0]['evd_variances']), np.array(evd_vals[2])-np.array(results['ow'][1024][1.0]['evd_variances']), alpha=0.35, facecolor='green')
ax1.plot(paths, evd_vals[3], alpha=0.8, label="SWAG", color='red')
ax1.fill_between(paths, np.array(evd_vals[3])+np.array(swag_results['ow'][1024][0.0]['evd_variances']), np.array(evd_vals[3])-np.array(swag_results['ow'][1024][0.0]['evd_variances']), alpha=0.35, facecolor='red')
ax1.plot(paths, evd_vals[4], alpha=0.8, label="GPIRL", color='purple')
ax1.fill_between(paths, np.array(evd_vals[4])+np.array(gpirl_results['ow'][1024][0.0]['evd_variances']), np.array(evd_vals[4])-np.array(gpirl_results['ow'][1024][0.0]['evd_variances']), alpha=0.35, facecolor='purple')
ax1.plot(paths, evd_vals[5], alpha=0.8, label="MaxEnt", color='brown')
ax1.fill_between(paths, np.array(evd_vals[5])+np.array(maxent_results['ow'][1024][0.0]['evd_variances'])/2, np.array(evd_vals[5])-np.array(maxent_results['ow'][1024][0.0]['evd_variances'])/2, alpha=0.35, facecolor='brown')
ax1.legend(fontsize='small')
ax1.set_title('Expected Value Difference')
ax1.set_xlabel('Number Of Paths')

#Save figure
fig3.savefig(GRAPHS_PATH + "evds.png")

# Plot zoom in evd lines
fig3, ax1 = plt.subplots(figsize=(5,5.5))

ax1.plot(paths[2:], evd_vals[0][2:], alpha=0.8, label="Ensembles", color='blue')
ax1.fill_between(paths[2:], np.array(evd_vals[0][2:])+np.array(results['ow'][1024][0.0]['evd_variances'][2:]), np.array(evd_vals[0][2:])-np.array(results['ow'][1024][0.0]['evd_variances'][2:]), alpha=0.35, facecolor='blue')
ax1.plot(paths[2:], evd_vals[1][2:], alpha=0.8, label="DNN", color='orange')
ax1.fill_between(paths[2:], np.array(evd_vals[1][2:])+np.array(results['ow'][1024][0.5]['evd_variances'][2:]), np.array(evd_vals[1][2:])-np.array(results['ow'][1024][0.5]['evd_variances'][2:]), alpha=0.35, facecolor='orange')
ax1.plot(paths[2:], evd_vals[2][2:], alpha=0.8, label="MC Dropout", color='green')
ax1.fill_between(paths[2:], np.array(evd_vals[2][2:])+np.array(results['ow'][1024][1.0]['evd_variances'][2:]), np.array(evd_vals[2][2:])-np.array(results['ow'][1024][1.0]['evd_variances'][2:]), alpha=0.35, facecolor='green')
ax1.plot(paths[2:], evd_vals[3][2:], alpha=0.8, label="SWAG", color='red')
ax1.fill_between(paths[2:], np.array(evd_vals[3][2:])+np.array(swag_results['ow'][1024][0.0]['evd_variances'][2:]), np.array(evd_vals[3][2:])-np.array(swag_results['ow'][1024][0.0]['evd_variances'][2:]), alpha=0.35, facecolor='red')

ax1.legend(fontsize='small')
ax1.set_title('Expected Value Difference')
ax1.set_xlabel('Number Of Paths')
#Save figure
fig3.savefig(GRAPHS_PATH + "zoomin_evds.png")



