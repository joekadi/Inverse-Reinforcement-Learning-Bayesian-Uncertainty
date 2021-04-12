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
from likelihood import scale_results
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


tensorboard_writer = SummaryWriter('./tensorboard_logs')
torch.set_printoptions(precision=5, threshold=1000)
torch.set_default_tensor_type(torch.DoubleTensor)

num_states_to_remove = [None, np.arange(0, 32, 1), np.arange(0, 64, 1), np.arange(0, 128, 1)]
states_to_remove = ['No Noise', 'States 1-32 Removed From Paths', 'States 1-64 Removed From Paths', 'States 1-128 Removed From Paths']

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
RESULTS_PATH = "./noisey_features/results/swag/"

#Create path for graphs 
GRAPHS_PATH = RESULTS_PATH + "/graphs/"
for path in [GRAPHS_PATH]:
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


results = {
        'ow': {

                64: {0.0: {0: None, 1: None, 2: None, 3: None}}, 
                128: {0.0: {0: None, 1: None, 2: None, 3: None}}, 
                256: {0.0: {0: None, 1: None, 2: None, 3: None}}, 
                512: {0.0: {0: None, 1: None, 2: None, 3: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None, 3: None}}
            }
            }



#Get lists of variants
worlds = ['ow']

paths = list(results['ow'].keys())
dropout_vals = [0.0]
variant_vals = list(results['ow'][paths[0]][dropout_vals[0]].keys())



#Populate results dict with results from files
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

# ----------------------------------------------------------------------------------------------------------------------------------------------
#Calc & plot Expected Policy Diference Uncertainty Error (EDPUE) line graphs

'''
TODO
- Calc & plot for 
    - GP
'''

#Calculate and store avg uncertainties and policies
swag_uncertainties_per_paths = []
swag_policies_per_paths = []
for path in paths:
    swag_agg_policy = torch.zeros(256,5)
    swag_agg_uncertainty = torch.zeros(256,1)
    swag_epdue = 0.0
    for variant in variant_vals:
        swag_policy = results['ow'][path][0.0][variant]['y_mc_relu_P']
        swag_uncertainty = torch.tensor(results['ow'][path][0.0][variant]['y_mc_std_relu'])    
        swag_uncertainty = swag_uncertainty.reshape(len(swag_uncertainty), 1)
        swag_agg_policy += swag_policy
        swag_agg_uncertainty += swag_uncertainty
    swag_uncertainties_per_paths.append(swag_agg_uncertainty/4)
    swag_policies_per_paths.append(swag_agg_policy/4)

#Calc avg policies
swag_avg_policy = sum(swag_policies_per_paths)/len(swag_policies_per_paths)
#Calc avg uncertainty
swag_avg_uncertainty = sum(swag_uncertainties_per_paths)/len(swag_uncertainties_per_paths)
#Calc avg EPD
swag_avg_epd = torch.mean(torch.abs(swag_avg_policy-mdp_solution['p']) , 1, True)

#Calc average epudes
swag_epdue = epdue(swag_avg_uncertainty.detach().numpy(), swag_avg_epd.detach().numpy())

# Scale to better see differences in visuals
scaler = MinMaxScaler()
swag_avg_epd = scaler.fit_transform(swag_avg_epd)
swag_avg_uncertainty = scaler.fit_transform(swag_avg_uncertainty)
   

#Plot
fig, ax = plot_epdue(swag_avg_uncertainty, swag_avg_epd)
textstr = 'EPDUE = ' + str(round(swag_epdue*100, 4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.98, 0.81, textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
ax.set_title('SWAG')       
ax.legend(loc='upper right', prop={'size': 6})
fig.tight_layout()
fig.savefig(GRAPHS_PATH + "swag_epdue.png")




# ----------------------------------------------------------------------------------------------------------------------------------------------
#Calc & plot Expected Normalized Calibration Error (ENCE) line graphs

'''
TODO
- Calc & plot for 
    - GP
'''


swag_ences_per_path = []
swag_uncertainties_per_paths = []
swag_policies_per_paths = []

#Calculate, store and print summed ence for path by variant
for path in paths:
    agg_policy = torch.zeros(256,5)
    agg_uncertainty = torch.zeros(256)
    swag_ence = 0.0
    for dropout_val in dropout_vals:
        for variant in variant_vals:
            policy = results['ow'][path][dropout_val][variant]['y_mc_relu_P']
            uncertainty = torch.tensor(results['ow'][path][dropout_val][variant]['y_mc_std_relu'])     
            scaler = MinMaxScaler()
            policy = scaler.fit_transform(policy)
            uncertainty = scaler.fit_transform(uncertainty.reshape(len(uncertainty), 1))
            
            agg_policy += policy
            agg_uncertainty += uncertainty

            swag_ence += (ence(mdp_solution['p'].squeeze(), policy.squeeze(), uncertainty.squeeze()))
        print('Path :', path, 'P val: ', dropout_val, 'Variant: ', variant, 'ence: ', swag_ence/4)
        swag_ences_per_path.append(swag_ence/4)
        swag_uncertainties_per_paths.append(agg_uncertainty/4)
        swag_policies_per_paths.append(agg_policy/4)


avg_ence = sum(swag_ences_per_path)/len(swag_ences_per_path)
print('Avg ence = ', avg_ence)
avg_policy = (sum(swag_policies_per_paths)/len(swag_policies_per_paths)).squeeze()
avg_uncertainty = (sum(swag_uncertainties_per_paths)/len(swag_uncertainties_per_paths)).squeeze().detach().numpy()


#Calculate ence from avg policies and avg uncertainties
swag_ence = ence(mdp_solution['p'].squeeze(), avg_policy, avg_uncertainty)*1000
print('SWAG ence = ', swag_ence)

#Plot ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), avg_policy.squeeze(), avg_uncertainty.squeeze())
textstr = 'ENCE = ' + str(round(swag_ence,3))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.45, 0.96,textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
ax.set_title('SWAG')   
ax.legend(loc='lower right', prop={'size': 6})    
fig.tight_layout()
fig.savefig(GRAPHS_PATH + "swag_ence.png")


#----------------------------------------------------------------------------------------------------------------------------------------------
'''
TODO
- Calc and print for 
    - GP
'''

# Calculate Coefficient of Variation (Cv) for STDS
# Useful for the case when ENCE = 0

scaler = MinMaxScaler()


swag_uncertainty = scaler.fit_transform(swag_uncertainty.reshape(len(swag_uncertainty), 1))

#Perhaps divide by len of set to keep within 0-1? Currently 
swag_cv = variation(swag_uncertainty)
print('SWAG coefficent of variation = ', swag_cv)