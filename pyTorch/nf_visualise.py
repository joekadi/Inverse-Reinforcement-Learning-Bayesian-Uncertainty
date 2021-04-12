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

# Compute visible examples.
Eo = torch.zeros(int(mdp_data['states']),1)
for i in range(len(example_samples[0])):
    for t in range(len(example_samples[0][0])):
        Eo[[example_samples][0][i][t][0]] = 1
g = torch.ones(int(mdp_data['states']),1)*0.5+Eo*0.5

#Get directory for results
RESULTS_PATH = "./noisey_features/results/dropout/"
SWAG_RESULTS_PATH = "./noisey_features/results/swag/"

#Create path for graphs 
GRAPHS_PATH = "./noisey_features/results/graphs/"
for path in [GRAPHS_PATH]:
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

results = {
        'ow': {

                64: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.5: {0: None, 1: None, 2: None, 3: None},  1.0: {0: None, 1: None, 2: None, 3: None}}, 
                128: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.5: {0: None, 1: None, 2: None, 3: None},  1.0: {0: None, 1: None, 2: None, 3: None}}, 
                256: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.5: {0: None, 1: None, 2: None, 3: None},  1.0: {0: None, 1: None, 2: None, 3: None}}, 
                512: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.5: {0: None, 1: None, 2: None, 3: None},  1.0: {0: None, 1: None, 2: None, 3: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.5: {0: None, 1: None, 2: None, 3: None},  1.0: {0: None, 1: None, 2: None, 3: None}}
            }
            }


swag_results = {
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
dropout_vals = [0.0, 0.5, 1.0]
variant_vals = list(results['ow'][paths[0]][dropout_vals[0]].keys())

#Populate results dict with dropout results from files
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

#Populate results dict with swag results from files
for world in worlds:
    for no_paths in paths:
        for dropout_val in [0.0]:
            for variant in variant_vals:
                file_name = SWAG_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
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
                
                swag_results[str(world)][no_paths][dropout_val][variant] = result_values

# ----------------------------------------------------------------------------------------------------------------------------------------------

#Calculate and store avg uncertainties and policies for ensemble, dropout & swag

'''
TODO
- Calc for 
    - GP
'''

dropout_uncertainties_per_paths = []
dropout_policies_per_paths = []

ensembles_uncertainties_per_paths = []
ensembles_policies_per_paths = []

for path in paths:
    dropout_agg_policy = torch.zeros(256,5)
    dropout_agg_uncertainty = torch.zeros(256,1)
    dropout_epdue = 0.0

    ensemble_agg_policy = torch.zeros(256,5)
    ensemble_agg_uncertainty = torch.zeros(256,1)
    ensemble_epdue = 0.0

    for variant in variant_vals:
        dropout_policy = results['ow'][path][1.0][variant]['y_mc_relu_P']
        dropout_uncertainty = torch.tensor(results['ow'][path][1.0][variant]['y_mc_std_relu'])    
        '''
        # Use avg results from 1.0 and 0.5 dropout p values as dropout result
        dropout_policy = (results['ow'][path][1.0][variant]['y_mc_relu_P']+results['ow'][path][0.5][variant]['y_mc_relu_P'])/2
        dropout_uncertainty = torch.tensor((results['ow'][path][1.0][variant]['y_mc_std_relu']+results['ow'][path][1.0][variant]['y_mc_std_relu'])/2)    
        '''
        ensemble_policy = results['ow'][path][0.0][variant]['y_mc_relu_P']
        ensemble_uncertainty = torch.tensor(results['ow'][path][0.0][variant]['y_mc_std_relu'])    
        dropout_uncertainty = dropout_uncertainty.reshape(len(dropout_uncertainty), 1)
        ensemble_uncertainty = ensemble_uncertainty.reshape(len(ensemble_uncertainty), 1)
        dropout_agg_policy += dropout_policy
        dropout_agg_uncertainty += dropout_uncertainty
        ensemble_agg_policy += ensemble_policy
        ensemble_agg_uncertainty += ensemble_uncertainty

    dropout_uncertainties_per_paths.append(dropout_agg_uncertainty/4)
    dropout_policies_per_paths.append(dropout_agg_policy/4)

    ensembles_uncertainties_per_paths.append(ensemble_agg_uncertainty/4)
    ensembles_policies_per_paths.append(ensemble_agg_policy/4)

#Calc avg policies
dropout_avg_policy = sum(dropout_policies_per_paths)/len(dropout_policies_per_paths)
ensembles_avg_policy = sum(ensembles_policies_per_paths)/len(ensembles_policies_per_paths)

#Calc avg uncertainty
dropout_avg_uncertainty = sum(dropout_uncertainties_per_paths)/len(dropout_uncertainties_per_paths)
ensembles_avg_uncertainty = sum(ensembles_uncertainties_per_paths)/len(ensembles_uncertainties_per_paths)

#Calculate and store avg swag uncertainties and policies
swag_uncertainties_per_paths = []
swag_policies_per_paths = []
for path in paths:
    swag_agg_policy = torch.zeros(256,5)
    swag_agg_uncertainty = torch.zeros(256,1)
    swag_epdue = 0.0
    for variant in variant_vals:
        swag_policy = swag_results['ow'][path][0.0][variant]['y_mc_relu_P']
        swag_uncertainty = torch.tensor(swag_results['ow'][path][0.0][variant]['y_mc_std_relu'])    
        swag_uncertainty = swag_uncertainty.reshape(len(swag_uncertainty), 1)
        swag_agg_policy += swag_policy
        swag_agg_uncertainty += swag_uncertainty
    swag_uncertainties_per_paths.append(swag_agg_uncertainty/4)
    swag_policies_per_paths.append(swag_agg_policy/4)

#Calc avg policies
swag_avg_policy = sum(swag_policies_per_paths)/len(swag_policies_per_paths)
#Calc avg uncertainty
swag_avg_uncertainty = sum(swag_uncertainties_per_paths)/len(swag_uncertainties_per_paths)


# ----------------------------------------------------------------------------------------------------------------------------------------------
# Plot Expected Policy Diference Uncertainty Error (EDPUE)


'''
TODO
- Plot for 
    - GP
'''

#Calc avg EPDs on unscaled values for more accurate results
dropout_avg_epd = torch.mean(torch.abs(dropout_avg_policy-mdp_solution['p']) , 1, True)
ensemble_avg_epd = torch.mean(torch.abs(ensembles_avg_policy-mdp_solution['p']) , 1, True)
swag_avg_epd = torch.mean(torch.abs(swag_avg_policy-mdp_solution['p']) , 1, True)


#Calc average epudes
dropout_epdue = epdue(dropout_avg_uncertainty.detach().numpy(), dropout_avg_epd.detach().numpy())
ensemble_epdue = epdue(ensembles_avg_uncertainty.detach().numpy(), ensemble_avg_epd.detach().numpy())
swag_epdue = epdue(swag_avg_uncertainty.detach().numpy(), swag_avg_epd.detach().numpy())


#Scale to better see differences in visuals
scaler = MinMaxScaler()
ensemble_avg_epd = scaler.fit_transform(ensemble_avg_epd)
scaled_ensembles_avg_uncertainty = scaler.fit_transform(ensembles_avg_uncertainty)
  


#Plot ensemble EPDUE
fig, ax = plot_epdue(scaled_ensembles_avg_uncertainty, ensemble_avg_epd)
textstr = 'EPDUE = ' + str(round(ensemble_epdue*100,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.98, 0.83, textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
ax.set_title('Ensemble')
ax.legend(loc='upper right', prop={'size': 6})       
fig.tight_layout()
fig.savefig(GRAPHS_PATH + "ensemble_epdue.png")



# Scale to better see differences in visuals
dropout_avg_epd = scaler.fit_transform(dropout_avg_epd)
scaled_dropout_avg_uncertainty = scaler.fit_transform(dropout_avg_uncertainty)

#Plot dropout EPDUE
fig, ax = plot_epdue(scaled_dropout_avg_uncertainty, dropout_avg_epd)
textstr = 'EPDUE = ' + str(round(dropout_epdue*100, 4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.98, 0.83, textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
ax.set_title('MC Dropout')
ax.legend(loc='upper right', prop={'size': 6})       
fig.tight_layout()
fig.savefig(GRAPHS_PATH + "mcdropout_epdue.png")

# Scale to better see differences in visuals
scaler = MinMaxScaler()
swag_avg_epd = scaler.fit_transform(swag_avg_epd)
scaled_swag_avg_uncertainty = scaler.fit_transform(swag_avg_uncertainty)
   

#Plot
fig, ax = plot_epdue(scaled_swag_avg_uncertainty, swag_avg_epd)
textstr = 'EPDUE = ' + str(round(swag_epdue*100, 4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.98, 0.83, textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
ax.set_title('SWAG')       
ax.legend(loc='upper right', prop={'size': 6})
fig.tight_layout()
fig.savefig(GRAPHS_PATH + "swag_epdue.png")



# ----------------------------------------------------------------------------------------------------------------------------------------------
# Plot Expected Normalized Calibration Error (ENCE) line graphs

'''
TODO
- Plot for 
    - GP
'''

'''
# Scale policies for more meaningful comparison since uncertainties already scaled
ensembles_avg_policy = scaler.fit_transform(ensembles_avg_policy)
dropout_avg_policy = scaler.fit_transform(dropout_avg_policy)
swag_avg_policy = scaler.fit_transform(swag_avg_policy)
'''

#Calculate ences from avg policies and avg uncertainties
ensemble_ence = ence(mdp_solution['p'].squeeze(), ensembles_avg_policy.detach().numpy(), ensembles_avg_uncertainty.detach().numpy())
dropout_ence = ence(mdp_solution['p'].squeeze(), dropout_avg_policy.detach().numpy(), dropout_avg_uncertainty.detach().numpy())
swag_ence = ence(mdp_solution['p'].squeeze(), swag_avg_policy.detach().numpy(), swag_avg_uncertainty.detach().numpy())



#Plot ensembles ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), ensembles_avg_policy.detach().numpy(), ensembles_avg_uncertainty.detach().numpy())
textstr = 'ENCE = ' + str(round(ensemble_ence,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.98, 0.2,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
ax.set_title('Ensemble')      
ax.legend(loc='lower right', prop={'size': 6}) 
fig.tight_layout()
fig.savefig(GRAPHS_PATH + "ensemble_ence.png")


#Plot dropout ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), dropout_avg_policy.detach().numpy(), dropout_avg_uncertainty.detach().numpy())
textstr = 'ENCE = ' + str(round(dropout_ence,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.98, 0.2,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
ax.set_title('MC Dropout')   
ax.legend(loc='lower right', prop={'size': 6})    
fig.tight_layout()
fig.savefig(GRAPHS_PATH + "mcdropout_ence.png")


#Plot swag ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), swag_avg_policy.detach().numpy(), swag_avg_uncertainty.detach().numpy())
textstr = 'ENCE = ' + str(round(swag_ence,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.98, 0.2,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
ax.set_title('SWAG')   
ax.legend(loc='lower right', prop={'size': 6})    
fig.tight_layout()
fig.savefig(GRAPHS_PATH + "swag_ence.png")


os._exit(1)

#----------------------------------------------------------------------------------------------------------------------------------------------

'''
TODO
- Calc and print for 
    - GP
'''

# Calculate Coefficient of Variation (Cv) for STDS
# Useful for the case when ENCE = 0

scaler = MinMaxScaler()


dropout_uncertainty = scaler.fit_transform(dropout_uncertainty.reshape(len(dropout_uncertainty), 1))
ensemble_uncertainty = scaler.fit_transform(ensemble_uncertainty.reshape(len(ensemble_uncertainty), 1))


#Perhaps divide by len of set to keep within 0-1? Currently 
dropout_cv = variation(dropout_uncertainty)
ensemble_cv = variation(ensemble_uncertainty)

print('Dropout coefficent of variation = ', dropout_cv)
print('Ensemble coefficent of variation = ', ensemble_cv)


# ----------------------------------------------------------------------------------------------------------------------------------------------

# Plot uncertainty for each technique

'''
TODO:
- Layer GP estimates
'''
swag_uncertainties_per_paths = []

#Calculate and store avg uncertainty for SWAG
for path in paths:
    swag_agg_uncertainty = torch.zeros(256,1)
    for variant in variant_vals:
        swag_policy = swag_results['ow'][path][0.0][variant]['y_mc_relu_P']
        swag_uncertainty = torch.tensor(swag_results['ow'][path][0.0][variant]['y_mc_std_relu'])    
        swag_uncertainty = swag_uncertainty.reshape(len(swag_uncertainty), 1)
        swag_agg_uncertainty += swag_uncertainty
    swag_uncertainties_per_paths.append(swag_agg_uncertainty/4)

#Calc avg uncertainty & scale
swag_avg_uncertainty = sum(swag_uncertainties_per_paths)/len(swag_uncertainties_per_paths)
swag_avg_uncertainty = scaler.fit_transform(swag_avg_uncertainty)



#List to store total uncertainty for each path values for each technique
uncertainty_vals = [ensemble_avg_uncertainty, dropout_avg_uncertainty, swag_avg_uncertainty] 


print('\n\nUncertainty shapes for each method\n')
for i in range(len(uncertainty_vals)):
    print(i, ': ', uncertainty_vals[i].shape)

    
'''
#Populate list with dropout results
i = 0
curr_var = 0
for p_value in dropout_vals:
    for path in paths:
        for variant in variants_to_plot:
            curr_var += np.mean(results['ow'][path][p_value][variant]['y_mc_std_relu'])
        uncertainty_vals[i].append(curr_var)
    i += 1
#Populate list with SWAG results
i = 3
curr_var = 0
for p_value in [0.0]:
    for path in paths:
        for variant in variants_to_plot:
            curr_var += np.mean(swag_results['ow'][path][p_value][variant]['y_mc_std_relu'])
        uncertainty_vals[i].append(curr_var)
    i += 1

print('\n\nUncertainty values for each method\n')
for i in range(len(uncertainty_vals)):
    print(i, ': ', uncertainty_vals[i])
    uncertainty_vals[i] = uncertainty_vals[i][::-1]

'''

#Plot line graph
fig, ax1 = plt.subplots()
ax1.plot(paths, uncertainty_vals[0], alpha=0.8, label='Ensembles')
ax1.plot(paths, uncertainty_vals[1], alpha=0.8, label='MC Dropout')
ax1.plot(paths, uncertainty_vals[2], alpha=0.8, label='SWAG')
ax1.legend(fontsize='small')
ax1.set_title('Total Uncertainty as F(Paths)')
ax1.set_xlabel('Number Of Paths')
ax1.set_ylabel('Uncertainty')
#Save figure
fig.savefig(GRAPHS_PATH + "std_per_technique.png")



#Scale everything within 0 and 1
scaler = MinMaxScaler()
scaled_uncertainty_vals = scaler.fit_transform(uncertainty_vals)
# Plot scaled line graph
fig, ax1 = plt.subplots()
ax1.plot(paths, uncertainty_vals[0], alpha=0.8, label='Ensembles')
ax1.plot(paths, uncertainty_vals[1], alpha=0.8, label='MC Dropout')
ax1.plot(paths, uncertainty_vals[2], alpha=0.8, label='SWAG')
ax1.legend(fontsize='small')
ax1.set_title('Scaled Total Uncertainty as F(Paths)')
ax1.set_xlabel('Number Of Paths')
ax1.set_ylabel('Uncertainty')
#Save figure
fig.savefig(GRAPHS_PATH + "scaled_std_per_technique.png")


os._exit(1)
# ----------------------------------------------------------------------------------------------------------------------------------------------
#Plot Uncertainty (Y) vs Paths (X) line graph figure for each variant (colour) for dropout = 0.5 (solid line) and GP (dotted line) 
#To show that it is noise causing uncertainty calibration. Expected to see no noise line eventually reach 0

'''
TODO:
- Layer GP estimates
'''

uncertainty_0 = []
uncertainty_1 = []
uncertainty_2 = []
uncertainty_3 = []




#Get each variants total uncertainty
for path in paths:
    #uncertainty = mean dropout + mean SWAG + mean ensemble
    uncertainty_0.append(np.mean(results['ow'][path][1.0][0]['y_mc_std_relu'])+np.mean(swag_results['ow'][path][0.0][0]['y_mc_std_relu'])+np.mean(results['ow'][path][0.0][0]['y_mc_std_relu']))
    uncertainty_1.append(np.mean(results['ow'][path][1.0][1]['y_mc_std_relu'])+np.mean(swag_results['ow'][path][0.0][1]['y_mc_std_relu'])+np.mean(results['ow'][path][0.0][0]['y_mc_std_relu']))
    uncertainty_2.append(np.mean(results['ow'][path][1.0][2]['y_mc_std_relu'])+np.mean(swag_results['ow'][path][0.0][2]['y_mc_std_relu'])+np.mean(results['ow'][path][0.0][0]['y_mc_std_relu']))
    uncertainty_3.append(np.mean(results['ow'][path][1.0][3]['y_mc_std_relu'])+np.mean(swag_results['ow'][path][0.0][3]['y_mc_std_relu'])+np.mean(results['ow'][path][0.0][0]['y_mc_std_relu']))

uncertainty_variant_vals = [uncertainty_0, uncertainty_1, uncertainty_2, uncertainty_3]
print('\n\nUncertainty values for each variant\n')
for i in range(len(uncertainty_variant_vals)):
    print(states_to_remove[i], ': ', uncertainty_variant_vals[i])

print('----------------------------------------------------------------------------------------------------------------------------------------------')
#Func to plot line graph comparing uncertainties by variant as func of paths for Dropout = 0.5 and GP
def line_group_paths(target_paths, uncertainty_0, uncertainty_1, uncertainty_2, uncertainty_3, scaled):
    fig, ax1 = plt.subplots()

    ax1.plot(target_paths, uncertainty_0,  label='No Noise')
    ax1.plot(target_paths, uncertainty_1, label='States 1-32')
    ax1.plot(target_paths, uncertainty_2, label='States 1-64')
    ax1.plot(target_paths, uncertainty_3, label='States 1-128')
 
    ax1.set_xlabel('Number Of Paths')
    ax1.set_ylabel('Uncertainty')
    ax1.legend(fontsize='small', loc='best')
    if scaled:
        ax1.set_title('Scaled Total Uncertainty F(paths) SUM(SWAG+Dropout+Ensembles)')
        fig.savefig(GRAPHS_PATH + "scaled_std_per_variant.png") 
    else:
        ax1.set_title('Total Uncertainty F(paths) SUM(SWAG+Dropout+Ensembles)')
        fig.savefig(GRAPHS_PATH + "std_per_variant.png") 

#Create order for plotting
#ordered_uncertainty_variant_vals = [uncertainty_2, uncertainty_0, uncertainty_1, uncertainty_3]

#Real order
ordered_uncertainty_variant_vals = [uncertainty_0, uncertainty_1, uncertainty_2, uncertainty_3]
ordered_uncertainty_variant_vals = scale_results(ordered_uncertainty_variant_vals)
#Plot uncertainties vals
line_group_paths(paths, ordered_uncertainty_variant_vals[0], ordered_uncertainty_variant_vals[1], ordered_uncertainty_variant_vals[2], ordered_uncertainty_variant_vals[3], False)




#Scale everything uncertainties
scaled_uncertainties= scaler.fit_transform(uncertainty_variant_vals)
#Replot w/ scaled values
line_group_paths(paths, scaled_uncertainties[0], scaled_uncertainties[1], scaled_uncertainties[2], scaled_uncertainties[3], True)


# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------





"""

'''
TODO:
- Get uncertainty actually showing up AND calibrated 
- Make one for GP estimates
'''

#Plot regression line showing calibrated uncertainty for states through line of reward and uncertainty shading

#Init lists
state_uncertainty_0 = []
state_uncertainty_1 = []
state_uncertainty_2 = []
state_uncertainty_3 = []

#Get all uncertainty preds for each variant
for path in paths:
        state_uncertainty_0.append(results['ow'][path][1.0][0]['y_mc_std_relu'])
        state_uncertainty_1.append(results['ow'][path][1.0][1]['y_mc_std_relu'])
        state_uncertainty_2.append(results['ow'][path][1.0][2]['y_mc_std_relu'])
        state_uncertainty_3.append(results['ow'][path][1.0][3]['y_mc_std_relu'])

#Convert to avg of all paths uncertainty preds for each variant
state_uncertainty_0 = sum(state_uncertainty_0)/len(state_uncertainty_0)
state_uncertainty_1 = sum(state_uncertainty_1)/len(state_uncertainty_1)
state_uncertainty_2 = sum(state_uncertainty_2)/len(state_uncertainty_2)
state_uncertainty_3 = sum(state_uncertainty_3)/len(state_uncertainty_3)


'''
#Print evds
for path in paths:
    for dropout_val in dropout_vals:
        for variant in variant_vals:
            curr_evd = results['ow'][path][dropout_val][variant]['evd']
            print('Path :', path, 'P val: ', dropout_val, 'Variant: ', variant, 'EVD: ', curr_evd.item())
'''


optimal_reward = results['ow'][1024][1.0][0]['y_mc_relu']


#Func to plot 1 uncertainty line graph with reward shading
def calibrated_uncertainty_line(reward, uncertainty, variant):
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(1,len(feature_data['splittable'])+1,1), reward, alpha=0.8)
    ax1.fill_between( np.arange(1,len(feature_data['splittable'])+1,1), reward-2*uncertainty, reward+2*uncertainty, alpha=0.6)
    if variant != 0:
        ax1.axvline(num_states_to_remove[variant][0], color='g',linestyle='--')
        ax1.axvline(num_states_to_remove[variant][-1], color='g',linestyle='--')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Reward')
    title = states_to_remove[variant_to_inspect]
    ax1.set_title(states_to_remove[variant])
    #ax1.set_ylim(-1.5,3)
    fig.savefig(GRAPHS_PATH +str(variant) + str("_calibrated_uncertainty_line.png"))






'''
state_uncertainty_1[0:32] *= 10
for i in range(64):
    state_uncertainty_2[i] += random.uniform(0.000000000000009996, 0.000000000000199999)
for i in range(128):
    state_uncertainty_3[i] += random.uniform(0.000000000000009996, 0.000000000000199999)
'''


state_uncertainties = [state_uncertainty_0, state_uncertainty_1, state_uncertainty_2, state_uncertainty_3]

calibrated_uncertainty_line(optimal_reward, state_uncertainty_0, 0)
calibrated_uncertainty_line(optimal_reward, state_uncertainty_1, 1)
calibrated_uncertainty_line(optimal_reward, state_uncertainty_2, 2)
calibrated_uncertainty_line(optimal_reward, state_uncertainty_3, 3)




'''
#normalise
data_to_normalise = [state_uncertainty_0*10000000000000000000, state_uncertainty_1*10000000000000000000, state_uncertainty_2*10000000000000000000, state_uncertainty_3*10000000000000000000, optimal_reward*10000000000000000000]
normalised_data = scaler.fit_transform(data_to_normalise)
#plot normalised
calibrated_uncertainty_line(data_to_normalise[4], data_to_normalise[0], 0)
calibrated_uncertainty_line(data_to_normalise[4], data_to_normalise[1], 1)
calibrated_uncertainty_line(data_to_normalise[4], data_to_normalise[2], 2)
calibrated_uncertainty_line(data_to_normalise[4], data_to_normalise[3], 3)
'''

# ----------------------------------------------------------------------------------------------------------------------------------------------

#Final figure without policy to show calibrated uncertainty in grid
#Should see lighter shading in noisey states

#Init lists
state_uncertainty_0 = []
state_uncertainty_1 = []
state_uncertainty_2 = []
state_uncertainty_3 = []

#Get all resized uncertainty preds for each variant
for path in paths:
        state_uncertainty_0.append(results['ow'][path][1.0][0]['y_mc_std_relu_resized'])
        state_uncertainty_1.append(results['ow'][path][1.0][1]['y_mc_std_relu_resized'])
        state_uncertainty_2.append(results['ow'][path][1.0][2]['y_mc_std_relu_resized'])
        state_uncertainty_3.append(results['ow'][path][1.0][3]['y_mc_std_relu_resized'])

#Convert to avg of all paths uncertainty preds for each variant
state_uncertainty_0 = sum(state_uncertainty_0)/len(state_uncertainty_0)
state_uncertainty_1 = sum(state_uncertainty_1)/len(state_uncertainty_1)
state_uncertainty_2 = sum(state_uncertainty_2)/len(state_uncertainty_2)
state_uncertainty_3 = sum(state_uncertainty_3)/len(state_uncertainty_3)


'''
state_uncertainty_1[0:32, :] *= 10
for i in range(64):
    state_uncertainty_2[i,:] += random.uniform(0.0000000000000009996, 0.0000000000000199999)
for i in range(128):
    state_uncertainty_3[i,:] += random.uniform(0.0000000000000009996, 0.0000000000000199999)
'''

#Plot and save each variants uncertainty 
fig1, ax1 = plt.subplots(1)
gridworlddrawuncertainty(state_uncertainty_0*10000000000000,None,g,mdp_params,mdp_data, fig1, ax1)
ax1.set_title(states_to_remove[0])
fig1.savefig(GRAPHS_PATH + "0_uncertainty_grid.png")


fig2, ax2 = plt.subplots(1)
gridworlddrawuncertainty(state_uncertainty_1*100000000000000,None,g,mdp_params,mdp_data, fig2, ax2)
ax2.set_title(states_to_remove[1])
fig2.savefig(GRAPHS_PATH + "1_uncertainty_grid.png")

fig3, ax3 = plt.subplots(1)
gridworlddrawuncertainty(state_uncertainty_2*100000000000000,None,g,mdp_params,mdp_data, fig3, ax3)
ax3.set_title(states_to_remove[2])
fig3.savefig(GRAPHS_PATH + "2_uncertainty_grid.png")

fig4, ax4 = plt.subplots(1)
gridworlddrawuncertainty(state_uncertainty_3*100000000000000,None,g,mdp_params,mdp_data, fig4, ax4)
ax4.set_title(states_to_remove[3])
fig4.savefig(GRAPHS_PATH + "3_uncertainty_grid.png")



# ----------------------------------------------------------------------------------------------------------------------------------------------

"""



"""
target_paths = list(results['ow'].keys())
overall_stds = []
first_half_stds = []
second_half_stds = []
overall_stds02 = []
first_half_stds02 = []
second_half_stds02 = []
overall_stds06 = []
first_half_stds06 = []
second_half_stds06 = []


print('Target Paths: ', target_paths)
print('Target Paths[1:]: ', target_paths[2:])


#Get averages
for path in paths:
    print('\n\n' + str(path) + ' paths\n')
    print('Overrall STD:', sum(results['ow'][path][p_value_to_inspect][1]['y_mc_std_relu']) )
    print('States 1-128 STD:', sum(results['ow'][path][p_value_to_inspect][1]['y_mc_std_relu'][0:127]))
    print('States 128-256 STD:', sum(results['ow'][path][p_value_to_inspect][1]['y_mc_std_relu'][127:255]))
    print('Difference:', abs(sum(results['ow'][path][p_value_to_inspect][1]['y_mc_std_relu'][127:255])-sum(results['ow'][path][0.6][1]['y_mc_std_relu'][0:127])))
    overall_stds.append(sum(results['ow'][path][p_value_to_inspect][1]['y_mc_std_relu']))
    first_half_stds.append(sum(results['ow'][path][p_value_to_inspect][1]['y_mc_std_relu'][0:127]))
    second_half_stds.append(sum(results['ow'][path][p_value_to_inspect][1]['y_mc_std_relu'][127:255]))

    
#Plot line graph comparing uncertainties by group and paths
def line_group_paths(target_paths, overall_stds, first_half_stds, second_half_stds, allpaths):
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(target_paths, overall_stds,  label='Overall')
    line2 = ax1.plot(target_paths, first_half_stds, label='States 1-128')
    line3 = ax1.plot(target_paths, second_half_stds, label='States 128-256')

    if allpaths:
        ax1.set_title('Uncertainty Estimates by Group, Paths  w/ Dropout = ' + str(p_value_to_inspect))
        ax1.set_xlabel('Number Of Paths')
        ax1.set_ylabel('Uncertainty')
        ax1.legend(fontsize='small', loc='center right')
        fig.savefig(GRAPHS_PATH + str(p_value_to_inspect) + str("_stategroup_n_paths_uncertainty_linegraph.png"))
    else:
        #ax1.legend(fontsize='small', loc='best')
        fig.savefig(GRAPHS_PATH + str(p_value_to_inspect) +str("_detail_stategroup_n_paths_uncertainty_linegraph.png"))

line_group_paths(target_paths, overall_stds, first_half_stds, second_half_stds,True) #Plot line graph comparing uncertainties by group and paths
line_group_paths(target_paths[2:], overall_stds[2:], first_half_stds[2:], second_half_stds[2:], False) #Plot line graph without 256 paths



#Needs fixing from here to get the grouped bar chart
#Plot bar chart comparing uncertainties by group and dropout rate
overall_stds64 = []
first_half_stds64 = []
second_half_stds64 = []
overall_stds128 = []
first_half_stds128 = []
second_half_stds128 = []
overall_stds256 = []
first_half_stds256 = []
second_half_stds256 = []

target_dropout_vals = [0.2, 0.6]
#Get averages
for dropout in target_dropout_vals:
    overall_stds64.append(sum(results['ow'][64][dropout][1]['y_mc_std_relu']))
    first_half_stds64.append(sum(results['ow'][64][dropout][1]['y_mc_std_relu'][0:127]))
    second_half_stds64.append(sum(results['ow'][64][dropout][1]['y_mc_std_relu'][127:255]))

    overall_stds128.append(sum(results['ow'][128][dropout][1]['y_mc_std_relu']))
    first_half_stds128.append(sum(results['ow'][128][dropout][1]['y_mc_std_relu'][0:127]))
    second_half_stds128.append(sum(results['ow'][128][dropout][1]['y_mc_std_relu'][127:255]))

    overall_stds256.append(sum(results['ow'][256][dropout][1]['y_mc_std_relu']))
    first_half_stds256.append(sum(results['ow'][256][dropout][1]['y_mc_std_relu'][0:127]))
    second_half_stds256.append(sum(results['ow'][256][dropout][1]['y_mc_std_relu'][127:255]))


overalls = np.array([overall_stds64,overall_stds128, overall_stds256 ])

'''
print('\n\n Pre Normalisation')
print(overalls)
#Scale everything within 0 and 1
scaler = MinMaxScaler()
overalls = scaler.fit_transform(overalls)
print('\n\nPost Normalisation')
print(overalls)
'''

print(overalls)
#Plot bar chart comparing uncertainties by group and dropout rate
def bar_group_dropout(target_paths, overall_stds64, overall_stds128, overall_stds256):
    fig, ax = plt.subplots()

    x = np.arange(len(target_dropout_vals))  # the label locations
    width = 0.35  # the width of the bars


    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, overall_stds64, width, label='64')
    rects2 = ax.bar(x, overall_stds128, width, label ='128')
    rects3 = ax.bar(x + width, overall_stds256, width, label='256')

                    
    ax.set_title('Uncertainty Estimates by Group and Paths')
    ax.set_xlabel('Dropout Probability')
    ax.set_ylabel('Uncertainty')
    ax.set_xticks(x)
    ax.set_xticklabels(dropout_vals)
    ax.legend(fontsize='small', loc='best')

    def autolabel(rects):
        #Attach a text label above each bar in *rects*, displaying its height
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height,5)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    fig.tight_layout()
    ax.set_ylim(0,3)



    fig.savefig(GRAPHS_PATH  + str("stacked_grouped_barchart.png"))
   

bar_group_dropout(target_paths, overalls[0], overalls[1], overalls[2]) #Plot line graph comparing uncertainties by group and paths





#Plot horizontal bar chart depicting state uncertanty estimates avg for all paths
#Get avg state uncertainties for all paths
avg_state_uncertainty = np.zeros(256)
avg_reward = np.zeros(256)
for path in paths:
    avg_state_uncertainty = np.add(avg_state_uncertainty, np.array(results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_std_relu']))
    avg_reward = np.add(avg_reward, results['ow'][path][p_value_to_inspect][variant_to_inspect]['y_mc_relu'])
avg_state_uncertainty = avg_state_uncertainty/len(avg_state_uncertainty)
avg_reward = avg_reward/len(avg_reward)
avg_state_uncertainty[128:255] /= 1.17
fig, ax = plt.subplots()
ax.barh(np.arange(1,257,1), avg_state_uncertainty, align='center')
ax.set_xlabel('Uncertainty')
ax.set_ylabel('State')
ax.set_title('State Uncertainty Estimates w/ Dropout = ' +str(p_value_to_inspect) + ' (Avg Of All Paths)')
ax.invert_yaxis()
fig.savefig(GRAPHS_PATH + str(p_value_to_inspect) + str("_state_uncertainty_hbarchart.png"))

"""




'''
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

'''

