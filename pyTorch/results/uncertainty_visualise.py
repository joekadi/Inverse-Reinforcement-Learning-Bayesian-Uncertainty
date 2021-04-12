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
sys.path[0] = "/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/pyTorch/"
from maxent_irl.obj_functions.NLLFunction import *
from maxent_irl.obj_functions.NLLModel import *
from benchmarks.gridworld import *
from benchmarks.objectworld import *
from maxent_irl.linearvalueiteration import *
import pprint
from maxent_irl.sampleexamples import *
import pprint
import numpy as np
import numpy
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
from utils import epdue, plot_epdue, ence, plot_ence, se, calibration_curve
from scipy.stats import variation, ttest_ind
from tabulate import tabulate
import csv

tensorboard_writer = SummaryWriter('./tensorboard_logs')
torch.set_printoptions(precision=5, threshold=1000000000)
torch.set_default_tensor_type(torch.DoubleTensor)

num_states_to_remove = [np.arange(0, 32, 1), np.arange(0, 64, 1), np.arange(0, 128, 1)]
states_to_remove = ['States 1-32 Removed From Paths', 'States 1-64 Removed From Paths', 'States 1-128 Removed From Paths']

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


torch.manual_seed(mdp_params['seed'])
np.random.seed(seed=mdp_params['seed'])
random.seed(mdp_params['seed'])

# Compute visible examples
Eo = torch.zeros(int(mdp_data['states']),1)
for i in range(len(example_samples[0])):
    for t in range(len(example_samples[0][0])):
        Eo[[example_samples][0][i][t][0]] = 1
g = torch.ones(int(mdp_data['states']),1)*0.5+Eo*0.5

#Get directory for results

#Get directory for noisey paths results
NP_RESULTS_PATH = "./noisey_paths/results/dropout/"
NP_SWAG_RESULTS_PATH = "./noisey_paths/results/swag/"
NP_GP_RESULTS_PATH = "./noisey_paths/results/gpirl/"
NP_ENSEMBLE_RESULTS_PATH = "./noisey_paths/results/ensembles/"


#Get directory for noisey features results
NF_RESULTS_PATH = "./noisey_features/results/dropout/"
NF_ENSEMBLE_RESULTS_PATH = "./noisey_features/results/ensembles/"
NF_SWAG_RESULTS_PATH = "./noisey_features/results/swag/"
NF_GP_RESULTS_PATH = "./noisey_features/results/gpirl/"


#Get directory for total results
TOTAL_ENSEMBLE_RESULTS_PATH = "./total_uncertainty/results/ensembles/"
TOTAL_GP_RESULTS_PATH = "./total_uncertainty/results/gpirl/"
TOTAL_SWAG_RESULTS_PATH = "./total_uncertainty/results/swag/"
TOTAL_DROPOUT_RESULTS_PATH = "./total_uncertainty/results/dropout/"

#Create directory for graphs 
NP_GRAPHS_PATH = "./noisey_paths/results/graphs/"
NF_GRAPHS_PATH = "./noisey_features/results/graphs/"
TOTAL_GRAPHS_PATH = "./total_uncertainty/graphs/"

#Get directory for regular results
DROPOUT_RESULTS_PATH = "./regular/results/dropout/"
SWAG_RESULTS_PATH = "./regular/results/swag/"
GPIRL_RESULTS_PATH = "./regular/results/gpirl/"

#Make repo's to save figures
for path in [NP_GRAPHS_PATH]:
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

for path in [NF_GRAPHS_PATH]:
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

for path in [TOTAL_GRAPHS_PATH]:
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

np_results = {
        'ow': {

                64: {0.0: {0: None, 1: None, 2: None}, 0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}, 
                128: {0.0: {0: None, 1: None, 2: None}, 0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}, 
                256: {0.0: {0: None, 1: None, 2: None}, 0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}, 
                512: {0.0: {0: None, 1: None, 2: None}, 0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None}, 0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}
            }
            }

np_swag_results = {
        'ow': {

                #64: {0.0: {0: None, 1: None, 2: None}}, 
                128: {0.0: {0: None, 1: None, 2: None}}, 
                256: {0.0: {0: None, 1: None, 2: None}}, 
                512: {0.0: {0: None, 1: None, 2: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None}}
            }
            }

np_gp_results = {
        'ow': {

                64: {0.0: {0: None, 1: None, 2: None}}, 
                128: {0.0: {0: None, 1: None, 2: None}}, 
                256: {0.0: {0: None, 1: None, 2: None}}, 
                512: {0.0: {0: None, 1: None, 2: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None}}
            }
            }

np_ensemble_results = {
        'ow': {

                64: {0.0: {0: None, 1: None, 2: None}}, 
                128: {0.0: {0: None, 1: None, 2: None}}, 
                256: {0.0: {0: None, 1: None, 2: None}}, 
                512: {0.0: {0: None, 1: None, 2: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None}}
            }
            }

nf_results = {
        'ow': {

                64: {0.0: {0: None, 1: None, 2: None}, 0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}, 
                128: {0.0: {0: None, 1: None, 2: None}, 0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}, 
                256: {0.0: {0: None, 1: None, 2: None}, 0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}, 
                512: {0.0: {0: None, 1: None, 2: None}, 0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None}, 0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}
            }
            }

nf_swag_results = {
        'ow': {

                #64: {0.0: {0: None, 1: None, 2: None}}, 
                128: {0.0: {0: None, 1: None, 2: None}}, 
                256: {0.0: {0: None, 1: None, 2: None}}, 
                512: {0.0: {0: None, 1: None, 2: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None}}
            }
            }

nf_gp_results = {
        'ow': {

                64: {0.0: {0: None, 1: None, 2: None}}, 
                128: {0.0: {0: None, 1: None, 2: None}}, 
                256: {0.0: {0: None, 1: None, 2: None}}, 
                512: {0.0: {0: None, 1: None, 2: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None}}
            }
            }

nf_ensemble_results = {
        'ow': {

                64: {0.0: {0: None, 1: None, 2: None}}, 
                128: {0.0: {0: None, 1: None, 2: None}}, 
                256: {0.0: {0: None, 1: None, 2: None}}, 
                512: {0.0: {0: None, 1: None, 2: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None}}
            }
            }

total_dropout_results = {
        'ow': {

                64: {0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}, 
                128: {0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}, 
                256: {0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}, 
                512: {0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}, 
                1024: {0.5: {0: None, 1: None, 2: None},  1.0: {0: None, 1: None, 2: None}}
            }
            }

total_ensemble_results = {
        'ow': {

                64: {0.0: {0: None, 1: None, 2: None}}, 
                128: {0.0: {0: None, 1: None, 2: None}}, 
                256: {0.0: {0: None, 1: None, 2: None}}, 
                512: {0.0: {0: None, 1: None, 2: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None}}
            }
            }

total_gp_results = {
        'ow': {

                64: {0.0: {0: None, 1: None, 2: None}}, 
                128: {0.0: {0: None, 1: None, 2: None}}, 
                256: {0.0: {0: None, 1: None, 2: None}}, 
                512: {0.0: {0: None, 1: None, 2: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None}}
            }
            }      

total_swag_results = {
        'ow': {

                64: {0.0: {0: None, 1: None, 2: None}}, 
                128: {0.0: {0: None, 1: None, 2: None}}, 
                256: {0.0: {0: None, 1: None, 2: None}}, 
                512: {0.0: {0: None, 1: None, 2: None}}, 
                1024: {0.0: {0: None, 1: None, 2: None}}
            }
            }  

#Dicts for regular results

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

    #stds = scaler.fit_transform(np.array(stds))

#Get lists of variants
worlds = ['ow']
paths = list(np_results['ow'].keys())
dropout_vals = [0.0, 0.5, 1.0]
variant_vals = list(np_results['ow'][paths[0]][dropout_vals[0]].keys())
scaler = MinMaxScaler()
#Populate NP results dict with results from files
for world in worlds:
    for no_paths in paths:
        for dropout_val in dropout_vals:
            for variant in variant_vals:
                file_name = NP_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
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
                
                np_results[str(world)][no_paths][dropout_val][variant] = result_values

#Populate NP SWAG results dict with results from files
for world in worlds:
    for no_paths in [128,256,512,1024]:
        for dropout_val in [0.0]:
            for variant in variant_vals:
                file_name = NP_SWAG_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
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
                
                np_swag_results[str(world)][no_paths][dropout_val][variant] = result_values

#Populate NP GPIRL results dict with results from files
for world in worlds:
    for no_paths in paths:
        for dropout_val in [0.0]:
            for variant in variant_vals:
                file_name = NP_GP_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
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
                
                np_gp_results[str(world)][no_paths][dropout_val][variant] = result_values

#Populate NF results dict with results from files
for world in worlds:
    for no_paths in paths:
        for dropout_val in dropout_vals:
            for variant in variant_vals:
                file_name = NF_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
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
                
                nf_results[str(world)][no_paths][dropout_val][variant] = result_values

#Populate NF SWAG results dict with results from files
for world in worlds:
    for no_paths in [128,256,512,1024]:
        for dropout_val in [0.0]:
            for variant in variant_vals:
                file_name = NF_SWAG_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
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
                
                nf_swag_results[str(world)][no_paths][dropout_val][variant] = result_values

#Populate NF GPIRL results dict with results from files
for world in worlds:
    for no_paths in paths:
        for dropout_val in [0.0]:
            for variant in variant_vals:
                file_name = NF_GP_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
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
                
                nf_gp_results[str(world)][no_paths][dropout_val][variant] = result_values

#Populate NF ensemble results dict with results from files
for world in worlds:
    for no_paths in paths:
        for dropout_val in [0.0]:
            for variant in variant_vals:
                file_name = NF_ENSEMBLE_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
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
                
                nf_ensemble_results[str(world)][no_paths][dropout_val][variant] = result_values

#Populate NP ensemble results dict with results from files
for world in worlds:
    for no_paths in paths:
        for dropout_val in [0.0]:
            for variant in variant_vals:
                file_name = NP_ENSEMBLE_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
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
                
                np_ensemble_results[str(world)][no_paths][dropout_val][variant] = result_values

#Populate total dropout results dict with results from files
for world in worlds:
    for no_paths in paths:
        for dropout_val in [0.5, 1.0]:
            for variant in variant_vals:
                file_name = TOTAL_DROPOUT_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
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
                
                total_dropout_results[str(world)][no_paths][dropout_val][variant] = result_values

#Populate total ensemble results dict with results from files
for world in worlds:
    for no_paths in [128,256,512,1024]:
        for dropout_val in [0.0]:
            for variant in variant_vals:
                file_name = TOTAL_ENSEMBLE_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_resultsproper_'+str(variant)+'.pkl'
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
                
                total_ensemble_results[str(world)][no_paths][dropout_val][variant] = result_values

#Populate total swag results dict with results from files
for world in worlds:
    for no_paths in [128,256,512,1024]:
        for dropout_val in [0.0]:
            for variant in variant_vals:
                file_name = TOTAL_SWAG_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
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
                
                total_swag_results[str(world)][no_paths][dropout_val][variant] = result_values

#Populate total gp results dict with results from files
for world in worlds:
    for no_paths in [128,256,512,1024]:
        for dropout_val in [0.0]:
            for variant in variant_vals:
                file_name = TOTAL_GP_RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
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
                
                total_gp_results[str(world)][no_paths][dropout_val][variant] = result_values

#Populate reg results dict with results from file
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
                                'y_mc_std_relu_resized': y_mc_std_relu_resized}
            
            results[str(world)][no_paths][dropout_val]= result_values

#Populate reg swag results dict with results from file
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
                                'y_mc_std_relu_resized': y_mc_std_relu_resized}
            
            swag_results[str(world)][no_paths][dropout_val] = result_values

#Populate reg GPIRL results dict with results from file
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
# ---------------------------------------------------------------------------------------------------------------------------------------------------

#Calculate and store avg uncertainties and policies for each method & variant

paths = [128,256,512,1024]


np_dropout_uncertainties_per_paths = []
np_dropout_policies_per_paths = []

np_ensembles_uncertainties_per_paths = []
np_ensembles_policies_per_paths = []


for path in paths:
    np_dropout_agg_policy = torch.zeros(256,5)
    np_dropout_agg_uncertainty = torch.zeros(256,1)
    np_dropout_epdue = 0.0

    np_ensemble_agg_policy = torch.zeros(256,5)
    np_ensemble_agg_uncertainty = torch.zeros(256,1)
    np_ensemble_epdue = 0.0

    for variant in variant_vals:
  
        
        # Use avg results from 1.0 and 0.5  dropout p values as dropout result
        np_dropout_policy = (np_results['ow'][path][0.5][variant]['y_mc_relu_P']+np_results['ow'][path][1.0][variant]['y_mc_relu_P'])/2#+np_results['ow'][path][0.0][variant]['y_mc_relu_P'])/3
        np_dropout_uncertainty = torch.tensor((np_results['ow'][path][0.5][variant]['y_mc_std_relu']+np_results['ow'][path][1.0][variant]['y_mc_std_relu'])/2)#+np_results['ow'][path][0.0][variant]['y_mc_std_relu'])/3)    
        
        np_ensemble_policy = np_ensemble_results['ow'][path][0.0][variant]['y_mc_relu_P']
        np_ensemble_uncertainty = torch.tensor(np_ensemble_results['ow'][path][0.0][variant]['y_mc_std_relu'])   


        np_dropout_uncertainty = np_dropout_uncertainty.reshape(len(np_dropout_uncertainty), 1)
        np_ensemble_uncertainty = np_ensemble_uncertainty.reshape(len(np_ensemble_uncertainty), 1)
        np_dropout_agg_policy += np_dropout_policy
        np_dropout_agg_uncertainty += np_dropout_uncertainty
        np_ensemble_agg_policy += np_ensemble_policy
        np_ensemble_agg_uncertainty += np_ensemble_uncertainty

    np_dropout_uncertainties_per_paths.append(np_dropout_agg_uncertainty/len(variant_vals))
    np_dropout_policies_per_paths.append(np_dropout_agg_policy/len(variant_vals))

    np_ensembles_uncertainties_per_paths.append(np_ensemble_agg_uncertainty/len(variant_vals))
    np_ensembles_policies_per_paths.append(np_ensemble_agg_policy/len(variant_vals))

#Calc avg policies
np_dropout_avg_policy = sum(np_dropout_policies_per_paths)/len(np_dropout_policies_per_paths)
np_ensembles_avg_policy = sum(np_ensembles_policies_per_paths)/len(np_ensembles_policies_per_paths)

#Calc avg uncertainty
np_dropout_avg_uncertainty = scaler.fit_transform(sum(np_dropout_uncertainties_per_paths)/len(np_dropout_uncertainties_per_paths))
np_ensembles_avg_uncertainty = scaler.fit_transform(sum(np_ensembles_uncertainties_per_paths)/len(np_ensembles_uncertainties_per_paths))


#Calculate and store avg swag and gp uncertainties and policies
np_swag_uncertainties_per_paths = []
np_swag_policies_per_paths = []

np_gp_uncertainties_per_paths = []
np_gp_policies_per_paths = []

for path in paths:
    np_swag_agg_policy = torch.zeros(256,5)
    np_swag_agg_uncertainty = torch.zeros(256,1)

    np_gp_agg_policy = torch.zeros(256,5)
    np_gp_agg_uncertainty = torch.zeros(256,1)

    np_swag_epdue = 0.0
    np_gp_epdue = 0.0

    for variant in variant_vals:
        np_swag_policy = np_swag_results['ow'][path][0.0][variant]['y_mc_relu_P']
        np_swag_uncertainty = torch.tensor(np_swag_results['ow'][path][0.0][variant]['y_mc_std_relu'])    
        np_swag_uncertainty = np_swag_uncertainty.reshape(len(np_swag_uncertainty), 1)
        np_swag_agg_policy += np_swag_policy
        np_swag_agg_uncertainty += np_swag_uncertainty

        np_gp_policy = np_gp_results['ow'][path][0.0][variant]['y_mc_relu_P']
        np_gp_uncertainty = torch.tensor(np_gp_results['ow'][path][0.0][variant]['y_mc_std_relu'])    
        np_gp_uncertainty = np_gp_uncertainty.reshape(len(np_gp_uncertainty), 1)
        np_gp_agg_policy += np_gp_policy
        np_gp_agg_uncertainty += np_gp_uncertainty


    np_swag_uncertainties_per_paths.append(np_swag_agg_uncertainty/len(variant_vals))
    np_swag_policies_per_paths.append(np_swag_agg_policy/len(variant_vals))


    np_gp_uncertainties_per_paths.append(np_gp_agg_uncertainty/len(variant_vals))
    np_gp_policies_per_paths.append(np_gp_agg_policy/len(variant_vals))

#Calc avg policies
np_swag_avg_policy = sum(np_swag_policies_per_paths)/len(np_swag_policies_per_paths)
np_gp_avg_policy = sum(np_gp_policies_per_paths)/len(np_gp_policies_per_paths)

#Calc avg uncertainty
np_swag_avg_uncertainty = scaler.fit_transform(sum(np_swag_uncertainties_per_paths)/len(np_swag_uncertainties_per_paths))
np_gp_avg_uncertainty = scaler.fit_transform(sum(np_gp_uncertainties_per_paths)/len(np_gp_uncertainties_per_paths))


nf_dropout_uncertainties_per_paths = []
nf_dropout_policies_per_paths = []

nf_ensembles_uncertainties_per_paths = []
nf_ensembles_policies_per_paths = []

for path in paths:
    nf_dropout_agg_policy = torch.zeros(256,5)
    nf_dropout_agg_uncertainty = torch.zeros(256,1)
    nf_dropout_epdue = 0.0

    nf_ensemble_agg_policy = torch.zeros(256,5)
    nf_ensemble_agg_uncertainty = torch.zeros(256,1)
    nf_ensemble_epdue = 0.0

    for variant in variant_vals:
        #nf_dropout_policy = nf_results['ow'][path][1.0][variant]['y_mc_relu_P']
        #nf_dropout_uncertainty = torch.tensor(nf_results['ow'][path][1.0][variant]['y_mc_std_relu'])    
        
        # Use avg results from 1.0 and 0.5 and 0.0 dropout p values as dropout result
        nf_dropout_policy = (nf_results['ow'][path][1.0][variant]['y_mc_relu_P'])#+nf_results['ow'][path][0.5][variant]['y_mc_relu_P'])/2#+nf_results['ow'][path][0.0][variant]['y_mc_relu_P'])/3
        nf_dropout_uncertainty = torch.tensor((nf_results['ow'][path][1.0][variant]['y_mc_std_relu']))#+nf_results['ow'][path][0.5][variant]['y_mc_std_relu'])/2)#+nf_results['ow'][path][0.0][variant]['y_mc_std_relu'])/3)    
        
        nf_ensemble_policy = nf_ensemble_results['ow'][path][0.0][variant]['y_mc_relu_P']
        nf_ensemble_uncertainty = torch.tensor(nf_ensemble_results['ow'][path][0.0][variant]['y_mc_std_relu'])  


        nf_dropout_uncertainty = nf_dropout_uncertainty.reshape(len(nf_dropout_uncertainty), 1)
        nf_ensemble_uncertainty = nf_ensemble_uncertainty.reshape(len(nf_ensemble_uncertainty), 1)
        nf_dropout_agg_policy += nf_dropout_policy
        nf_dropout_agg_uncertainty += nf_dropout_uncertainty
        nf_ensemble_agg_policy += nf_ensemble_policy
        nf_ensemble_agg_uncertainty += nf_ensemble_uncertainty

    nf_dropout_uncertainties_per_paths.append(nf_dropout_agg_uncertainty/len(variant_vals))
    nf_dropout_policies_per_paths.append(nf_dropout_agg_policy/len(variant_vals))

    nf_ensembles_uncertainties_per_paths.append(nf_ensemble_agg_uncertainty/len(variant_vals))
    nf_ensembles_policies_per_paths.append(nf_ensemble_agg_policy/4)

#Calc avg policies
nf_dropout_avg_policy = sum(nf_dropout_policies_per_paths)/len(nf_dropout_policies_per_paths)
nf_ensembles_avg_policy = sum(nf_ensembles_policies_per_paths)/len(nf_ensembles_policies_per_paths)

#Calc avg uncertainty
nf_dropout_avg_uncertainty = scaler.fit_transform(sum(nf_dropout_uncertainties_per_paths)/len(nf_dropout_uncertainties_per_paths))
nf_ensembles_avg_uncertainty = scaler.fit_transform(sum(nf_ensembles_uncertainties_per_paths)/len(nf_ensembles_uncertainties_per_paths))

#Calculate and store avg swag uncertainties and policies
nf_swag_uncertainties_per_paths = []
nf_swag_policies_per_paths = []
nf_swag_rewards_per_paths = []

nf_gp_uncertainties_per_paths = []
nf_gp_policies_per_paths = []


for path in paths:
    nf_swag_agg_policy = torch.zeros(256,5)
    nf_swag_agg_uncertainty = torch.zeros(256,1)
    nf_swag_agg_reward = torch.zeros(256,1)
    nf_swag_epdue = 0.0

    nf_gp_agg_policy = torch.zeros(256,5)
    nf_gp_agg_uncertainty = torch.zeros(256,1)
    nf_gp_epdue = 0.0


    for variant in variant_vals:
        nf_swag_policy = nf_swag_results['ow'][path][0.0][variant]['y_mc_relu_P']
        nf_swag_uncertainty = torch.tensor(nf_swag_results['ow'][path][0.0][variant]['y_mc_std_relu'])    
        nf_swag_reward = nf_swag_results['ow'][path][0.0][variant]['y_mc_relu']    

        nf_swag_uncertainty = nf_swag_uncertainty.reshape(len(nf_swag_uncertainty), 1)
        nf_swag_agg_policy += nf_swag_policy
        nf_swag_agg_uncertainty += nf_swag_uncertainty

        nf_swag_reward = nf_swag_reward.reshape(len(nf_swag_reward), 1)


        nf_swag_agg_reward += nf_swag_reward

        nf_gp_policy = nf_gp_results['ow'][path][0.0][variant]['y_mc_relu_P']
        nf_gp_uncertainty = torch.tensor(nf_gp_results['ow'][path][0.0][variant]['y_mc_std_relu'])    
        nf_gp_uncertainty = nf_gp_uncertainty.reshape(len(nf_gp_uncertainty), 1)
        nf_gp_agg_policy += nf_gp_policy
        nf_gp_agg_uncertainty += nf_gp_uncertainty


    nf_swag_uncertainties_per_paths.append(nf_swag_agg_uncertainty/4)
    nf_swag_policies_per_paths.append(nf_swag_agg_policy/4)
    nf_swag_rewards_per_paths.append(nf_swag_agg_reward/4)

    nf_gp_uncertainties_per_paths.append(nf_gp_agg_uncertainty/len(variant_vals))
    nf_gp_policies_per_paths.append(nf_gp_agg_policy/len(variant_vals))


#Calc avg policies
nf_swag_avg_policy = sum(nf_swag_policies_per_paths)/len(nf_swag_policies_per_paths)
nf_gp_avg_policy = sum(nf_gp_policies_per_paths)/len(nf_gp_policies_per_paths)

#Calc avg uncertainty
nf_swag_avg_uncertainty = scaler.fit_transform(sum(nf_swag_uncertainties_per_paths)/len(nf_swag_uncertainties_per_paths))
nf_gp_avg_uncertainty = scaler.fit_transform(sum(nf_gp_uncertainties_per_paths)/len(nf_gp_uncertainties_per_paths))

#Calc avg reward
nf_swag_avg_reward = sum(nf_swag_rewards_per_paths)/len(nf_swag_rewards_per_paths)

#Calculate and store avg total uncertainties and policies

total_swag_uncertainties_per_paths = []
total_swag_policies_per_paths = []

total_gp_uncertainties_per_paths = []
total_gp_policies_per_paths = []

total_ensembles_uncertainties_per_paths = []
total_ensembles_policies_per_paths = []

total_dropout_uncertainties_per_paths = []
total_dropout_policies_per_paths = []

for path in paths:
    total_swag_agg_policy = torch.zeros(256,5)
    total_swag_agg_uncertainty = torch.zeros(256,1)

    total_gp_agg_policy = torch.zeros(256,5)
    total_gp_agg_uncertainty = torch.zeros(256,1)

    total_ensembles_agg_policy = torch.zeros(256,5)
    total_ensembles_agg_uncertainty = torch.zeros(256,1)

    total_dropout_agg_policy = torch.zeros(256,5)
    total_dropout_agg_uncertainty = torch.zeros(256,1)

    total_swag_epdue = 0.0
    total_gp_epdue = 0.0
    total_ensembles_epdue = 0.0

    for variant in variant_vals:
        total_swag_policy = np_swag_results['ow'][path][0.0][variant]['y_mc_relu_P']
        total_swag_uncertainty = torch.tensor(total_swag_results['ow'][path][0.0][variant]['y_mc_std_relu'])    
        total_swag_uncertainty = total_swag_uncertainty.reshape(len(total_swag_uncertainty), 1)
        total_swag_agg_policy += total_swag_policy
        total_swag_agg_uncertainty += total_swag_uncertainty

        total_gp_policy = total_gp_results['ow'][path][0.0][variant]['y_mc_relu_P']
        total_gp_uncertainty = torch.tensor(total_gp_results['ow'][path][0.0][variant]['y_mc_std_relu'])    
        total_gp_uncertainty = total_gp_uncertainty.reshape(len(total_gp_uncertainty), 1)
        total_gp_agg_policy += total_gp_policy
        total_gp_agg_uncertainty += total_gp_uncertainty


        total_ensembles_policy = total_ensemble_results['ow'][path][0.0][variant]['y_mc_relu_P']
        total_ensembles_uncertainty = torch.tensor(total_ensemble_results['ow'][path][0.0][variant]['y_mc_std_relu'])    
        total_ensembles_uncertainty = total_ensembles_uncertainty.reshape(len(total_ensembles_uncertainty), 1)
        total_ensembles_agg_policy += total_ensembles_policy
        total_ensembles_agg_uncertainty += total_ensembles_uncertainty

        # Use avg results from 1.0 and 0.5  dropout p values as dropout result
        total_dropout_policy = (total_dropout_results['ow'][path][0.5][variant]['y_mc_relu_P']+total_dropout_results['ow'][path][1.0][variant]['y_mc_relu_P'])/2#+np_results['ow'][path][0.0][variant]['y_mc_relu_P'])/3
        total_dropout_uncertainty = torch.tensor((total_dropout_results['ow'][path][0.5][variant]['y_mc_std_relu']+total_dropout_results['ow'][path][1.0][variant]['y_mc_std_relu'])/2)#+np_results['ow'][path][0.0][variant]['y_mc_std_relu'])/3) 
        total_dropout_uncertainty = total_dropout_uncertainty.reshape(len(total_dropout_uncertainty), 1)
        total_dropout_agg_policy += total_dropout_policy
        total_dropout_agg_uncertainty += total_dropout_uncertainty


    total_swag_uncertainties_per_paths.append(total_swag_agg_uncertainty/len(variant_vals))
    total_swag_policies_per_paths.append(total_swag_agg_policy/len(variant_vals))

    total_gp_uncertainties_per_paths.append(total_gp_agg_uncertainty/len(variant_vals))
    total_gp_policies_per_paths.append(total_gp_agg_policy/len(variant_vals))

    total_ensembles_uncertainties_per_paths.append(total_ensembles_agg_uncertainty/len(variant_vals))
    total_ensembles_policies_per_paths.append(total_ensembles_agg_policy/len(variant_vals))

    total_dropout_uncertainties_per_paths.append(total_dropout_agg_uncertainty/len(variant_vals))
    total_dropout_policies_per_paths.append(total_dropout_agg_policy/len(variant_vals))

#Calc total avg policies
total_swag_avg_policy = sum(total_swag_policies_per_paths)/len(total_swag_policies_per_paths)
total_gp_avg_policy = sum(total_gp_policies_per_paths)/len(total_gp_policies_per_paths)
total_ensembles_avg_policy = sum(total_ensembles_policies_per_paths)/len(total_ensembles_policies_per_paths)
total_dropout_avg_policy = sum(total_dropout_policies_per_paths)/len(total_dropout_policies_per_paths)
#Calc total avg uncertainty
total_swag_avg_uncertainty = scaler.fit_transform(sum(total_swag_uncertainties_per_paths)/len(total_swag_uncertainties_per_paths))
total_gp_avg_uncertainty = scaler.fit_transform(sum(total_gp_uncertainties_per_paths)/len(total_gp_uncertainties_per_paths))
total_ensembles_avg_uncertainty = sum(total_ensembles_uncertainties_per_paths)/len(total_ensembles_uncertainties_per_paths)
total_ensembles_avg_uncertainty = scaler.fit_transform(total_ensembles_avg_uncertainty)
total_dropout_avg_uncertainty = scaler.fit_transform(sum(total_dropout_uncertainties_per_paths)/len(total_dropout_uncertainties_per_paths))


 
# ----------------------------------------------------------------------------------------------------------------------------------------------
# NP: Plot Expected Normalized Calibration Error (ENCE) line graphs

#Calculate ences from avg policies and avg uncertainties

np_dropout_ence = ence(mdp_solution['p'].squeeze(), np_dropout_avg_policy, np_dropout_avg_uncertainty)
np_swag_ence = ence(mdp_solution['p'].squeeze(), np_swag_avg_policy, np_swag_avg_uncertainty)
np_gp_ence = ence(mdp_solution['p'].squeeze(), np_gp_avg_policy, np_gp_avg_uncertainty) 




#Plot dropout ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), np_dropout_avg_policy, np_dropout_avg_uncertainty)
textstr = 'ENCE = ' + str(round(np_dropout_ence,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.92, 0.14,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
  
fig.tight_layout()
fig.savefig(NP_GRAPHS_PATH + "mcdropout_ence.png")


#Plot swag ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), np_swag_avg_policy, np_swag_avg_uncertainty)
textstr = 'ENCE = ' + str(round(np_swag_ence,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.62, 0.94,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
 
fig.tight_layout()
fig.savefig(NP_GRAPHS_PATH + "swag_ence.png")


#Plot gp ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), np_gp_avg_policy, np_gp_avg_uncertainty)
textstr = 'ENCE = ' + str(round(np_gp_ence,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.92, 0.14,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
#ax.set_title('SWAG')   
#ax.legend(loc='lower right', prop={'size': 6})    
fig.tight_layout()
fig.savefig(NP_GRAPHS_PATH + "gp_ence.png")




# ----------------------------------------------------------------------------------------------------------------------------------------------
# NF: Plot Expected Normalized Calibration Error (ENCE) line graphs


#Calculate ences from avg policies and avg uncertainties
nf_ensemble_ence = ence(mdp_solution['p'].squeeze(), nf_ensembles_avg_policy, nf_ensembles_avg_uncertainty)
nf_dropout_ence = ence(mdp_solution['p'].squeeze(), nf_dropout_avg_policy, nf_dropout_avg_uncertainty)
nf_swag_ence = ence(mdp_solution['p'].squeeze(), nf_swag_avg_policy, nf_swag_avg_uncertainty)
nf_gp_ence = ence(mdp_solution['p'].squeeze(), np_gp_avg_policy, np_gp_avg_uncertainty)




#Plot ensembles ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), nf_ensembles_avg_policy, nf_ensembles_avg_uncertainty)
textstr = 'ENCE = ' + str(round(nf_ensemble_ence,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.92, 0.14,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
#ax.set_title('Ensemble')      
#ax.legend(loc='lower right', prop={'size': 6}) 
fig.tight_layout()
fig.savefig(NF_GRAPHS_PATH + "ensemble_ence.png")


#Plot dropout ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), nf_dropout_avg_policy, nf_dropout_avg_uncertainty)
textstr = 'ENCE = ' + str(round(nf_dropout_ence,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.92, 0.14,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
#ax.set_title('MC Dropout')   
#ax.legend(loc='lower right', prop={'size': 6})    
fig.tight_layout()
fig.savefig(NF_GRAPHS_PATH + "mcdropout_ence.png")




#Plot swag ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), nf_swag_avg_policy, nf_swag_avg_uncertainty)

textstr = 'ENCE = ' + str(round(nf_swag_ence,4))

props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.62, 0.94,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
#ax.set_title('SWAG')   
#ax.legend(loc='lower right', prop={'size': 6})    
fig.tight_layout()
fig.savefig(NF_GRAPHS_PATH + "swag_ence.png")


#Plot gp ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), nf_gp_avg_policy, nf_gp_avg_uncertainty)

textstr = 'ENCE = ' + str(round(nf_gp_ence,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.92, 0.14,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
#ax.set_title('SWAG')   
#ax.legend(loc='lower right', prop={'size': 6})    
fig.tight_layout()
fig.savefig(NF_GRAPHS_PATH + "gp_ence.png")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# TOTAL: Plot Expected Normalized Calibration Error (ENCE) line graphs


#Calculate ences from avg policies and avg uncertainties
total_ensemble_ence = ence(mdp_solution['p'].squeeze(), total_ensembles_avg_policy, total_ensembles_avg_uncertainty)
total_dropout_ence = ence(mdp_solution['p'].squeeze(), total_dropout_avg_policy, total_dropout_avg_uncertainty)
total_swag_ence = ence(mdp_solution['p'].squeeze(), total_swag_avg_policy, total_swag_avg_uncertainty)
total_gp_ence = ence(mdp_solution['p'].squeeze(), total_gp_avg_policy, total_gp_avg_uncertainty)


#Plot ensembles ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), total_ensembles_avg_policy, total_ensembles_avg_uncertainty)
textstr = 'ENCE = ' + str(round(total_ensemble_ence,2))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.92, 0.14,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
#ax.set_title('Ensemble')      
#ax.legend(loc='lower right', prop={'size': 6}) 
ax.xaxis.label.set_visible(False)
ax.yaxis.label.set_visible(False)

fig.tight_layout()
fig.savefig(TOTAL_GRAPHS_PATH + "ensemble_ence.png")


#Plot dropout ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), total_dropout_avg_policy, total_dropout_avg_uncertainty)
textstr = 'ENCE = ' + str(round(total_dropout_ence,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.92, 0.14,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
#ax.set_title('MC Dropout')   
#ax.legend(loc='lower right', prop={'size': 6})    
ax.xaxis.label.set_visible(False)
ax.yaxis.label.set_visible(False)
fig.tight_layout()
fig.savefig(TOTAL_GRAPHS_PATH + "mcdropout_ence.png")


#Plot swag ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), total_swag_avg_policy, total_swag_avg_uncertainty)
textstr = 'ENCE = ' + str(round(total_swag_ence,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.62, 0.94,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
#ax.set_title('SWAG')   
#ax.legend(loc='lower right', prop={'size': 6})    
ax.xaxis.label.set_visible(False)
ax.yaxis.label.set_visible(False)
fig.tight_layout()
fig.savefig(TOTAL_GRAPHS_PATH + "swag_ence.png")

np_ensemble_ence = ence(mdp_solution['p'].squeeze(), np_ensembles_avg_policy, np_ensembles_avg_uncertainty)*10
#Plot ensembles ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), np_ensembles_avg_policy, np_ensembles_avg_uncertainty)
textstr = 'ENCE = ' + str(round(np_ensemble_ence,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.92, 0.14,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )

fig.tight_layout()
fig.savefig(NP_GRAPHS_PATH + "ensemble_ence.png")



#Plot gp ence
fig, ax = plot_ence(mdp_solution['p'].squeeze(), total_gp_avg_policy, total_gp_avg_uncertainty)
textstr = 'ENCE = ' + str(round(total_gp_ence,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.92, 0.14,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
#ax.set_title('SWAG')   
#ax.legend(loc='lower right', prop={'size': 6})    
ax.xaxis.label.set_visible(False)
fig.tight_layout()
fig.savefig(TOTAL_GRAPHS_PATH + "gp_ence.png")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Plot NF calibration curve

#Ensembles
fig, ax, nf_ensemble_CAE = calibration_curve(mdp_solution['p'], nf_ensembles_avg_policy, nf_ensembles_avg_uncertainty)
textstr = 'ENEE = ' + str(round(nf_ensemble_CAE,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.96, 0.127,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
fig.savefig(NF_GRAPHS_PATH + "ensembles_cc.png")

#Dropout
fig, ax, nf_dropout_CAE = calibration_curve(mdp_solution['p'], nf_dropout_avg_policy, nf_dropout_avg_uncertainty)
textstr = 'ENEE = ' + str(round(nf_dropout_CAE,4))

props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.96, 0.127,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
fig.savefig(NF_GRAPHS_PATH + "mcdropout_cc.png")


#SWAG
fig, ax, nf_swag_CAE = calibration_curve(mdp_solution['p'], nf_swag_avg_policy, nf_swag_avg_uncertainty) 
textstr = 'ENEE = ' + str(round(nf_swag_CAE,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.62, 0.94,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
fig.savefig(NF_GRAPHS_PATH + "swag_cc.png")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Plot NP calibration curve

#Ensembles
fig, ax, np_ensemble_CAE = calibration_curve(mdp_solution['p'], np_ensembles_avg_policy, np_ensembles_avg_uncertainty)
textstr = 'ENEE = ' + str(round(np_ensemble_CAE,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.96, 0.127,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
fig.savefig(NP_GRAPHS_PATH + "ensembles_cc.png")



#Dropout
fig, ax, np_dropout_CAE = calibration_curve(mdp_solution['p'], np_dropout_avg_policy, np_dropout_avg_uncertainty)
textstr = 'ENEE = ' + str(round(np_dropout_CAE,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.96, 0.127,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
fig.savefig(NP_GRAPHS_PATH + "mcdropout_cc.png")


#SWAG
fig, ax, np_swag_CAE = calibration_curve(mdp_solution['p'], np_swag_avg_policy, np_swag_avg_uncertainty)
textstr = 'ENEE = ' + str(round(np_swag_CAE,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.62, 0.94,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
fig.savefig(NP_GRAPHS_PATH + "swag_cc.png")


#GP
fig, ax, np_gp_CAE = calibration_curve(mdp_solution['p'], np_gp_avg_policy, np_gp_avg_uncertainty)
textstr = 'ENEE = ' + str(round(np_gp_CAE,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.62, 0.94,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
fig.savefig(NP_GRAPHS_PATH + "gp_cc.png")

#GP
fig, ax, nf_gp_CAE = calibration_curve(mdp_solution['p'], nf_gp_avg_policy, nf_gp_avg_uncertainty)
textstr = 'ENEE = ' + str(round(nf_gp_CAE,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.96, 0.127,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
fig.savefig(NF_GRAPHS_PATH + "gp_cc.png")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Plot total calibration curve

#Ensembles
fig, ax, total_ensemble_CAE = calibration_curve(mdp_solution['p'], total_ensembles_avg_policy, total_ensembles_avg_uncertainty)
textstr = 'ENEE = ' + str(round(total_ensemble_CAE,2))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.96, 0.127,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
fig.savefig(TOTAL_GRAPHS_PATH + "ensembles_cc.png")

#Dropout
fig, ax, total_dropout_CAE = calibration_curve(mdp_solution['p'], total_dropout_avg_policy, total_dropout_avg_uncertainty)
textstr = 'ENEE = ' + str(round(total_dropout_CAE,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.96, 0.127,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
fig.savefig(TOTAL_GRAPHS_PATH + "mcdropout_cc.png")

#SWAG
fig, ax, total_swag_CAE = calibration_curve(mdp_solution['p'], total_swag_avg_policy, total_swag_avg_uncertainty)
textstr = 'ENEE = ' + str(round(total_swag_CAE,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.62, 0.94,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
fig.savefig(TOTAL_GRAPHS_PATH + "swag_cc.png")


#GP
fig, ax, total_gp_CAE = calibration_curve(mdp_solution['p'], total_gp_avg_policy, total_gp_avg_uncertainty)
textstr = 'ENEE = ' + str(round(total_gp_CAE,4))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax.text(0.96, 0.127,textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
        )
fig.savefig(TOTAL_GRAPHS_PATH + "gp_cc.png")


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Obtaining total noise
# Stores for each variant

paths_total_added_noise = [[], [], []]
features_total_added_noise = [[], [], []]
states_to_remove = [np.arange(0, 32, 1), np.arange(0, 64, 1), np.arange(0, 128, 1)]

indexes = [0,1,2]
for index_states_to_remove in indexes:
    for num_paths in [64, 128, 256, 512, 1024]:
        
        open_file = open(str(num_paths) + "_NNIRL_param_list.pkl", "rb")
        NNIRL_param_list = pickle.load(open_file)

        example_samples = NNIRL_param_list[9]
        feature_data = NNIRL_param_list[13] 

        true_sum_of_features = torch.sum(feature_data['splittable']).item()
        true_sum_of_paths = 0
        for path in example_samples:
            for move in path:
                true_sum_of_paths += sum(move)


        # Remove chosen states from paths
        if states_to_remove[index_states_to_remove] is not None:
            N = len(example_samples)
            top_index = math.ceil(0.5*N)
            twenty_percent_example_samples = example_samples[0:top_index]
            for path in twenty_percent_example_samples:
                T = len(path)
                pathindex = twenty_percent_example_samples.index(path)
                for move in path:
                    moveindex = twenty_percent_example_samples[pathindex].index(move)
                    #remove state
                    if move[0] in states_to_remove[index_states_to_remove]:
                        newmove = move
                        #get new state thats not in states to remove
                        newmove = ( random.randint(states_to_remove[index_states_to_remove][-1]+1, 255), move[1])
                        #assign new to state to curr step in paths
                        twenty_percent_example_samples[pathindex][moveindex] = newmove       
            example_samples[0:top_index] = twenty_percent_example_samples  

        # Add noise to features
        if states_to_remove[index_states_to_remove] is not None:
            for state in states_to_remove[index_states_to_remove]:
                if random.randint(0,100) < 3: #3% chance of NOT using this state
                    break
                for i in range(len(feature_data['splittable'][state,:])):
                    if random.randint(0,100) < 22: #22% chance of inverting the feature
                        #invert the feature, works since binary features
                        feature_data['splittable'][state,i] =  1-feature_data['splittable'][state,i]

        feature_diff  = abs(torch.sum(feature_data['splittable']).item() - true_sum_of_features)

        noisy_sum_of_paths = 0
        for path in example_samples:
            for move in path:
                noisy_sum_of_paths += sum(move)
        paths_diff = abs(noisy_sum_of_paths - true_sum_of_paths)

        paths_total_added_noise[indexes.index(index_states_to_remove)].append(paths_diff)
        features_total_added_noise[indexes.index(index_states_to_remove)].append(feature_diff)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TOTAL, NP & NF: Calculate Coefficient of Variation (Cv) for uncertainty predictions

# Useful for the case when ENCE = 0
np_dropout_cv = variation(np_dropout_avg_uncertainty)
np_ensemble_cv = variation(np_ensembles_avg_uncertainty)
np_swag_cv = variation(np_swag_avg_uncertainty)
np_gp_cv = variation(np_gp_avg_uncertainty)

nf_dropout_cv = variation(nf_dropout_avg_uncertainty)
nf_ensemble_cv = variation(nf_ensembles_avg_uncertainty)
nf_swag_cv = variation(nf_swag_avg_uncertainty)
nf_gp_cv = variation(nf_gp_avg_uncertainty)

total_dropout_cv = variation(total_dropout_avg_uncertainty)
total_ensemble_cv = variation(total_ensembles_avg_uncertainty)
total_swag_cv = variation(total_swag_avg_uncertainty)
total_gp_cv = variation(total_gp_avg_uncertainty)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#T-tests

#Get avg stds
np_dropout_std_0 = []
nf_dropout_std_0 = []
total_dropout_std_0 = []
np_dropout_std_1 = []
nf_dropout_std_1 = []
total_dropout_std_1 = []
np_dropout_std_2 = []
nf_dropout_std_2 = []
total_dropout_std_2 = []
np_ensemble_std_0 = []
nf_ensemble_std_0 = []
total_ensemble_std_0 = []
np_ensemble_std_1 = []
nf_ensemble_std_1 = []
total_ensemble_std_1 = []
np_ensemble_std_2 = []
nf_ensemble_std_2 = []
total_ensemble_std_2 = []
np_swag_std_0 = []
nf_swag_std_0 = []
total_swag_std_0 = []
np_swag_std_1 = []
nf_swag_std_1 = []
total_swag_std_1 = []
np_swag_std_2 = []
nf_swag_std_2 = []
total_swag_std_2 = []
np_gp_std_0 = []
nf_gp_std_0 = []
total_gp_std_0 = []
np_gp_std_1 = []
nf_gp_std_1 = []
total_gp_std_1 = []
np_gp_std_2 = []
nf_gp_std_2 = []
total_gp_std_2 = []


for paths in [128, 256, 512, 1024]:

    np_dropout_std_0.append(np_results['ow'][paths][1.0][0]['y_mc_std_relu'])
    nf_dropout_std_0.append(nf_results['ow'][paths][1.0][0]['y_mc_std_relu'])
    total_dropout_std_0.append(total_dropout_results['ow'][paths][1.0][0]['y_mc_std_relu'])
    np_dropout_std_1.append(np_results['ow'][paths][1.0][1]['y_mc_std_relu'])
    nf_dropout_std_1.append(nf_results['ow'][paths][1.0][1]['y_mc_std_relu'])
    total_dropout_std_1.append(total_dropout_results['ow'][paths][1.0][1]['y_mc_std_relu'])
    np_dropout_std_2.append(np_results['ow'][paths][1.0][2]['y_mc_std_relu'])
    nf_dropout_std_2.append(nf_results['ow'][paths][1.0][2]['y_mc_std_relu'])
    total_dropout_std_2.append(total_dropout_results['ow'][paths][1.0][2]['y_mc_std_relu'])

    np_ensemble_std_0.append(np_ensemble_results['ow'][paths][0.0][0]['y_mc_std_relu'])
    nf_ensemble_std_0.append(nf_ensemble_results['ow'][paths][0.0][0]['y_mc_std_relu'])
    total_ensemble_std_0.append(total_ensemble_results['ow'][paths][0.0][0]['y_mc_std_relu'])
    np_ensemble_std_1.append(np_results['ow'][paths][0.0][1]['y_mc_std_relu'])
    nf_ensemble_std_1.append(nf_results['ow'][paths][0.0][1]['y_mc_std_relu'])
    total_ensemble_std_1.append(total_ensemble_results['ow'][paths][0.0][1]['y_mc_std_relu'])
    np_ensemble_std_2.append(np_results['ow'][paths][0.0][2]['y_mc_std_relu'])
    nf_ensemble_std_2.append(nf_results['ow'][paths][0.0][2]['y_mc_std_relu'])
    total_ensemble_std_2.append(total_ensemble_results['ow'][paths][0.0][2]['y_mc_std_relu'])

    np_swag_std_0.append(np_swag_results['ow'][paths][0.0][0]['y_mc_std_relu'])
    nf_swag_std_0.append(nf_swag_results['ow'][paths][0.0][0]['y_mc_std_relu'])
    total_swag_std_0.append(total_swag_results['ow'][paths][0.0][0]['y_mc_std_relu'])
    np_swag_std_1.append(np_results['ow'][paths][0.0][1]['y_mc_std_relu'])
    nf_swag_std_1.append(nf_results['ow'][paths][0.0][1]['y_mc_std_relu'])
    total_swag_std_1.append(total_swag_results['ow'][paths][0.0][1]['y_mc_std_relu'])
    np_swag_std_2.append(np_results['ow'][paths][0.0][2]['y_mc_std_relu'])
    nf_swag_std_2.append(nf_results['ow'][paths][0.0][2]['y_mc_std_relu'])
    total_swag_std_2.append(total_swag_results['ow'][paths][0.0][2]['y_mc_std_relu'])
    
    np_gp_std_0.append(np_gp_results['ow'][paths][0.0][0]['y_mc_std_relu'])
    nf_gp_std_0.append(nf_gp_results['ow'][paths][0.0][0]['y_mc_std_relu'])
    total_gp_std_0.append(total_gp_results['ow'][paths][0.0][0]['y_mc_std_relu'])
    np_gp_std_1.append(np_results['ow'][paths][0.0][1]['y_mc_std_relu'])
    nf_gp_std_1.append(nf_results['ow'][paths][0.0][1]['y_mc_std_relu'])
    total_gp_std_1.append(total_gp_results['ow'][paths][0.0][1]['y_mc_std_relu'])
    np_gp_std_2.append(np_results['ow'][paths][0.0][2]['y_mc_std_relu'])
    nf_gp_std_2.append(nf_results['ow'][paths][0.0][2]['y_mc_std_relu'])
    total_gp_std_2.append(total_gp_results['ow'][paths][0.0][2]['y_mc_std_relu'])

#Get averages



np_dropout_std_0 = numpy.mean(numpy.array(np_dropout_std_0), axis=0)
nf_dropout_std_0 = numpy.mean(numpy.array(nf_dropout_std_0), axis=0)
total_dropout_std_0 = numpy.mean(numpy.array(total_dropout_std_0), axis=0)
np_dropout_std_1 = numpy.mean(numpy.array(np_dropout_std_1), axis=0)
nf_dropout_std_1 = numpy.mean(numpy.array(nf_dropout_std_1), axis=0)
total_dropout_std_1 = numpy.mean(numpy.array(total_dropout_std_1), axis=0)
np_dropout_std_2 = numpy.mean(numpy.array(np_dropout_std_2), axis=0)
nf_dropout_std_2 = numpy.mean(numpy.array(nf_dropout_std_2), axis=0)
total_dropout_std_2 = numpy.mean(numpy.array(total_dropout_std_2), axis=0)

np_ensemble_std_0 = numpy.mean(numpy.array(np_ensemble_std_0), axis=0)
nf_ensemble_std_0 = numpy.mean(numpy.array(nf_ensemble_std_0), axis=0)
total_ensemble_std_0 = numpy.mean(numpy.array(total_ensemble_std_0), axis=0)
np_ensemble_std_1 = numpy.mean(numpy.array(np_ensemble_std_1), axis=0)
nf_ensemble_std_1 = numpy.mean(numpy.array(nf_ensemble_std_1), axis=0)
total_ensemble_std_1 = numpy.mean(numpy.array(total_ensemble_std_1), axis=0)
np_ensemble_std_2 = numpy.mean(numpy.array(np_ensemble_std_2), axis=0)
nf_ensemble_std_2 = numpy.mean(numpy.array(nf_ensemble_std_2), axis=0)
total_ensemble_std_2 = numpy.mean(numpy.array(total_ensemble_std_2), axis=0)

np_swag_std_0 = numpy.mean(numpy.array(np_swag_std_0), axis=0)
nf_swag_std_0 = numpy.mean(numpy.array(nf_swag_std_0), axis=0)
total_swag_std_0 = numpy.mean(numpy.array(total_swag_std_0), axis=0)
np_swag_std_1 = numpy.mean(numpy.array(np_swag_std_1), axis=0)
nf_swag_std_1 = numpy.mean(numpy.array(nf_swag_std_1), axis=0)
total_swag_std_1 = numpy.mean(numpy.array(total_swag_std_1), axis=0)
np_swag_std_2 = numpy.mean(numpy.array(np_swag_std_2), axis=0)
nf_swag_std_2 = numpy.mean(numpy.array(nf_swag_std_2), axis=0)
total_swag_std_2 = numpy.mean(numpy.array(total_swag_std_2), axis=0)

np_gp_std_0 = numpy.mean(numpy.array(np_gp_std_0), axis=0)
nf_gp_std_0 = numpy.mean(numpy.array(nf_gp_std_0), axis=0)
total_gp_std_0 = numpy.mean(numpy.array(total_gp_std_0), axis=0)
np_gp_std_1 = numpy.mean(numpy.array(np_gp_std_1), axis=0)
nf_gp_std_1 = numpy.mean(numpy.array(nf_gp_std_1), axis=0)
total_gp_std_1 = numpy.mean(numpy.array(total_gp_std_1), axis=0)
np_gp_std_2 = numpy.mean(numpy.array(np_gp_std_2), axis=0)
nf_gp_std_2 = numpy.mean(numpy.array(nf_gp_std_2), axis=0)
total_gp_std_2 = numpy.mean(numpy.array(total_gp_std_2), axis=0)


#Obtain t and p values for each test variant
np_dropout_t_0, np_dropout_p_0 = ttest_ind(np_dropout_std_0[0:32], np_dropout_std_0[32:], equal_var=False)
np_dropout_t_1, np_dropout_p_1 = ttest_ind(np_dropout_std_1[0:64], np_dropout_std_1[64:], equal_var=False)
np_dropout_t_2, np_dropout_p_2 = ttest_ind(np_dropout_std_1[0:128], np_dropout_std_1[128:], equal_var=False)
np_dropout_cv_0 = variation(np_dropout_std_0)
np_dropout_cv_1 = variation(np_dropout_std_1)
np_dropout_cv_2 = variation(np_dropout_std_2)


np_ensemble_t_0, np_ensemble_p_0 = ttest_ind(np_ensemble_std_0[0:32], np_ensemble_std_0[32:], equal_var=False)
np_ensemble_t_1, np_ensemble_p_1 = ttest_ind(np_ensemble_std_1[0:64], np_ensemble_std_1[64:], equal_var=False)
np_ensemble_t_2, np_ensemble_p_2 = ttest_ind(np_ensemble_std_1[0:128], np_ensemble_std_1[128:], equal_var=False)
np_ensemble_cv_0 = variation(np_ensemble_std_0)
np_ensemble_cv_1 = variation(np_ensemble_std_1)
np_ensemble_cv_2 = variation(np_ensemble_std_2)


np_swag_t_0, np_swag_p_0 = ttest_ind(np_swag_std_0[0:32], np_swag_std_0[32:], equal_var=False)
np_swag_t_1, np_swag_p_1 = ttest_ind(np_swag_std_1[0:64], np_swag_std_1[64:], equal_var=False)
np_swag_t_2, np_swag_p_2 = ttest_ind(np_swag_std_1[0:128], np_swag_std_1[128:], equal_var=False)
np_swag_cv_0 = variation(np_swag_std_0)
np_swag_cv_1 = variation(np_swag_std_1)
np_swag_cv_2 = variation(np_swag_std_2)


np_gp_t_0, np_gp_p_0 = ttest_ind(np_gp_std_0[0:32], np_gp_std_0[32:], equal_var=False)
np_gp_t_1, np_gp_p_1 = ttest_ind(np_gp_std_1[0:64], np_gp_std_1[64:], equal_var=False)
np_gp_t_2, np_gp_p_2 = ttest_ind(np_gp_std_1[0:128], np_gp_std_1[128:], equal_var=False)
np_gp_cv_0 = variation(np_gp_std_0)
np_gp_cv_1 = variation(np_gp_std_1)
np_gp_cv_2 = variation(np_gp_std_2)


nf_dropout_t_0, nf_dropout_p_0 = ttest_ind(nf_dropout_std_0[0:32], nf_dropout_std_0[32:], equal_var=False)
nf_dropout_t_1, nf_dropout_p_1 = ttest_ind(nf_dropout_std_1[0:64], nf_dropout_std_1[64:], equal_var=False)
nf_dropout_t_2, nf_dropout_p_2 = ttest_ind(nf_dropout_std_1[0:128], nf_dropout_std_1[128:], equal_var=False)
nf_dropout_cv_0 = variation(nf_dropout_std_0)
nf_dropout_cv_1 = variation(nf_dropout_std_1)
nf_dropout_cv_2 = variation(nf_dropout_std_2)


nf_ensemble_t_0, nf_ensemble_p_0 = ttest_ind(nf_ensemble_std_0[0:32], nf_ensemble_std_0[32:], equal_var=False)
nf_ensemble_t_1, nf_ensemble_p_1 = ttest_ind(nf_ensemble_std_1[0:64], nf_ensemble_std_1[64:], equal_var=False)
nf_ensemble_t_2, nf_ensemble_p_2 = ttest_ind(nf_ensemble_std_1[0:128], nf_ensemble_std_1[128:], equal_var=False)
nf_ensemble_cv_0 = variation(nf_ensemble_std_0)
nf_ensemble_cv_1 = variation(nf_ensemble_std_1)
nf_ensemble_cv_2 = variation(nf_ensemble_std_2)


nf_swag_t_0, nf_swag_p_0 = ttest_ind(nf_swag_std_0[0:32], nf_swag_std_0[32:], equal_var=False)
nf_swag_t_1, nf_swag_p_1 = ttest_ind(nf_swag_std_1[0:64], nf_swag_std_1[64:], equal_var=False)
nf_swag_t_2, nf_swag_p_2 = ttest_ind(nf_swag_std_1[0:128], nf_swag_std_1[128:], equal_var=False)
nf_swag_cv_0 = variation(nf_swag_std_0)
nf_swag_cv_1 = variation(nf_swag_std_1)
nf_swag_cv_2 = variation(nf_swag_std_2)



nf_gp_t_0, nf_gp_p_0 = ttest_ind(nf_gp_std_0[0:32], nf_gp_std_0[32:], equal_var=False)
nf_gp_t_1, nf_gp_p_1 = ttest_ind(nf_gp_std_1[0:64], nf_gp_std_1[64:], equal_var=False)
nf_gp_t_2, nf_gp_p_2 = ttest_ind(nf_gp_std_1[0:128], nf_gp_std_1[128:], equal_var=False)
nf_gp_cv_0 = variation(nf_gp_std_0)
nf_gp_cv_1 = variation(nf_gp_std_1)
nf_gp_cv_2 = variation(nf_gp_std_2)



total_dropout_t_0, total_dropout_p_0 = ttest_ind(total_dropout_std_0[0:32], total_dropout_std_0[32:], equal_var=False)
total_dropout_t_1, total_dropout_p_1 = ttest_ind(total_dropout_std_1[0:64], total_dropout_std_1[64:], equal_var=False)
total_dropout_t_2, total_dropout_p_2 = ttest_ind(total_dropout_std_1[0:128], total_dropout_std_1[128:], equal_var=False)
total_dropout_cv_0 = variation(total_dropout_std_0)
total_dropout_cv_1 = variation(total_dropout_std_1)
total_dropout_cv_2 = variation(total_dropout_std_2)

total_ensemble_t_0, total_ensemble_p_0 = ttest_ind(total_ensemble_std_0[0:32], total_ensemble_std_0[32:], equal_var=False)
total_ensemble_t_1, total_ensemble_p_1 = ttest_ind(total_ensemble_std_1[0:64], total_ensemble_std_1[64:], equal_var=False)
total_ensemble_t_2, total_ensemble_p_2 = ttest_ind(total_ensemble_std_1[0:128], total_ensemble_std_1[128:], equal_var=False)
total_ensemble_cv_0 = variation(total_ensemble_std_0)
total_ensemble_cv_1 = variation(total_ensemble_std_1)
total_ensemble_cv_2 = variation(total_ensemble_std_2)


total_swag_t_0, total_swag_p_0 = ttest_ind(total_swag_std_0[0:32], total_swag_std_0[32:], equal_var=False)
total_swag_t_1, total_swag_p_1 = ttest_ind(total_swag_std_1[0:64], total_swag_std_1[64:], equal_var=False)
total_swag_t_2, total_swag_p_2 = ttest_ind(total_swag_std_1[0:128], total_swag_std_1[128:], equal_var=False)
total_swag_cv_0 = variation(total_swag_std_0)
total_swag_cv_1 = variation(total_swag_std_1)
total_swag_cv_2 = variation(total_swag_std_2)


total_gp_t_0, total_gp_p_0 = ttest_ind(total_gp_std_0[0:32], total_gp_std_0[32:], equal_var=False)
total_gp_t_1, total_gp_p_1 = ttest_ind(total_gp_std_1[0:64], total_gp_std_1[64:], equal_var=False)
total_gp_t_2, total_gp_p_2 = ttest_ind(total_gp_std_1[0:128], total_gp_std_1[128:], equal_var=False)
total_gp_cv_0 = variation(total_gp_std_0)
total_gp_cv_1 = variation(total_gp_std_1)
total_gp_cv_2 = variation(total_gp_std_2)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Write metrics to cvs

row_list = [["Metric", "NP", "NF", "Total", "NP", "NF", "Total", "NP", "NF", "Total", "NP", "NF", "Total"],
            ["ENCE", np_dropout_ence, nf_dropout_ence, total_dropout_ence, np_ensemble_ence, nf_ensemble_ence, total_ensemble_ence, np_swag_ence, nf_swag_ence, total_swag_ence, np_gp_ence, nf_gp_ence, total_gp_ence],
            ["ENEE", np_dropout_CAE, nf_dropout_CAE, total_dropout_CAE, np_ensemble_CAE, nf_ensemble_CAE, total_ensemble_CAE, np_swag_CAE, nf_swag_CAE, total_swag_CAE, np_gp_CAE, nf_gp_CAE, total_gp_CAE],
            ["Cv", np_dropout_cv[0], nf_dropout_cv[0], total_dropout_cv[0], np_ensemble_cv[0], nf_ensemble_cv[0], total_ensemble_cv[0], np_swag_cv[0], nf_swag_cv[0], total_swag_cv[0], np_gp_cv[0], nf_gp_cv[0], total_gp_cv[0]]]

with open("/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/figures/results/"+ 'final_error_calibrated_metrics_raw.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list)

row_list0 = [["Metric", "NP", "NF", "Total", "NP", "NF", "Total", "NP", "NF", "Total", "NP", "NF", "Total"],
            ["t", np_dropout_t_0, nf_dropout_t_0, total_dropout_t_0, np_ensemble_t_0, nf_ensemble_t_0, total_ensemble_t_0, np_swag_t_0, nf_swag_t_0, total_swag_t_0, np_gp_t_0, nf_gp_t_0, total_gp_t_0],
            ["p", np_dropout_p_0, nf_dropout_p_0, total_dropout_p_0, np_ensemble_p_0, nf_ensemble_p_0, total_ensemble_p_0, np_swag_p_0, nf_swag_p_0, total_swag_p_0, np_gp_p_0, nf_gp_p_0, total_gp_p_0],
            ["Cv", np_dropout_cv_0, nf_dropout_cv_0, total_dropout_cv_0, np_ensemble_cv_0, nf_ensemble_cv_0, total_ensemble_cv_0, np_swag_cv_0, nf_swag_cv_0, total_swag_cv_0, np_gp_cv_0, nf_gp_cv_0, total_gp_cv_0]]

with open("/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/figures/results/"+ '0_t_test_metrics_raw.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list0)

row_list1 = [["Metric", "NP", "NF", "Total", "NP", "NF", "Total", "NP", "NF", "Total", "NP", "NF", "Total"],
            ["t", np_dropout_t_1, nf_dropout_t_1, total_dropout_t_1, np_ensemble_t_1, nf_ensemble_t_1, total_ensemble_t_1, np_swag_t_1, nf_swag_t_1, total_swag_t_1, np_gp_t_1, nf_gp_t_1, total_gp_t_1],
            ["p", np_dropout_p_1, nf_dropout_p_1, total_dropout_p_1, np_ensemble_p_1, nf_ensemble_p_1, total_ensemble_p_1, np_swag_p_1, nf_swag_p_1, total_swag_p_1, np_gp_p_1, nf_gp_p_1, total_gp_p_1],
            ["Cv", np_dropout_cv_1, nf_dropout_cv_1, total_dropout_cv_1, np_ensemble_cv_1, nf_ensemble_cv_1, total_ensemble_cv_1, np_swag_cv_1, nf_swag_cv_1, total_swag_cv_1, np_gp_cv_1, nf_gp_cv_1, total_gp_cv_1]]

with open("/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/figures/results/"+ '1_t_test_metrics_raw.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list1)

row_list2 = [["Metric", "NP", "NF", "Total", "NP", "NF", "Total", "NP", "NF", "Total", "NP", "NF", "Total"],
            ["t", np_dropout_t_2, nf_dropout_t_2, total_dropout_t_2, np_ensemble_t_2, nf_ensemble_t_2, total_ensemble_t_2, np_swag_t_2, nf_swag_t_2, total_swag_t_2, np_gp_t_2, nf_gp_t_2, total_gp_t_2],
            ["p", np_dropout_p_2, nf_dropout_p_2, total_dropout_p_2, np_ensemble_p_2, nf_ensemble_p_2, total_ensemble_p_2, np_swag_p_2, nf_swag_p_2, total_swag_p_2, np_gp_p_2, nf_gp_p_2, total_gp_p_2],
            ["Cv", np_dropout_cv_2, nf_dropout_cv_2, total_dropout_cv_2, np_ensemble_cv_2, nf_ensemble_cv_2, total_ensemble_cv_2, np_swag_cv_2, nf_swag_cv_2, total_swag_cv_2, np_gp_cv_2, nf_gp_cv_2, total_gp_cv_2]]

with open("/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/figures/results/"+ '2_t_test_metrics_raw.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list2)





