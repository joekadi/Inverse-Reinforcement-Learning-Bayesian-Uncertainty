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


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def save_NP_vars_for_GP(mdp_data, example_samples, feature_data, mdp_solution, num_paths, index_states_to_remove):
    """
    Saves "noisey" parameters to .mat file for GPIRL to use
    Since GPIRL is used for baseline comparison of uncertainty estimates
    """

    #Convert everything to numpy so that savemat parses maintains correct structure
    
    # Cast example samples to numpy 
    np_example_samples = np.array(example_samples)+1 #+1 since matlab don't 0 index

    # Cast mdp_data to numpy
    np_mdp_data = {}
    np_mdp_data['states'] = mdp_data['states']
    np_mdp_data['actions'] = mdp_data['actions']
    np_mdp_data['discount'] = mdp_data['discount']
    np_mdp_data['determinism'] = mdp_data['determinism']
    np_mdp_data['sa_s'] = mdp_data['sa_s'].detach().numpy()+1 #+1 since matlab don't 0 index
    np_mdp_data['sa_p'] = mdp_data['sa_p'].detach().numpy()
    np_mdp_data['map1'] = mdp_data['map1'].detach().numpy()
    np_mdp_data['map2'] = mdp_data['map2'].detach().numpy()
    np_mdp_data['c1array'] = mdp_data['c1array']
    np_mdp_data['c2array'] = mdp_data['c2array']


    np_feature_data = {}
    np_feature_data['stateadjacency'] = feature_data['stateadjacency']
    np_feature_data['splittable'] = feature_data['splittable'].detach().numpy()

    np_mdp_solution = {}
    np_mdp_solution['logp'] = mdp_solution['logp'].detach().numpy()
    np_mdp_solution['p'] = mdp_solution['p'].detach().numpy()
    np_mdp_solution['q'] = mdp_solution['q'].detach().numpy()
    np_mdp_solution['v'] = mdp_solution['v'].detach().numpy()

    #Create path for GP params
    GP_VARS_PATH = "/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/pyTorch/variables_for_GPIRL/NP/"

    for path in [GP_VARS_PATH]:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    params_for_GP = {'mdp_data': np_mdp_data, 'example_samples': np_example_samples, 'feature_data': np_feature_data, 'mdp_solution': np_mdp_solution}


    file_name = GP_VARS_PATH+'OW_'+ str(num_paths)+'_NP_model_'+str(index_states_to_remove)+'.mat' 
    savemat(file_name, params_for_GP)

def save_reg_vars_for_GP(mdp_data, example_samples, feature_data, mdp_solution, num_paths):
    """
    Saves "noisey" parameters to .mat file for GPIRL to use
    Since GPIRL is used for baseline comparison of uncertainty estimates
    """

    #Convert everything to numpy so that savemat parses maintains correct structure
    
    # Cast example samples to numpy 
    np_example_samples = np.array(example_samples)+1 #+1 since matlab don't 0 index

    # Cast mdp_data to numpy
    np_mdp_data = {}
    np_mdp_data['states'] = mdp_data['states']
    np_mdp_data['actions'] = mdp_data['actions']
    np_mdp_data['discount'] = mdp_data['discount']
    np_mdp_data['determinism'] = mdp_data['determinism']
    np_mdp_data['sa_s'] = mdp_data['sa_s'].detach().numpy()+1 #+1 since matlab don't 0 index
    np_mdp_data['sa_p'] = mdp_data['sa_p'].detach().numpy()
    np_mdp_data['map1'] = mdp_data['map1'].detach().numpy()
    np_mdp_data['map2'] = mdp_data['map2'].detach().numpy()
    np_mdp_data['c1array'] = mdp_data['c1array']
    np_mdp_data['c2array'] = mdp_data['c2array']


    np_feature_data = {}
    np_feature_data['stateadjacency'] = feature_data['stateadjacency']
    np_feature_data['splittable'] = feature_data['splittable'].detach().numpy()

    np_mdp_solution = {}
    np_mdp_solution['logp'] = mdp_solution['logp'].detach().numpy()
    np_mdp_solution['p'] = mdp_solution['p'].detach().numpy()
    np_mdp_solution['q'] = mdp_solution['q'].detach().numpy()
    np_mdp_solution['v'] = mdp_solution['v'].detach().numpy()

    #Create path for GP params
    GP_VARS_PATH = "/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/pyTorch/variables_for_GPIRL/reg/"

    for path in [GP_VARS_PATH]:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    params_for_GP = {'mdp_data': np_mdp_data, 'example_samples': np_example_samples, 'feature_data': np_feature_data, 'mdp_solution': np_mdp_solution}


    file_name = GP_VARS_PATH+'OW_'+ str(num_paths)+'_reg_model.mat' 
    savemat(file_name, params_for_GP)

def save_NF_vars_for_GP(mdp_data, example_samples, feature_data, mdp_solution, num_paths, index_states_to_remove):
    """
    Saves "noisey" parameters to .mat file for GPIRL to use
    Since GPIRL is used for baseline comparison of uncertainty estimates
    """

    #Convert everything to numpy so that savemat parses maintains correct structure
    
    # Cast example samples to numpy 
    np_example_samples = np.array(example_samples)+1 #+1 since matlab don't 0 index

    # Cast mdp_data to numpy
    np_mdp_data = {}
    np_mdp_data['states'] = mdp_data['states']
    np_mdp_data['actions'] = mdp_data['actions']
    np_mdp_data['discount'] = mdp_data['discount']
    np_mdp_data['determinism'] = mdp_data['determinism']
    np_mdp_data['sa_s'] = mdp_data['sa_s'].detach().numpy()+1 #+1 since matlab don't 0 index
    np_mdp_data['sa_p'] = mdp_data['sa_p'].detach().numpy()
    np_mdp_data['map1'] = mdp_data['map1'].detach().numpy()
    np_mdp_data['map2'] = mdp_data['map2'].detach().numpy()
    np_mdp_data['c1array'] = mdp_data['c1array']
    np_mdp_data['c2array'] = mdp_data['c2array']


    np_feature_data = {}
    np_feature_data['stateadjacency'] = feature_data['stateadjacency']
    np_feature_data['splittable'] = feature_data['splittable'].detach().numpy()

    np_mdp_solution = {}
    np_mdp_solution['logp'] = mdp_solution['logp'].detach().numpy()
    np_mdp_solution['p'] = mdp_solution['p'].detach().numpy()
    np_mdp_solution['q'] = mdp_solution['q'].detach().numpy()
    np_mdp_solution['v'] = mdp_solution['v'].detach().numpy()

    #Create path for GP params
    GP_VARS_PATH = "/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/pyTorch/variables_for_GPIRL/NF/"

    for path in [GP_VARS_PATH]:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    params_for_GP = {'mdp_data': np_mdp_data, 'example_samples': np_example_samples, 'feature_data': np_feature_data, 'mdp_solution': np_mdp_solution}


    file_name = GP_VARS_PATH+'OW_'+ str(num_paths)+'_NF_model_'+str(index_states_to_remove)+'.mat' 
    savemat(file_name, params_for_GP)

def save_total_vars_for_GP(mdp_data, example_samples, feature_data, mdp_solution, num_paths, index_states_to_remove):

    """
    Saves "noisey" parameters to .mat file for GPIRL to use
    Since GPIRL is used for baseline comparison of uncertainty estimates
    """

    #Convert everything to numpy so that savemat parses maintains correct structure
    
    # Cast example samples to numpy 
    np_example_samples = np.array(example_samples)+1 #+1 since matlab don't 0 index

    # Cast mdp_data to numpy
    np_mdp_data = {}
    np_mdp_data['states'] = mdp_data['states']
    np_mdp_data['actions'] = mdp_data['actions']
    np_mdp_data['discount'] = mdp_data['discount']
    np_mdp_data['determinism'] = mdp_data['determinism']
    np_mdp_data['sa_s'] = mdp_data['sa_s'].detach().numpy()+1 #+1 since matlab don't 0 index
    np_mdp_data['sa_p'] = mdp_data['sa_p'].detach().numpy()
    np_mdp_data['map1'] = mdp_data['map1'].detach().numpy()
    np_mdp_data['map2'] = mdp_data['map2'].detach().numpy()
    np_mdp_data['c1array'] = mdp_data['c1array']
    np_mdp_data['c2array'] = mdp_data['c2array']


    np_feature_data = {}
    np_feature_data['stateadjacency'] = feature_data['stateadjacency']
    np_feature_data['splittable'] = feature_data['splittable'].detach().numpy()

    np_mdp_solution = {}
    np_mdp_solution['logp'] = mdp_solution['logp'].detach().numpy()
    np_mdp_solution['p'] = mdp_solution['p'].detach().numpy()
    np_mdp_solution['q'] = mdp_solution['q'].detach().numpy()
    np_mdp_solution['v'] = mdp_solution['v'].detach().numpy()

    #Create path for GP params
    GP_VARS_PATH = "/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/pyTorch/variables_for_GPIRL/total/"

    for path in [GP_VARS_PATH]:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    params_for_GP = {'mdp_data': np_mdp_data, 'example_samples': np_example_samples, 'feature_data': np_feature_data, 'mdp_solution': np_mdp_solution}


    file_name = GP_VARS_PATH+'OW_'+ str(num_paths)+'_total_model_'+str(index_states_to_remove)+'.mat' 
    savemat(file_name, params_for_GP)


def save_vars(index_states_to_remove, num_paths, reg):


    states_to_remove = [np.arange(0, 32, 1), np.arange(0, 64, 1), np.arange(0, 128, 1)]


    print('\n\n... Saving for paths = ', num_paths, ' ...\n\n')

    # Load variables
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
    example_samples = NNIRL_param_list[9]
    mdp_params = NNIRL_param_list[10] 
    r = NNIRL_param_list[11] 
    mdp_solution = NNIRL_param_list[12] 
    feature_data = NNIRL_param_list[13] 
    trueNLL = NNIRL_param_list[14]
    normalise = NNIRL_param_list[15]
    user_input = NNIRL_param_list[16]
    worldtype = NNIRL_param_list[17]

    np_example_samples = example_samples
    nf_feature_data = feature_data

    if reg==False:
        # Remove chosen states from paths
        if states_to_remove[index_states_to_remove] is not None:
            N = len(example_samples)
            top_index = math.ceil(0.5*N)
            twenty_percent_example_samples = np_example_samples[0:top_index]
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
            np_example_samples[0:top_index] = twenty_percent_example_samples  
        
    if reg==False:
        # Add noise to features
        if states_to_remove[index_states_to_remove] is not None:
            print('\n... adding noise to features ...\n')
            for state in states_to_remove[index_states_to_remove]:
                if random.randint(0,100) < 3: #3% chance of NOT using this state
                    break
                for i in range(len(nf_feature_data['splittable'][state,:])):
                    if random.randint(0,100) < 22: #22% chance of inverting the feature
                        #invert the feature, works since binary features
                        nf_feature_data['splittable'][state,i] =  1-nf_feature_data['splittable'][state,i]


    #Save respective variables

    save_total_vars_for_GP(mdp_data, np_example_samples, nf_feature_data, mdp_solution, num_paths, index_states_to_remove)

    save_reg_vars_for_GP(mdp_data, example_samples, feature_data, mdp_solution, num_paths)

    if reg==False:
        save_NP_vars_for_GP(mdp_data, np_example_samples, feature_data, mdp_solution, num_paths, index_states_to_remove)
        save_NF_vars_for_GP(mdp_data, example_samples, nf_feature_data, mdp_solution, num_paths, index_states_to_remove)



if __name__ == "__main__":

    start_time = time.time()
    for world in ['ow']:
        for num_paths in [64,128,256,512,1024]:
                for variant in [0,1,2]:
                    save_vars(variant, num_paths, False)
    run_time = (time.time() - start_time)
    print('All variables saved in ' + str(run_time) + 'seconds')
