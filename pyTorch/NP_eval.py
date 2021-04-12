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

tensorboard_writer = SummaryWriter('./tensorboard_logs')
torch.set_printoptions(precision=5, sci_mode=False, threshold=1000)
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        index_states_to_remove = int(str(sys.argv[1]))
        print('\n... got which noisey states from cmd line ...\n')
    else:
        index_states_to_remove = 0
        print('\n... got which noisey states from pre-defined variable ...\n')

    
    if len(sys.argv) > 2:
        dropout_val = float(str(sys.argv[2]))
        print('\n... got dropout value from cmd line ...\n')
    else:
        dropout_val = 0.2
        print('\n... got dropout value from pre-defined variable ...\n')

    if len(sys.argv) > 3:
        num_paths = int(str(sys.argv[3]))
        print('\n... got number of paths value from cmd line ...\n')
    else:
        num_paths = 64
        print('\n... got dropout value from pre-defined variable ...\n')
    
    if index_states_to_remove < 0 or index_states_to_remove > 3:
        raise Exception("Index of states to remove from paths must be within range 0 - 3")


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
    #example_samples = NNIRL_param_list[9]
    mdp_params = NNIRL_param_list[10] 
    r = NNIRL_param_list[11] 
    mdp_solution = NNIRL_param_list[12] 
    feature_data = NNIRL_param_list[13] 
    trueNLL = NNIRL_param_list[14]
    normalise = NNIRL_param_list[15]
    user_input = NNIRL_param_list[16]
    worldtype = NNIRL_param_list[17]

    #Print what benchmark
    if(user_input):
        if worldtype == "gridworld" or worldtype == "gw" or worldtype == "grid":
            print('\n... evaluating on GridWorld benchmark ...\n')
        elif worldtype == "objectworld" or worldtype == "ow" or worldtype == "obj":
            print('\n... evaluating on ObjectWorld benchmark ...\n')
    else:
        print('\n... evaluating on GridWorld benchmark ... \n')

    #Load model
    irl_model = torch.load('./noisey_paths/models/'+str(worldtype)+'_'+str(dropout_val)+'_'+str(num_paths)+ '_NP_model_'+str(index_states_to_remove)+'.pth') 
    
    num_preds = 1000 # Number of samples
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

    if normalise:
        #Scale everything within 0 and 1
        scaler = MinMaxScaler()

        y_mc_relu = scaler.fit_transform(y_mc_relu.reshape(-1,1))
        y_mc_std_relu = scaler.fit_transform(y_mc_std_relu.reshape(-1,1))
        y_mc_tanh = scaler.fit_transform(y_mc_tanh.reshape(-1,1))
        y_mc_std_tanh = scaler.fit_transform(y_mc_std_tanh.reshape(-1,1))
    
    # Extract full reward function
    y_mc_relu_reward = torch.from_numpy(y_mc_relu)
    y_mc_relu_reward = y_mc_relu_reward.reshape(len(y_mc_relu_reward), 1)
    y_mc_relu_reward = y_mc_relu_reward.repeat((1, 5))

    #Solve with learned reward functions
    y_mc_relu_v, y_mc_relu_q, y_mc_relu_logp, y_mc_relu_P = linearvalueiteration(mdp_data, y_mc_relu_reward)


    '''
    # Print results
    print("\nTrue R has:\n - negated likelihood: {}\n - EVD: {}".format(trueNLL,  irl_model.NLL.calculate_EVD(truep, r)))
    print("\nPred R with ReLU activation has:\n - negated likelihood: {}\n - EVD: {}".format(irl_model.NLL.apply(y_mc_relu_reward, initD, mu_sa, muE, feature_data['splittable'], mdp_data), irl_model.NLL.calculate_EVD(truep, y_mc_relu_reward)))
    '''

    # Initalise loss function
    NLL = NLLFunction()
    # Assign loss function constants
    NLL.F = feature_data['splittable']
    NLL.muE = muE
    NLL.mu_sa = mu_sa
    NLL.initD = initD
    NLL.mdp_data = mdp_data


    #Save results
    print('\n... saving results ...\n')

    # Create path for trained models
    RESULTS_PATH = "./noisey_paths/results/"
    for path in [RESULTS_PATH]:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    NP_results = [y_mc_relu, y_mc_std_relu, y_mc_relu_reward, y_mc_relu_v, y_mc_relu_P, y_mc_relu_q, NLL.calculate_EVD(truep, y_mc_relu_reward), run_time, num_preds]
    file_name = RESULTS_PATH+str(worldtype)+'_'+str(dropout_val)+'_'+ str(num_paths)+ '_results_'+str(index_states_to_remove)+'.pkl'
    open_file = open(file_name, "wb")
    pickle.dump(NP_results, open_file)
    open_file.close()




