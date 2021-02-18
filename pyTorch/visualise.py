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

    states_to_remove = [np.arange(0, 32, 1), np.arange(0, 64, 1), np.arange(0, 128, 1)]
    #get which states to remove
    if len(sys.argv) > 1:
        index_states_to_remove = int(str(sys.argv[1]))
        print('\n... got which noisey states from cmd line ...\n')
    else:
        index_states_to_remove = 0
        print('\n... got which noisey states from pre-defined variable ...\n')

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

    #Load results from NP_eval to plot 
    file_name = './NP_results/results_'+str(index_states_to_remove)+'.pkl'
    open_file = open(file_name, "rb")
    results = pickle.load(open_file)
    open_file.close()
    y_mc_relu = results[0]
    y_mc_std_relu = results[1]
    y_mc_relu_reward = results[2]
    y_mc_relu_v = results[3]
    y_mc_relu_P = results[4]
    y_mc_relu_q = results[5]

    # Plot regression line w/ uncertainty shading
    f, ax1 = plt.subplots()
    ax1.plot(np.arange(1,len(feature_data['splittable'])+1,1), y_mc_relu, alpha=0.8)
    ax1.fill_between(np.arange(1,len(feature_data['splittable'])+1,1), (y_mc_relu-2*y_mc_std_relu).squeeze(), (y_mc_relu+2*y_mc_std_relu).squeeze(), alpha=0.3)
    ax1.axvline(states_to_remove[index_states_to_remove][-1], color='g',linestyle='--')
    ax1.set_title('w/ ReLU non-linearities')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Reward')
    plt.show()

    # Convert std arrays to correct size for final figures
    y_mc_std_relu_resized = torch.from_numpy(y_mc_std_relu)
    y_mc_std_relu_resized = y_mc_std_relu_resized.reshape(len(y_mc_std_relu_resized), 1)
    y_mc_std_relu_resized = y_mc_std_relu_resized.repeat((1, 5))

    # Result dict for predicitons with ReLU non-linearities
    irl_result_relu = { 
        'r': y_mc_relu_reward,
        'v': y_mc_relu_v,
        'p': y_mc_relu_P,
        'q': y_mc_relu_q,
        'r_itr': [y_mc_relu_reward],
        'model_r_itr': [y_mc_relu_reward],
        'p_itr': [y_mc_relu_P],
        'model_p_itr':[y_mc_relu_P],
        #'time': run_time,
        'uncertainty': y_mc_std_relu_resized,
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

    # Plot final figures for predicitons with ReLU non-linearities
    if(user_input):
        if worldtype == "gridworld" or worldtype == "gw" or worldtype == "grid":
            gwVisualise(test_result_relu)
        elif worldtype == "objectworld" or worldtype == "ow" or worldtype == "obj":
            owVisualise(test_result_relu)
    else:
        gwVisualise(test_result_relu)
    




