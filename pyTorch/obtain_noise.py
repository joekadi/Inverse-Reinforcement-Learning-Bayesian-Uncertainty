
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


torch.manual_seed(0)
np.random.seed(seed=0)
random.seed(0)

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

        '''
        if num_paths == 64:
            print('true paths')
            print(example_samples[60])
        '''


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
        '''
        if num_paths == 64:
            print('noisey paths')
            print(example_samples[60])
            os._exit(1)
        '''
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


print(paths_total_added_noise)
print(features_total_added_noise)