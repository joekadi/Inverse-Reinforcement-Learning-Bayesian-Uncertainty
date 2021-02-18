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

class LitModel(pl.LightningModule):

    NLL = None
    F = None
    muE = None
    mu_sa = None
    initD = None
    mdp_data = None
    truep = None
    learned_feature_weights = None
    configuration_dict = None

    def __init__(self, no_features, activation, configuration_dict):
        super().__init__()
        self.model = nn.Sequential()
        self.model.add_module('input', nn.Linear(no_features, configuration_dict['no_neurons_in_hidden_layers']))
        if activation == 'relu':
            self.model.add_module('relu0', nn.ReLU())
        elif activation == 'tanh':
            self.model.add_module('tanh0', nn.Tanh())
        for i in range(configuration_dict['no_hidden_layers']):
            self.model.add_module('dropout'+str(i+1), nn.Dropout(p=configuration_dict['p']))
            self.model.add_module('hidden'+str(i+1), nn.Linear(configuration_dict['no_neurons_in_hidden_layers'], configuration_dict['no_neurons_in_hidden_layers']))
            if activation == 'relu':
                self.model.add_module('relu'+str(i+1), nn.ReLU())
            elif activation == 'tanh':
                self.model.add_module('tanh'+str(i+1), nn.Tanh())
        self.model.add_module('dropout'+str(i+2), nn.Dropout(p=configuration_dict['p']))
        self.model.add_module('final', nn.Linear(configuration_dict['no_neurons_in_hidden_layers'], no_features))

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.configuration_dict['base_lr'], weight_decay=1e-2)
        return optimizer 

    def training_step(self, batch, batch_idx):
        X = batch
        #output = torch.empty(X.shape[1], 1, dtype=torch.double)
        output = self(X[0,:].view(-1)) 
        output = output.reshape(len(output), 1)
        loss = self.NLL.apply(output, self.initD, self.mu_sa, self.muE, self.F, self.mdp_data)
        evd = self.NLL.calculate_EVD(self.truep, torch.matmul(self.F, output))
        #self.learned_feature_weights = output #store current
        tensorboard_writer.add_scalar('train_loss',loss,batch_idx)
        tensorboard_writer.add_scalar('train_evd',evd,batch_idx)
        return loss


if __name__ == "__main__":


    noisey_features= False
    train_models = True
    noisey_paths = False
    plot_regression_line = False
    state_to_remove = 36

    T = 1000 # Number of samples

    # Initalise task on clearML
    task = Task.init(project_name='MSci-Project', task_name='LitModel Run, n=8, b=1, t=1000, no noise')
    
    # Load variables
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

    #Print what benchmark
    if(user_input):
        if worldtype == "gridworld" or worldtype == "gw" or worldtype == "grid":
            print('\n ... running on GridWorld benchmark ... \n')
        elif worldtype == "objectworld" or worldtype == "ow" or worldtype == "obj":
            print('\n ... running on ObjectWorld benchmark ... \n')
    else:
        print('\n ... running on GridWorld benchmark ... \n')

    #Print true R loss 
    print('\n ... true reward loss is', trueNLL.item() ,'... \n')
    
    # Initalise loss function
    NLL = NLLFunction()

    # Remove chosen state from paths
    if noisey_paths:
        print('\n... removing state', state_to_remove, 'from paths ...\n')
        N = len(example_samples)
        for path in example_samples:
            T = len(path)
            pathindex = example_samples.index(path)
            for move in path:
                moveindex = example_samples[pathindex].index(move)
                #remove state
                if move[0] == state_to_remove:
                    newmove = (random.randint(0, len(feature_data['splittable'][0]-1)), move[1])
                    example_samples[pathindex][moveindex] = newmove                
        initD, mu_sa, muE, F, mdp_data = NLL.calc_var_values(mdp_data, N, T, example_samples, feature_data)  # calculate required variables

    # Add noise to features
    if noisey_features:
        #add noise to features at states 12, 34 and 64 (when mdp_params.n=8)
        print('\n... adding noise to features at states 12, 34 and 64 ...\n')
        feature_data['splittable'][11,:] = torch.rand(feature_data['splittable'].shape[1])
        feature_data['splittable'][33,:] = torch.rand(feature_data['splittable'].shape[1])
        feature_data['splittable'][63,:] = torch.rand(feature_data['splittable'].shape[1])


    # Connect configuration dict
    configuration_dict = {'number_of_epochs': 2, 'base_lr': 0.05, 'p': 0.02, 'no_hidden_layers': 3, 'no_neurons_in_hidden_layers': len(feature_data['splittable'][0])*2 } #set config params for clearml
    configuration_dict = task.connect(configuration_dict)

    # Assign loss function constants
    NLL.F = feature_data['splittable']
    NLL.muE = muE
    NLL.mu_sa = mu_sa
    NLL.initD = initD
    NLL.mdp_data = mdp_data

    # Load features
    train_loader = torch.utils.data.DataLoader(feature_data['splittable'], num_workers = 8)
    
    # Define trainer
    trainer = pl.Trainer(max_epochs=configuration_dict['number_of_epochs'])

    # Define networks
    model2 = [LitModel(len(feature_data['splittable'][0]), 'relu', configuration_dict),LitModel(len(feature_data['splittable'][0]), 'tanh', configuration_dict)] #init model
    
    # Assign constants
    for model in model2:
        model.NLL = NLL
        model.F = feature_data['splittable']
        model.muE = muE
        model.mu_sa = mu_sa
        model.initD = initD
        model.mdp_data = mdp_data
        model.truep = truep
        model.configuration_dict = configuration_dict

    # Train models
    if train_models:
        start_time = time.time()
        [trainer.fit(model, train_loader) for model in model2]
        run_time = (time.time() - start_time)
        print('\n... Finished training models ...\n')

        # Save models
        PATH = './NN_IRL.pth'
        for ind, model in enumerate(model2):
            torch.save(model.model, 'IRL_model_'+str(ind)+'.pth') #maybe change to just PATH
        tensorboard_writer.close()
    
    irl_models = [torch.load('IRL_model_'+str(ind)+'.pth') for ind in [0,1]] # Load models

    # Make predicitons w/ trained models
    print('\n... Making predictions w/ trained models ...\n')
    for i in range(len(feature_data['splittable'])):
        Yt_hat_relu = np.array([torch.matmul(feature_data['splittable'],irl_models[0](feature_data['splittable'][i].view(-1)).reshape(len(feature_data['splittable'][0]),1)).data.cpu().numpy() for _ in range(T)]).squeeze()
        Yt_hat_tanh = np.array([torch.matmul(feature_data['splittable'], irl_models[1](feature_data['splittable'][i].view(-1)).reshape(len(feature_data['splittable'][0]),1)).data.cpu().numpy() for _ in range(T)]).squeeze()

    # Extract mean and std of predictions
    y_mc_relu = Yt_hat_relu.mean(axis=0)
    y_mc_std_relu = Yt_hat_relu.std(axis=0)

    y_mc_tanh = Yt_hat_tanh.mean(axis=0)
    y_mc_std_tanh = Yt_hat_tanh.std(axis=0)


    if normalise:
        #Scale everything within 0 and 1
        scaler = MinMaxScaler()

        y_mc_relu = scaler.fit_transform(y_mc_relu.reshape(-1,1))
        y_mc_std_relu = scaler.fit_transform(y_mc_std_relu.reshape(-1,1))
        y_mc_tanh = scaler.fit_transform(y_mc_tanh.reshape(-1,1))
        y_mc_std_tanh = scaler.fit_transform(y_mc_std_tanh.reshape(-1,1))
    
    

    # Extract full reward functions
    y_mc_relu_reward = torch.from_numpy(y_mc_relu)
    y_mc_relu_reward = y_mc_relu_reward.reshape(len(y_mc_relu_reward), 1)
    y_mc_relu_reward = y_mc_relu_reward.repeat((1, 5))

    y_mc_tanh_reward = torch.from_numpy(y_mc_tanh)
    y_mc_tanh_reward = y_mc_tanh_reward.reshape(len(y_mc_tanh_reward), 1)
    y_mc_tanh_reward = y_mc_tanh_reward.repeat((1, 5))

    #Solve with learned reward functions
    y_mc_relu_v, y_mc_relu_q, y_mc_relu_logp, y_mc_relu_P = linearvalueiteration(mdp_data, y_mc_relu_reward)
    y_mc_tanh_v, y_mc_tanh_q, y_mc_tanh_logp, y_mc_tanh_P = linearvalueiteration(mdp_data, y_mc_tanh_reward)


    # Print results
    print("\nTrue R has:\n - negated likelihood: {}\n - EVD: {}".format(trueNLL,  NLL.calculate_EVD(truep, r)))
    print("\nPred R with ReLU activation has:\n - negated likelihood: {}\n - EVD: {}".format(NLL.apply(y_mc_relu_reward, initD, mu_sa, muE, feature_data['splittable'], mdp_data), NLL.calculate_EVD(truep, y_mc_relu_reward)))
    print("\nPred R with TanH activation has:\n - negated likelihood: {}\n - EVD: {}\n".format(NLL.apply(y_mc_tanh_reward, initD, mu_sa, muE, feature_data['splittable'], mdp_data), NLL.calculate_EVD(truep, y_mc_tanh_reward)))
   

    if plot_regression_line:
        # Plot regression line w/ uncertainty shading
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        ax1.plot(np.arange(1,len(feature_data['splittable'])+1,1), y_mc_relu, alpha=0.8)
        ax1.fill_between(np.arange(1,len(feature_data['splittable'])+1,1), (y_mc_relu-2*y_mc_std_relu).squeeze(), (y_mc_relu+2*y_mc_std_relu).squeeze(), alpha=0.3)
        ax1.set_title('w/ ReLU non-linearities')
        ax1.set_xlabel('State')
        ax1.set_ylabel('Reward')

        ax2.plot(np.arange(1,len(feature_data['splittable'])+1,1), y_mc_tanh, alpha=0.8)
        ax2.fill_between(np.arange(1,len(feature_data['splittable'])+1,1), (y_mc_tanh-2*y_mc_std_tanh).squeeze(), (y_mc_tanh+2*y_mc_std_tanh).squeeze(), alpha=0.3)
        ax2.set_title('w/ TanH non-linearities')
        ax2.set_xlabel('State')
        plt.show()

    # Convert std arrays to correct size for final figures
    y_mc_std_relu_resized = torch.from_numpy(y_mc_std_relu)
    y_mc_std_relu_resized = y_mc_std_relu_resized.reshape(len(y_mc_std_relu_resized), 1)
    y_mc_std_relu_resized = y_mc_std_relu_resized.repeat((1, 5))

    y_mc_std_tanh_resized = torch.from_numpy(y_mc_std_tanh)
    y_mc_std_tanh_resized = y_mc_std_tanh_resized.reshape(len(y_mc_std_tanh_resized), 1)
    y_mc_std_tanh_resized = y_mc_std_tanh_resized.repeat((1, 5))


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
        'uncertainty_figure_title': 'Uncertainty w/ ReLU non-linearities',
        #'model_itr': [pred_feature_weights], #commented since feature weights never returned, only final R to matmul with features in the eval loop
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

    # Result dict for predicitons with TanH non-linearities
    irl_result_tanh = { 
        'r': y_mc_tanh_reward,
        'v': y_mc_tanh_v,
        'p': y_mc_tanh_P,
        'q': y_mc_tanh_q,
        'r_itr': [y_mc_tanh_reward],
        'model_r_itr': [y_mc_tanh_reward],
        'p_itr': [y_mc_tanh_P],
        'model_p_itr':[y_mc_tanh_P],
        #'time': run_time,
        'uncertainty': y_mc_std_tanh_resized,
        'truth_figure_title': 'Truth R & P',
        'pred_reward_figure_title': 'Pred R & P w/ TanH non-linearities',
        'uncertainty_figure_title': 'Uncertainty w/ TanH non-linearities',
        #'model_itr': [pred_feature_weights], #commented since feature weights never returned, only final R to matmul with features in the eval loop
    }

    # Ground truth dict for predicitons with Tanh non-linearities
    test_result_tanh = { 
        'irl_result': irl_result_tanh,
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

    # Plot final figures for predicitons with TanH non-linearities
    if(user_input):
        if worldtype == "gridworld" or worldtype == "gw" or worldtype == "grid":
            gwVisualise(test_result_tanh)
        elif worldtype == "objectworld" or worldtype == "ow" or worldtype == "obj":
            owVisualise(test_result_tanh)
    else:
        gwVisualise(test_result_tanh)
    




