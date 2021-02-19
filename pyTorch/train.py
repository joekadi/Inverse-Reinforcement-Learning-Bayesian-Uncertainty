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
    index_states_to_remove = None

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

    # Initalise task on clearML
    task = Task.init(project_name='MSci-Project', task_name='Train - Regular')
    
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
            print('\n... training on GridWorld benchmark ... \n')
        elif worldtype == "objectworld" or worldtype == "ow" or worldtype == "obj":
            print('\n... training on ObjectWorld benchmark ... \n')
    else:
        print('\n... training on GridWorld benchmark ... \n')

    #Print true R loss 
    print('\n... true reward loss is', trueNLL.item() ,'... \n')
    
    # Initalise loss function
    NLL = NLLFunction()
   
    # Connect configuration dict
    configuration_dict = {'number_of_epochs': 1, 'base_lr': 0.05, 'p': 0.02, 'no_hidden_layers': 3, 'no_neurons_in_hidden_layers': len(feature_data['splittable'][0])*2 } #set config params for clearml
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

    # Define network
    model = LitModel(len(feature_data['splittable'][0]), 'relu', configuration_dict)
    
    # Assign constants
    model.NLL = NLL
    model.F = feature_data['splittable']
    model.muE = muE
    model.mu_sa = mu_sa
    model.initD = initD
    model.mdp_data = mdp_data
    model.truep = truep
    model.configuration_dict = configuration_dict


    # Train model
    start_time = time.time()
    trainer.fit(model, train_loader)
    run_time = (time.time() - start_time)
    print('\n... Finished training models ...\n')
    
    # Create path for trained models
    TRAINED_MODELS_PATH = "./regular/models/"
    for path in [TRAINED_MODELS_PATH]:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    # Save model and new features
    torch.save(model.model, TRAINED_MODELS_PATH +str(len(example_samples))+'_REG_model.pth') 
    tensorboard_writer.close()