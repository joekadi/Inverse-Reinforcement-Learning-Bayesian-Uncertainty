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
import pandas as pd
import time
import math as math
import random
import torchvision
import torchvision.transforms as transforms
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
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
print('\npl version:', pl.__version__)

class LitModel(nn.Module):

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

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha

if __name__ == "__main__":

    if len(sys.argv) > 1:
        index_states_to_remove = int(str(sys.argv[1]))
        print('\n... got which noisey states from cmd line ...\n')
    else:
        raise Exception("Index States To Remove not supplied")

    if len(sys.argv) > 2:
        dropout_val = float(str(sys.argv[2]))
        print('\n... got dropout value from cmd line ...\n')
    else:
        raise Exception("Dropout Val not supplied")

    if len(sys.argv) > 3:
        num_paths = int(str(sys.argv[3]))
        print('\n... got number of paths value from cmd line ...\n')
    else:
        raise Exception("Number of Paths not supplied")

    if index_states_to_remove < 0 or index_states_to_remove > 3:
        raise Exception("Index of states to remove from paths must be within range 0 - 3")

    states_to_remove = [np.arange(0, 32, 1), np.arange(0, 64, 1), np.arange(0, 128, 1)]

    #task = Task.init(project_name='MSci-Project', task_name='Train - Noisey paths')#Initalise task on clearML
    
    print('\n\n... Running for Paths = ', num_paths, ' ...\n\n')

    
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


    torch.manual_seed(mdp_params['seed'])
    np.random.seed(seed=mdp_params['seed'])
    random.seed(mdp_params['seed'])


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
        initD, mu_sa, muE, F, mdp_data = NLL.calc_var_values(mdp_data, N, T, example_samples, feature_data)  # calculate required variables


    # Connect configuration dict
    configuration_dict = {'number_of_epochs': 3, 'base_lr': 0.1, 'p': dropout_val, 'no_hidden_layers': 3, 'no_neurons_in_hidden_layers': len(feature_data['splittable'][0])*2 } #set config params for clearml
    #configuration_dict = task.connect(configuration_dict)

    # Assign loss function constants
    NLL.F = feature_data['splittable']
    NLL.muE = muE
    NLL.mu_sa = mu_sa
    NLL.initD = initD
    NLL.mdp_data = mdp_data

    # Load features
    train_loader = torch.utils.data.DataLoader(feature_data['splittable'], num_workers = 8)
    
    # Define network
    model = LitModel(len(feature_data['splittable'][0]), 'relu', configuration_dict)
    
    # Assign model constants
    model.F = feature_data['splittable']
    model.muE = muE
    model.mu_sa = mu_sa
    model.initD = initD
    model.mdp_data = mdp_data
    model.truep = truep
    model.configuration_dict = configuration_dict
    model.index_states_to_remove = index_states_to_remove

    # Initialise components to train
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration_dict['base_lr'], weight_decay=1e-2)
    swa_model = AveragedModel(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    swa_start = 255
    swa_scheduler = SWALR(optimizer, swa_lr=0.1)

    #Train model
    start_time = time.time()

    loss = 1000 #init loss 
    diff = 1000 #init diff
    threshold = 0.1
    i = 0
    swa_n = 1
    #for epoch in range(configuration_dict['number_of_epochs']):
    while diff >= threshold and i < 765:
        for batch in train_loader:
            prevLoss = loss
            optimizer.zero_grad()
            X = batch
            output = model(X[0,:].view(-1))
            output = output.reshape(len(output), 1)
            loss = NLL.apply(output, model.initD, model.mu_sa, model.muE, model.F, model.mdp_data)
            
            evd = NLL.calculate_EVD(model.truep, torch.matmul(model.F, output))
            loss.backward()
            tensorboard_writer.add_scalar('train_loss',loss,i)
            tensorboard_writer.add_scalar('train_evd',evd,i)
            optimizer.step()
            diff = abs(loss-prevLoss)
            print(i, ':', 'Loss : ', loss.item(), ' EVD: ', evd.item(), 'Diff: ', diff)
            i += 1
            if i > swa_start and (i - swa_start) % 20 == 0:
                moving_average(swa_model, model, 1.0/ (swa_n + 1))
                swa_n += 1
                swa_model.update_parameters(model)
                swa_scheduler.step()
                torch.optim.swa_utils.update_bn(train_loader, swa_model)
            else:
                scheduler.step()


    # Update bn statistics for the swa_model at the end
    torch.optim.swa_utils.update_bn(train_loader, swa_model)
    run_time = (time.time() - start_time)
    print('\n... Finished training models ...\n')
    
    # Create path for trained models
    TRAINED_MODELS_PATH = "./noisey_paths/models/realswag/"
    for path in [TRAINED_MODELS_PATH]:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    # Save swa_model and new features
    torch.save(swa_model.state_dict(), TRAINED_MODELS_PATH+str(worldtype)+'_'+str(dropout_val)+'_'+ str(len(example_samples))+'_NP_model_'+str(index_states_to_remove)+'.pth') 
    tensorboard_writer.close()
    
