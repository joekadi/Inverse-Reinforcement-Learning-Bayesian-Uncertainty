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


    def __init__(self, no_features, configuration_dict):
        super().__init__()
        self.layer_1 = nn.Linear(no_features, configuration_dict['i2'])
        self.layer_2 = nn.Linear(configuration_dict['i2'], configuration_dict['h1_out'])
        self.layer_3 = nn.Linear(configuration_dict['h1_out'], configuration_dict['h2_out'])
        self.layer_4 = nn.Linear(configuration_dict['h2_out'], no_features)

    def forward(self, x):
        x  = Functional.relu(self.layer_1(x))
        x = Functional.relu(self.layer_2(x))
        x = Functional.relu(self.layer_3(x))
        x = self.layer_4(x)

        return x

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
        self.learned_feature_weights = output #store current
        self.log('train_loss', loss)
        self.log('train_evd', evd)
        return loss


if __name__ == "__main__":

    if len(sys.argv) > 1:
        worldtype = str(sys.argv[1]) #benchmark type curr only gw or ow
        user_input = True
    else:
        user_input = False

      
    task = Task.init(project_name='MSci-Project', task_name='LitModel Run, n=8, b=1, normal') #init task on ClearML
    configuration_dict = {'number_of_epochs': 1, 'base_lr': 0.1, 'i2': 32, 'h1_out': 32, 'h2_out': 32} #set config params for clearml
    configuration_dict = task.connect(configuration_dict)
    
    #load variables from file
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
    noisey_features = NNIRL_param_list[10] 
    mdp_params = NNIRL_param_list[11] 
    r = NNIRL_param_list[12] 
    mdp_solution = NNIRL_param_list[13] 
    feature_data = NNIRL_param_list[14] 
    trueNLL = NNIRL_param_list[15]
    NLL = NLLFunction()  # initialise NLL
    #assign constants
    NLL.F = feature_data['splittable']
    NLL.muE = muE
    NLL.mu_sa = mu_sa
    NLL.initD = initD
    NLL.mdp_data = mdp_data

    train_loader = torch.utils.data.DataLoader(feature_data['splittable'], num_workers = 8)
    trainer = pl.Trainer(max_epochs=configuration_dict['number_of_epochs'])

    model = LitModel(len(feature_data['splittable'][0]), configuration_dict)

    model.NLL = NLL
    model.F = feature_data['splittable']
    model.muE = muE
    model.mu_sa = mu_sa
    model.initD = initD
    model.mdp_data = mdp_data
    model.truep = truep
    model.configuration_dict = configuration_dict
    print("\nTrue R has:\n - negated likelihood: {}\n - EVD: {}".format(trueNLL,  NLL.calculate_EVD(truep, r)))


    start_time = time.time()
    trainer.fit(model, train_loader)
    run_time = (time.time() - start_time)

    #calculate, format and print results
    if(model.learned_feature_weights.shape != (mdp_data['states'],5)):
        #convert to full reward
        predictedR = torch.matmul(feature_data['splittable'], model.learned_feature_weights)
        predictedR = predictedR.repeat((1, 5))

    predictedv, predictedq, predictedlogp, predictedP = linearvalueiteration(mdp_data, predictedR)
    print("\nPredicted R has:\n - negated likelihood: {}\n - EVD: {}".format(NLL.apply(predictedR, initD, mu_sa, muE, F, mdp_data), NLL.calculate_EVD(truep, predictedR )))

    irl_result = { #models IRL results
        'r': predictedR,
        'v': predictedv,
        'p': predictedP,
        'q': predictedq,
        'r_itr': [predictedR],
        'model_itr': [model.learned_feature_weights],
        'model_r_itr': [predictedR],
        'p_itr': [predictedP],
        'model_p_itr':[predictedP],
        'time': run_time
    }

    test_result = { #ground truth metrics
        'irl_result': irl_result,
        'true_r': r,
        'example_samples': [example_samples],
        'mdp_data': mdp_data,
        'mdp_params': mdp_params,
        'mdp_solution': mdp_solution,
        'feature_data': feature_data
    }

    #call respective draw method
    if(user_input):
        if worldtype == "gridworld" or worldtype == "gw" or worldtype == "grid":
            gwVisualise(test_result)
        elif worldtype == "objectworld" or worldtype == "ow" or worldtype == "obj":
            owvisualise(test_result)
    else:
        gwVisualise(test_result)


    PATH = './NN_IRL.pth'
    torch.save(net.state_dict(), PATH)
    tensorboard_writer.close()

    #return model, model.learned_feature_weights, (time.time() - start_time)


