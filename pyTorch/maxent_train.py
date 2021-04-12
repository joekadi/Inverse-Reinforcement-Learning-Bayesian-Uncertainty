
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
from scipy.optimize import minimize, check_grad
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


class CustomLH:   

    mu_sa = None
    F = None
    mdp_data = None
    muE = None
    initD = None

    def NLL_with_grad(self, fW):
        #Returns NLL w.r.t input r

        if(torch.is_tensor(fW) == False):
            lh_r = torch.tensor(fW) #cast to tensor
        if(lh_r.shape != (self.mdp_data['states'],5)):
            #convert to full reward
            r = torch.matmul(self.F, lh_r)
            r = torch.reshape(r, (len(r), 1))
            r = r.repeat((1, 5))

        if r.shape != (int(mdp_data['states']),5):
            print(r.shape)
            raise Exception("Reward shape not (states, 5)")

        #Solve MDP with current reward
        v, q, logp, p = linearvalueiteration(self.mdp_data, r) 
   
        #Calculate likelihood from logp
        likelihood = torch.empty(self.mu_sa.shape, requires_grad=True)
        likelihood = torch.sum(torch.sum(logp*self.mu_sa))

        #Calculate gradient
        
        #Calc gradient w.r.t to forward inputs 
        D = torch.from_numpy(fW)
        D = linearmdpfrequency(self.mdp_data,p,self.initD) #Compute state visitation count D
        D = D.clone().detach().requires_grad_(True) #cast to tensor

        
        dr = self.muE - torch.matmul(torch.t(self.F),D) #Compute gradient
      
        return -likelihood.detach().numpy(), -dr.detach().numpy()


    def calculate_EVD(self, trueP, currR):

        if(currR.shape != (len(currR),5)):
            currR = currR.repeat((1, 5))

        if currR.shape != trueP.shape:
            raise Exception("Reward shapenot (states, 5) instead it's", + str(currR.shape))

        v, q, logp, currP = linearvalueiteration(self.mdp_data, currR)
        #Expected Value Diff = diff in policies since exact True R values never actually learned, only it's structure
        evd=torch.max(torch.abs(currP-trueP))
        return evd


if __name__ == "__main__":

    if len(sys.argv) > 1:
        num_paths = int(str(sys.argv[1]))
        print('\n... got number of paths value from cmd line ...\n')
    else:
        raise Exception("Number of Paths not supplied")

    #task = Task.init(project_name='MSci-Project', task_name='Train - Noisey paths') #Initalise task on clearML
    
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
        
        
    lh = CustomLH()
    lh.mu_sa = mu_sa
    lh.mdp_data = mdp_data
    lh.F = feature_data['splittable']
    lh.muE = muE
    lh.initD = initD

    # initial estimated feature weight vector
    fW = torch.randn(len(feature_data['splittable'][0]), 1)  

    #minimise likelihood
    res = minimize(lh.NLL_with_grad, fW, jac=True, method="L-BFGS-B", options={
                'disp': True, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})

    r = res.x #assign feature weight vector 


    #Get full reward
    if(torch.is_tensor(r) == False):
            r = torch.tensor(r) #cast to tensor
    if(r.shape != (mdp_data['states'],5)):
        #convert to full reward
        r = torch.matmul(feature_data['splittable'], r)
        r = torch.reshape(r, (len(r), 1))
        r = r.repeat((1, 5))

    # Create path to store results
    RESULTS_PATH = "./regular/results/maxent/"
    for path in [RESULTS_PATH]:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    results = [r, lh.calculate_EVD(truep, r)]
    file_name = RESULTS_PATH+str(worldtype)+'_'+str(num_paths)+'_results.pkl'
    open_file = open(file_name, "wb")
    pickle.dump(results, open_file)
    open_file.close()