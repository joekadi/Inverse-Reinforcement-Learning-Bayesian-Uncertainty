import torch 
import torch.nn as nn
from torch.nn.functional import softplus
import torch.nn.functional as F
from nonlinearBNN import *
from nonlinearnn import *
from linearnn import *
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import gradcheck
from torch.autograd import Variable
import torch
from torch.utils.data import random_split
from scipy.optimize import minimize, check_grad
import os
import sys
from NLLFunction import *
from myNLL import *
from likelihood import *
from gridworld import *
from objectworld import *
from linearvalueiteration import *
import pprint
from sampleexamples import *
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import math as math
import copy 
import torchvision
import torchvision.transforms as transforms
from clearml import Task

from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

from torch.utils.tensorboard import SummaryWriter
tensorboard_writer = SummaryWriter('./tensorboard_logs')

def likelihood(r, initD, mu_sa, muE, F, mdp_data):
    #Returns NLL w.r.t input r

    '''
    if(torch.is_tensor(r) == False):
        r = torch.tensor(r) #cast to tensor
    if(r.shape != (mdp_data['states'],5)):
        #reformat to be in shape (states,actions)
        r = torch.reshape(r, (int(mdp_data['states']),1))
        r = r.repeat((1, 5))
    '''

    if(torch.is_tensor(r) == False):
        r = torch.tensor(r) #cast to tensor
    if(r.shape != (mdp_data['states'],5)):
        #convert to full reward
        r = torch.matmul(F, r)


    #Solve MDP with current reward
    v, q, logp, p = linearvalueiteration(mdp_data, r) 

    #Calculate likelihood from logp
    likelihood = torch.empty(mu_sa.shape, requires_grad=True)


    likelihood = torch.sum(torch.sum(logp*mu_sa)) #for scalar likelihood

    #LH for each state as tensor size (states,1)
    #mul = logp*mu_sa #hold
    #likelihood = torch.sum(mul, dim=1)
    #likelihood.requires_grad = True
    
    
    return -likelihood

def run_single_NN(threshold, optim_type, net, NLL, X, initD, mu_sa, muE, F, mdp_data, configuration_dict, truep, NLL_EVD_plots):
    
    task = Task.init(project_name='MSci-Project', task_name='run_single_NN') #init task on ClearML

    start_time = time.time() #to time execution
    #tester = testers() #to use testing functions

    # lists for printing
    NLList = []
    iterations = []
    evdList = []

    i = 0 #track iterations
    finalOutput = None #store final est R
    loss = 1000 #init loss 
    diff = 1000 #init diff
    evd = 10 #init val
    configuration_dict = task.connect(configuration_dict)  #enabling configuration override by clearml

    if (optim_type == 'Adam'):
        print('\nOptimising with torch.Adam\n')
        optimizer = torch.optim.Adam(
            net.parameters(), lr=configuration_dict.get('base_lr', 0.1), weight_decay=1e-2) #weight decay for l2 regularisation
        #while(evd > threshold): #termination criteria: evd threshold
        for p in range(configuration_dict.get('number_of_epochs', 3)): #termination criteria: no of iters
        #while diff >= threshold: #termination criteria: loss diff
            prevLoss = loss
            net.zero_grad()
            
            output = torch.empty(len(X[0]), 1)
            indexer = 0
            for j in range(len(X[0])):
                thisR = net(X[:, j].view(-1, len(X[:, j])))
                output[indexer] = thisR
                indexer += 1
            finalOutput = output

            loss = NLL.apply(output, initD, mu_sa, muE, F, mdp_data) #use this line for custom gradient
            #loss = likelihood(output, initD, mu_sa, muE, F, mdp_data) #use this line for auto gradient
            #tester.checkgradients_NN(output, NLL) # check gradients
            loss.backward()  # propagate grad through network
            #nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
            evd = NLL.calculate_EVD(truep, torch.matmul(X, output))  # calc EVD
            optimizer.step()

            #printline to show est R
            #print('{}: output:\n {} | EVD: {} | loss: {} '.format(i, torch.matmul(X, output).repeat(1, 5).detach().numpy(), evd, loss.detach().numpy()))

            #printline to hide est R
            print('{}: | EVD: {} | loss: {} | diff {}'.format(i, evd, loss.detach().numpy(), diff))
            # store metrics for printing
            NLList.append(loss.item())
            iterations.append(i)
            evdList.append(evd.item())
            finaloutput = output
            tensorboard_writer.add_scalar('loss', loss.detach().numpy(), i)
            tensorboard_writer.add_scalar('evd', evd, i)
            tensorboard_writer.add_scalar('diff', diff, i)

            i += 1
            diff = abs(prevLoss-loss)

    else:
        print('\implement LBFGS\n')
        
        
    PATH = './NN_IRL.pth'
    torch.save(net.state_dict(), PATH)
    tensorboard_writer.close()

    if NLL_EVD_plots:
        # plot
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        ax1.plot(iterations, NLList)
        ax1.plot(iterations, NLList, 'r+')
        ax1.set_title('NLL')

        ax2.plot(iterations, evdList)
        ax2.plot(iterations, evdList, 'r+')
        ax2.set_title('Expected Value Diff')
        plt.show()
    
    
    print("\nruntime: --- %s seconds ---\n" % (time.time() - start_time) )
    return net, finalOutput, (time.time() - start_time)