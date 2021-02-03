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

import pickle

from torch.utils.tensorboard import SummaryWriter
tensorboard_writer = SummaryWriter('./tensorboard_logs')

torch.set_printoptions(precision=5, sci_mode=False, threshold=100000)
torch.set_default_tensor_type(torch.DoubleTensor)
np.set_printoptions(precision=5, threshold=100000, suppress=False)

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

'''
def run_single_NN(threshold, optim_type, net, NLL, X, initD, mu_sa, muE, F, mdp_data, configuration_dict, truep, NLL_EVD_plots):

    task = Task.init(project_name='MSci-Project', task_name='NNIRL run from main') #init task on ClearML
    configuration_dict = task.connect(configuration_dict)  #enabling configuration override by clearml

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

    if (optim_type == 'Adam'):
        print('\nOptimising with torch.Adam\n')
        optimizer = torch.optim.Adam(
            net.parameters(), lr=configuration_dict.get('base_lr', 0.07500000000000001), weight_decay=1e-2) #weight decay for l2 regularisation
        #while(evd > threshold): #termination criteria: evd threshold
        for p in range(configuration_dict.get('number_of_epochs', 3)): #termination criteria: no of iters
       # while diff >= threshold: #termination criteria: loss diff
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

'''

def ensemble_selector(loss_function, optim_for_loss, y_hats, X, init_size=1,
                      replacement=True, max_iter=100):
    """Implementation of the algorithm of Caruana et al. (2004) 'Ensemble
    Selection from Libraries of Models'. Given a loss function mapping
    predicted and ground truth values to a scalar along with a dictionary of
    models with predicted and ground truth values, constructs an optimal
    ensemble minimizing ensemble loss, by default allowing models to appear
    several times in the ensemble.

    Parameters
    ----------
    loss_function: function
        accepting two arguments - numpy arrays of predictions and true values - 
        and returning a scalar
    y_hats: dict
        with keys being model names and values being numpy arrays of predicted
        values
    init_size: int
        number of models in the initial ensemble, picked by the best loss.
        Default is 1
    replacement: bool
        whether the models should be returned back to the pool of models once
        added to the ensemble. Default is True
    max_iter: int
        number of iterations for selection with replacement to perform. Only
        relevant if 'replacement' is True, otherwise iterations continue until
        the dataset is exhausted i.e.
        min(len(y_hats.keys())-init_size, max_iter). Default is 100

    Returns
    -------
    ensemble_loss: pd.Series
        with loss of the ensemble over iterations
    model_weights: pd.DataFrame
        with model names across columns and ensemble selection iterations
        across rows. Each value is the weight of a model in the ensemble

    """
    # Step 1: compute losses
    losses = dict()
    for model, y_hat in y_hats.items():
        if optim_for_loss:
            losses[model] = loss_function.apply(
                y_hat, initD, mu_sa, muE, F, mdp_data)
        else:
            losses[model] = loss_function.calculate_EVD(truep, torch.matmul(X, y_hat)).item()

    # Get the initial ensemble comprised of the best models
    losses = pd.Series(losses).sort_values()
    init_ensemble = losses.iloc[:init_size].index.tolist()

    # Compute its loss
    if init_size == 1:
        # Take the best loss
        init_loss = losses.loc[init_ensemble].values[0]
        y_hat_avg = y_hats[init_ensemble[0]].detach().clone()
    else:
        # Average the predictions over several models
        y_hat_avg = np.array(
            [y_hats[mod] for mod in init_ensemble]).mean(axis=0)
        if optim_for_loss:
            init_loss = loss_function.apply(
                y_hat, initD, mu_sa, muE, F, mdp_data)
        else:
            init_loss = loss_function.calculate_EVD(truep, torch.matmul(X, y_hat)).item()

    # Define the set of available models
    if replacement:
        available_models = list(y_hats.keys())
    else:
        available_models = losses.index.difference(init_ensemble).tolist()
        # Redefine maximum number of iterations
        max_iter = min(len(available_models), max_iter)

    # Sift through the available models keeping track of the ensemble loss
    # Redefine variables for the clarity of exposition
    current_loss = init_loss
    current_size = init_size

    loss_progress = [current_loss]
    ensemble_members = [init_ensemble]
    for i in range(max_iter):
        # Compute weights for predictions
        w_current = current_size / (current_size + 1)
        w_new = 1 / (current_size + 1)

        # Try all models one by one
        tmp_losses = dict()
        tmp_y_avg = dict()
        for mod in available_models:
            tmp_y_avg[mod] = w_current * y_hat_avg + w_new * y_hats[mod]
            if optim_for_loss:
                tmp_losses[mod] = loss_function.apply(
                    tmp_y_avg[mod], initD, mu_sa, muE, F, mdp_data).item()
            else:
                tmp_losses[mod] = loss_function.calculate_EVD(
                    truep, torch.matmul(X, tmp_y_avg[mod])).item()

        # Locate the best trial
        best_model = pd.Series(tmp_losses).sort_values().index[0]

        # Update the loop variables and record progress
        current_loss = tmp_losses[best_model]
        loss_progress.append(current_loss)
        y_hat_avg = tmp_y_avg[best_model]
        current_size += 1
        ensemble_members.append(ensemble_members[-1] + [best_model])

        if not replacement:
            available_models.remove(best_model)
    # Organize the output
    ensemble_loss = pd.Series(loss_progress, name="loss")
    model_weights = pd.DataFrame(index=ensemble_loss.index,
                                 columns=y_hats.keys())
    for ix, row in model_weights.iterrows():
        weights = pd.Series(ensemble_members[ix]).value_counts()
        weights = weights / weights.sum()
        model_weights.loc[ix, weights.index] = weights

    return ensemble_loss, model_weights.fillna(0).astype(float)

def run_NN_ensemble(models_to_train, max_epochs, iters_per_epoch, learning_rate, X):

    # use ensemble selector to generate ensemble of NN's optimised by min loss or min evd depending on "opitim_for_loss" variable


    models_to_train = models_to_train  # train this many models
    max_epochs = max_epochs  # for this many epochs
    learning_rate = learning_rate

    # Define model names
    model_names = ["M" + str(m) for m in range(models_to_train)]

    # Create paths
    TRAINING_PATH = "./training/"
    for path in [TRAINING_PATH]:
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    # Train a pool of ensemble candidates
    print("... training pool of ensemble candidates ... \n")

    for model_name in model_names:
        # Define the model and optimiser
        net = NonLinearNet(len(X))
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        train_loss = []
        train_evd = []

        for epoch in range(max_epochs):

            # Training loop
            net.train()
            epoch_loss = []
            epoch_evd = []
            for i in range(iters_per_epoch):
                # Compute predicted R
                yhat = torch.empty(len(X[0]), 1)
                indexer = 0
                for f in range(len(X[0])):
                    this_state_r = net(X[:, f].view(-1, len(X[:, f])))
                    yhat[indexer] = this_state_r
                    indexer += 1
                
                
                 

                # compute loss and EVD
                loss = NLL.apply(yhat, initD, mu_sa, muE, F, mdp_data)
                evd = NLL.calculate_EVD(truep, torch.matmul(X, yhat))

                print('{} | EVD: {} | loss: {} '.format(
                    i, evd, loss.detach().numpy()))

                # Backpropogate and update weights
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), clip_value=3)
                optimizer.step()

                # Append loss and EVD estimates
                epoch_loss.append(loss.item())
                epoch_evd.append(evd.item())

            # Compute metrics for this epoch
            train_loss.append(np.array(epoch_loss).mean())
            train_evd.append(np.array(epoch_evd).mean())

            print("\nModel Name", model_name, "Epoch", epoch,
                  "Loss", train_loss[-1], "EVD", train_evd[-1], "\n")

            # Save the checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": pd.DataFrame({"train_loss": train_loss, "train_evd": train_evd}).astype(float),
                "train_loss": train_loss[-1],
                "train_evd": train_evd[-1]
            }, TRAINING_PATH + model_name + "_epoch_" + str(epoch) + ".p")

    # print(os.listdir(TRAINING_PATH))

    print("\n... done ...\n")

    # For each model pick the checkpoint with the lowest validation loss, then:
    # 1. compute losses and accuracies on the validation and test set
    # 2. get predictions on the validation set
    trained_models = {}
    metrics = {}
    y_hats_test = {}

    x_test = X  # features for test

    for model_name in model_names:
        # Load the last checkpoint
        last_checkpoint = torch.load(
            TRAINING_PATH + model_name + "_epoch_" + str(max_epochs-1) + ".p")

        # Find the best checkpoint by train loss
        best_by_train_loss = last_checkpoint["history"].sort_values(
            "train_loss").index[0]
        best_checkpoint = torch.load(
            TRAINING_PATH + model_name + "_epoch_" + str(best_by_train_loss) + ".p")

        # Restore the best checkpoint
        net = NonLinearNet()
        net.load_state_dict(best_checkpoint["model_state_dict"])
        net.eval()

        # Compute predictions on the validation and test sets, compute the
        # metrics for the latter (validation stuff has already been saved)
        # Compute predicted R
        y_hat_test = torch.empty(len(X[0]), 1)
        indexer = 0
        for j in range(len(x_test[0])):
            this_state_r = net(x_test[:, j].view(-1, len(x_test[:, j])))
            y_hat_test[indexer] = this_state_r
            indexer += 1
       

        # compute evd & loss
        test_loss = NLL.apply(y_hat_test, initD, mu_sa,
                              muE, F, mdp_data).item()
        test_evd = NLL.calculate_EVD(truep, torch.matmul(X, y_hat_test))

        # Store the outputs
        trained_models[model_name] = net
        metrics[model_name] = {
            "train_loss": best_checkpoint["train_loss"],
            "train_evd": best_checkpoint["train_evd"],
            "test_loss": test_loss,
            "test_evd": test_evd}

        # Store models loss in dict
        y_hats_test[model_name] = y_hat_test

    # Convert the metrics dict to a dataframe
    metrics = pd.DataFrame(metrics).T.astype(float)
    print(metrics)

    # Separate dataframes for losses and accuracies
    metrics_loss = metrics.filter(like="loss").stack().reset_index()
    metrics_loss.columns = ["model", "train/test", "loss"]

    metrics_evd = metrics.filter(like="evd").stack().reset_index()
    metrics_evd.columns = ["model", "train/test", "evd"]

    # Plot losses and accuracies
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    sns.barplot(x="model", y="loss", hue="train/test", data=metrics_loss,
                alpha=0.75, saturation=0.90, palette=["#1f77b4", "#ff7f0e"],
                ax=ax[0])
    sns.barplot(x="model", y="evd", hue="train/test", data=metrics_evd,
                alpha=0.75, saturation=0.90, palette=["#1f77b4", "#ff7f0e"],
                ax=ax[1])

    ax[0].set_ylim(metrics_loss["loss"].min() - 1e-2,
                   metrics_loss["loss"].max() + 1e-2)
    ax[1].set_ylim(metrics_evd["evd"].min()-3e-3,
                   metrics_evd["evd"].max()+3e-3)

    ax[0].set_title("Loss", fontsize=17)
    ax[1].set_title("Expected Value Difference", fontsize=17)

    for x in ax:
        x.xaxis.set_tick_params(rotation=0, labelsize=15)
        x.yaxis.set_tick_params(rotation=0, labelsize=15)
        x.set_xlabel("Model", visible=True, fontsize=15)
        x.set_ylabel("", visible=False)

        handles, labels = x.get_legend_handles_labels()
        x.legend(handles=handles, labels=labels, fontsize=15)

    fig.tight_layout(w_pad=5)

    ensemble_loss, model_weights = ensemble_selector(
        loss_function=NLL, optim_for_loss=True, y_hats=y_hats_test, X=X, init_size=1, replacement=True, max_iter=10)

    print("\nEnsemble Loss:")
    print(ensemble_loss)
    print("Ensemble Model Weight:")
    print(model_weights)

    # Locate non-zero weights and sort models by their average weight
    weights_to_plot = model_weights.loc[:, (model_weights != 0).any()]
    weights_to_plot = weights_to_plot[
        weights_to_plot.mean().sort_values(ascending=False).index]

    # A palette corresponding to the number of models with non-zero weights
    palette = sns.cubehelix_palette(weights_to_plot.shape[1], reverse=True)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    weights_to_plot.plot(kind="bar", stacked=True, color=palette, ax=ax,
                         alpha=0.85)
    ax.margins(x=0.0)
    ax.set_xlabel("Optimization Step", fontsize=15, visible=True)
    ax.set_ylabel("Ensemble Weight", fontsize=15, visible=True)
    ax.yaxis.set_tick_params(rotation=0, labelsize=15)
    ax.xaxis.set_tick_params(rotation=0, labelsize=15)
    ax.legend(loc="best", bbox_to_anchor=(1, 0.92),
              frameon=True, edgecolor="k", fancybox=False,
              framealpha=0.7, shadow=False, ncol=1, fontsize=15)
    fig.tight_layout()

    # Compute the test loss for each ensemble iteration
    ensemble_loss_test = []
    for _, row in model_weights.iterrows():
        # Compute test prediction for this iteration of ensemble weights
        tmp_y_hat = np.array(
            [y_hats_test[model_name] * weight
                for model_name, weight in row.items()]
        ).sum(axis=0)

        ensemble_loss_test.append(
            NLL.apply(tmp_y_hat, initD, mu_sa, muE, F, mdp_data).item())
    ensemble_loss_test = pd.Series(ensemble_loss_test)

    # Compute loss of an ensemble which equally weights each model in the pool
    losses = []
    for model, predictedR in y_hats_test.items():
        losses.append(NLL.apply(predictedR, initD, mu_sa, muE, F, mdp_data))
    ens_loss_test_avg = sum(losses) / len(losses)
    ens_loss_test_avg = ens_loss_test_avg.item()

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 7), sharey=False)

    ax.plot(ensemble_loss_test, color="#1f77b4", lw=2.75,
            label="ensemble loss")
    ax.plot(pd.Series(ensemble_loss_test[0], ensemble_loss_test.index),
            color="k", lw=1.75, ls="--", dashes=(5, 5),
            label="baseline 1: best model on validation set")
    ax.plot(pd.Series(ens_loss_test_avg, ensemble_loss.index),
            color="r", lw=1.75, ls="--", dashes=(5, 5),
            label="baseline 2: average of all models")
    ax.set_title("Test Loss", fontsize=17)

    ax.margins(x=0.0)
    ax.set_xlabel("Optimization Step", fontsize=15, visible=True)
    ax.set_ylabel("", fontsize=15, visible=False)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.92),
              frameon=True, edgecolor="k", fancybox=False,
              framealpha=0.7, shadow=False, ncol=1, fontsize=15)
    fig.tight_layout(w_pad=3.14)

    # EVD-minimising ensemble on the test set
    ensemble_acc, model_weights = ensemble_selector(
        loss_function=NLL, optim_for_loss=False, y_hats=y_hats_test, X=X, init_size=1, replacement=True, max_iter=10)

    # Compute evd of the equally weighted ensemble
    evds = []
    for model, predictedR in y_hats_test.items():
        evds.append(NLL.calculate_EVD(truep, torch.matmul(X, predictedR)))
    ens_acc_test_avg = sum(evds) / len(evds)
    ens_acc_test_avg = ens_acc_test_avg.item()

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 7), sharey=False)

    ax.plot(ensemble_acc, color="#1f77b4", lw=2.75, label="ensemble EVD")
    ax.plot(pd.Series(ensemble_acc[0], ensemble_acc.index), color="k",
            lw=1.75, ls="--", dashes=(5, 5), label="baseline 1: best model")
    ax.plot(pd.Series(ens_acc_test_avg, ensemble_loss.index), color="r", lw=1.75,
            ls="--", dashes=(5, 5), label="baseline 2: average of all models")
    ax.set_title("Test EVD", fontsize=17)

    ax.margins(x=0.0)
    ax.set_xlabel("Optimization Step", fontsize=15, visible=True)
    ax.set_ylabel("", fontsize=15, visible=False)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.72),
              frameon=True, edgecolor="k", fancybox=False,
              framealpha=0.7, shadow=False, ncol=1, fontsize=15)
    fig.tight_layout()

    plt.show()

    return y_hats_test, model_weights

def run_single_NN():

    task = Task.init(project_name='MSci-Project', task_name='NNIRL Run') #init task on ClearML
   
    #load variables from file
    open_file = open("NNIRL_param_list.pkl", "rb")
    NNIRL_param_list = pickle.load(open_file)
    open_file.close()
    threshold = NNIRL_param_list[0]
    optim_type = NNIRL_param_list[1]
    net = NNIRL_param_list[2]
    X = NNIRL_param_list[3]
    initD = NNIRL_param_list[4]
    mu_sa = NNIRL_param_list[5]
    muE = NNIRL_param_list[6]
    F = NNIRL_param_list[7]
    #F = F.type(torch.DoubleTensor)
    mdp_data = NNIRL_param_list[8]
    configuration_dict = NNIRL_param_list[9]
    truep = NNIRL_param_list[10] 
    NLL_EVD_plots = NNIRL_param_list[11]
    NLL = NLLFunction()  # initialise NLL
    #assign constants
    NLL.F = F
    NLL.muE = muE
    NLL.mu_sa = mu_sa
    NLL.initD = initD
    NLL.mdp_data = mdp_data

    configuration_dict = task.connect(configuration_dict)  #enabling configuration override by clearml

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

    if (optim_type == 'Adam'):
        print('\nOptimising with torch.Adam\n')
        optimizer = torch.optim.Adam(
            net.parameters(), lr=configuration_dict.get('base_lr', 0.07500000000000001), weight_decay=1e-2) #weight decay for l2 regularisation
        #while(evd > threshold): #termination criteria: evd threshold
        #for p in range(configuration_dict.get('number_of_epochs', 3)): #termination criteria: no of iters in config dict
        while diff >= threshold: #termination criteria: loss diff
            prevLoss = loss
            net.zero_grad()
            
            output = torch.empty(len(X[0]), 1, dtype=torch.double)
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


run_single_NN()