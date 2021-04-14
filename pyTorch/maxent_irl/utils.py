import torch
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import math
import os

torch.set_printoptions(precision=5, sci_mode=False, threshold=1000)
torch.set_default_tensor_type(torch.DoubleTensor)

def rmv(stds):
    """
    #Computes the root mean variance
    :param stds: A list of stds 
    :return: The computed rmv
    """


    return math.sqrt(abs(np.sum(stds))/len(stds))

def rmse(targets, preds):
    """
    Computes the root mean squared error.
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return math.sqrt(mean_squared_error(targets, preds))

def se(targets, preds):
    """
    Computes the squared error.
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed se.
    """

    se = (targets-preds)**2

    return torch.mean(se,1)

def ence(targets, preds, stds, n_bins=256):

    """
    #Computes the expected Normalized Calibration Error (ENCE)
    #Proposed in https://arxiv.org/pdf/1905.11659.pdf
    :param targets: A list of targets
    :param preds: A list of predictions
    :param stds: A list of stds for each prediciton
    :return: The computed ence
    """

    #Conver to numpy
    targets = targets.detach().numpy()
    preds = preds.detach().numpy()

    #Generate bin indexes for slicing
    bin_indexes = np.arange(0, len(preds)+len(preds)/n_bins, len(preds)/n_bins)

    rmses = []
    rmvs = []

    #Calculate RMSE and RMV for each bin
    start_indx = 0
    stop_indx = 1
    while(stop_indx != len(bin_indexes)):
        rmses.append(rmse(targets[int(math.ceil(bin_indexes[start_indx])):int(math.ceil(bin_indexes[stop_indx]))], preds[int(math.ceil(bin_indexes[start_indx])):int(math.ceil(bin_indexes[stop_indx]))]))
        rmvs.append(rmv(stds[int(math.ceil(bin_indexes[start_indx])):int(math.ceil(bin_indexes[stop_indx]))]))
        start_indx +=1
        stop_indx +=1

    curr_rmse = np.sum(rmses)
    curr_rmv = np.sum(rmvs)

    if curr_rmv != 0:
        return (abs(curr_rmv-curr_rmse)/curr_rmv)/n_bins*10000 #scale up for readability
    else:
        raise Exception('Root Mean Variance = 0')

def plot_ence(targets, preds, stds, n_bins=256):

    """
    #Plot variance along X
    #Plot respective MSE along Y
    #Display ENCE on graph
    """

    #Convert to numpy
    targets = targets.detach().numpy()
    preds = preds.detach().numpy()
    #Generate bin indexes for slicing
    bin_indexes = np.arange(0, len(preds)+len(preds)/n_bins, len(preds)/n_bins)
    rmses = []
    rmvs = []
    #Calculate RMSE and RMV for each bin
    start_indx = 0
    stop_indx = 1
    while(stop_indx != len(bin_indexes)):
        rmses.append(rmse(targets[int(math.ceil(bin_indexes[start_indx])):int(math.ceil(bin_indexes[stop_indx]))], preds[int(math.ceil(bin_indexes[start_indx])):int(math.ceil(bin_indexes[stop_indx]))]))
        rmvs.append(rmv(stds[int(math.ceil(bin_indexes[start_indx])):int(math.ceil(bin_indexes[stop_indx]))]))
        start_indx +=1
        stop_indx +=1
        
    rmses = np.array(rmses)
    rmvs = np.array(rmvs)
    fig, ax = plt.subplots(1, figsize=(2,2))

    ax.plot(rmses, rmvs, label='Actual') #Line
    ax.plot([0,1], [0,1], 'k--', label='Goal', alpha=0.5)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    return fig, ax

def calibration_curve(targets, preds, stds, n_bins=256):
   
    #Convert to numpy
    targets = targets.detach().numpy()
    preds = preds.detach().numpy()
    #Generate bin indexes for slicing
    bin_indexes = np.arange(0, len(preds)+len(preds)/n_bins, len(preds)/n_bins)
    confidences_inbins = []
    accuracies_inbins = [] 
    start_indx = 0
    stop_indx = 1
    while(stop_indx != len(bin_indexes)):
        accuracies_inbins.append(evda(targets[int(math.ceil(bin_indexes[start_indx])):int(math.ceil(bin_indexes[stop_indx]))], preds[int(math.ceil(bin_indexes[start_indx])):int(math.ceil(bin_indexes[stop_indx]))]))
        confidences_inbins.append(rmv(stds[int(math.ceil(bin_indexes[start_indx])):int(math.ceil(bin_indexes[stop_indx]))]))
        start_indx +=1
        stop_indx +=1
    accuracies_inbins = np.sort(accuracies_inbins, axis=None)
    confidences_inbins = np.sort(confidences_inbins, axis=None)
    fig, ax = plt.subplots(1, figsize=(2,2))
    ax.plot(accuracies_inbins, confidences_inbins, label='Actual') #Line
    ax.plot([0,1], [0,1], 'k--', label='Goal', alpha=0.5) #Goal
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    fig.tight_layout()
    confidences = np.sum(confidences_inbins)
    accuracies = np.sum(accuracies_inbins)
    return fig, ax, (abs(confidences-accuracies)/confidences)/n_bins*10000

def evda(trueP, currP):
    trueP = torch.tensor(trueP)
    currP = torch.tensor(currP)
    #Expected Value Diff = diff in policies since exact True R values never actually learned, only it's structure
    evd=torch.max(torch.abs(currP-trueP))
    return evd

def epdue(uncertainty, epd):

    """
    #Computes Expected Policy Difference Uncertainty Error
    :param uncertainty: A list of variances
    :param epd: A list of expected policy differences per pred
    :return: the computed epdue
    """

    return np.mean(np.abs(epd-uncertainty))  

def plot_epdue(uncertainty, epd):
    fig, ax = plt.subplots(1, figsize=(3,3.5))
    

    ax.plot(uncertainty, epd, 'ro') #Dots
    ax.plot(uncertainty, epd, label='Actual') #Line
    ax.plot([0,1], [0,1], 'k--', label='Goal') #Goal

    #ax.scatter(np.arange(1,257,1), uncertainty, label='Uncertainty', s=10)
    #ax.scatter(np.arange(1,257,1), epd, label='EVD', s=10)

    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('EVD')
    #ax.legend(loc='center right', prop={'size': 6})

    return fig, ax
