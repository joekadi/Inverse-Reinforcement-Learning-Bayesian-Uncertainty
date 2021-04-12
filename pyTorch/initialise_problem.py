'''
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions import Gamma, Poisson, Normal, Binomial
import pyro.distributions as dist
import pyro
'''
import torch 
import torch.nn as nn
from torch.nn.functional import softplus
import torch.nn.functional as F
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
from gridworld import *
from objectworld import *
from linearvalueiteration import *
from NNIRL import *
import pprint
from sampleexamples import *
from train import *
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import math as math
import copy 
import torchvision
import torchvision.transforms as transforms
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from torch.utils.tensorboard import SummaryWriter
import pickle

tensorboard_writer = SummaryWriter('./tensorboard_logs')
print('\n\ntorch version:', torch.__version__)
torch.set_printoptions(precision=5, sci_mode=False, threshold=1000)
torch.set_default_tensor_type(torch.DoubleTensor)
np.set_printoptions(precision=5, threshold=1000, suppress=False)

class testers:

    def checkgradients_NN(self, input, linear):
        # gradcheck takes a tuple of tensors as input, check if your gradient
        # evaluated with these tensors are close enough to numerical
        # approximations and returns True if they all verify this condition.
        test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
        print('Gradient check; {}'.format(test))

    def checkgradients(self, lh, mdp_params, k):
        # checking gradient for k reward function points

        rewardsToCheck = []
        for i in range(k):
            checkR = np.random.randn(
                mdp_params['n']**2, 1)  # initial estimated R
            rewardsToCheck.append(checkR)

        print("... checking gradient ...\n")

        total_rmse = []
        true_gradients = []
        expected_gradients = []

        for reward in rewardsToCheck:
            rmse, true_grad, expected_grad = check_grad(
                lh.negated_likelihood, lh.calc_gradient, [reward], epsilon=1e-4)
            total_rmse.append(rmse)
            true_gradients.append(true_grad.item())
            expected_gradients.append(expected_grad)

        # print and plot RMSE
        print('\nGradient check terminated with a RMSE of {} from {} gradient checks\n'.format(
            sum(total_rmse)/len(total_rmse), len(rewardsToCheck)))
        plt.plot(true_gradients, expected_gradients, 'bo')
        plt.title('check_grad() gradient comparison')
        plt.xlabel('True Gradients')
        plt.ylabel('Expected Gradients')
        plt.grid(b='True', which='minor')
        plt.show()

    def test_gradient(self, lh, testr):
        print('Gradient for test r is \n{}'.format(lh.calc_gradient(testr)))
        return(lh.calc_gradient(testr))

    def test_likelihood(self, lh, testr):
        print('Likelihood for test r is {}'.format(
            lh.negated_likelihood(testr)))
        return(lh.negated_likelihood(testr))

    def compare_gandLH_with_matlab(self, lh):
        torchG = self.est_gradient(lh, testr)
        torchL = self.test_likelihood(lh, testr)

        testr = np.array(
            [[5.11952e+01],
             [2.17734e+05],
                [1.01630e+0],
                [1.44944e-07]])
        matlabG = np.array([[-0227.937600000000],
                            [8139.016753098902],
                            [-3837.240000000000],
                            [-4073.850000000000]])

        matlabL = 1.772136688141655e+09

        print('Elementwise diff torch gradient - matlab gradient is \n {}'.format(
            np.subtract(torchG.detach().cpu().numpy(), matlabG)))
        print('Likelihood diff is {}'.format(torchL - matlabL))

    def linearNN(self, evdThreshold, optim_type):
        net = LinearNet()
        tester = testers()

        # initialise rewards by finding true weights for NN. feed features through NN using true Weights to get ground truth reward.

        # initalise with some noise? can we still uncover sensible reward

        # put an l2 regulisariton weight decay on the network weights. fine tune the lambda value
        #  bias = false on weight params seems to work when inital R is 0

        # check gradients with torch.gradcheck

        X = torch.Tensor([[0, 0],
                          [1, 0],
                          [2, 0],
                          [3, 0]])  # for NN(state feature vector) = reward

        '''
		X = torch.Tensor([[0],
				  [1],
				  [2],
				  [3]]) #for (4,4) NN
		'''

        evd = 10
        lr = 0.1
        finaloutput = None
        # lists for printing
        NLList = []
        iterations = []
        evdList = []
        i = 0

        if (optim_type == 'Adam'):
            print('\nOptimising with torch.Adam\n')
            # inital adam optimiser, weight decay for l2 regularisation
            optimizer = torch.optim.Adam(
                net.parameters(), lr=lr, weight_decay=1e-2)
            while(evd > evdThreshold):
                net.zero_grad()

                # build output vector as reward for each state w.r.t its features
                output = torch.empty(len(X))
                indexer = 0
                for f in X:
                    thisR = net(f.view(-1, len(f)))
                    output[indexer] = thisR
                    indexer += 1

                # get loss from curr output
                loss = NLL.apply(output, initD, mu_sa, muE, F, mdp_data)

                # check gradients
                #tester.checkgradients_NN(output, NLL)

                #print('Output {} with grad fn {}'.format(output, output.grad_fn))
                #print('Loss {} with grad fn {}'.format(loss, loss.grad_fn))

                loss.backward()  # propagate grad through network
                evd = NLL.calculate_EVD(truep, output)  # calc EVD
                '''
				j = 1
				for p in net.parameters():
					print('Gradient of parameter {} with shape {} is {}'.format(j, p.shape, p.grad))
					j +=1
				j = 0
				'''

                optimizer.step()

                # Printline when LH is vector
                #print('{}: output: {} | EVD: {} | loss: {} | {}'.format(i, output.detach().numpy(), evd,loss.detach().numpy(), sum(loss).detach().numpy()))
                # Printline when LH scalar
                print('{}: output: {} | EVD: {} | loss: {} '.format(
                    i, output.detach().numpy(), evd, loss.detach().numpy()))

                # store metrics for printing
                NLList.append(loss.item())
                iterations.append(i)
                evdList.append(evd.item())
                finaloutput = output
                i += 1
        else:
            print('\nOptimising with torch.LBFGS\n')
            optimizer = torch.optim.LBFGS(net.parameters(), lr=lr)

            def closure():
                net.zero_grad()
                output = net(X.view(-1, 4))  # when NLL layer is (4,4)
                loss = NLL.negated_likelihood(output)
                loss = sum(loss)
                evd = NLL.calculate_EVD(truep)
                print('{}: output: {} | EVD: {} | loss: {}'.format(
                    i, output.detach().numpy(), evd, loss.detach().numpy()))
                current_gradient = NLL.calc_gradient(output)
                #print('Current gradient \n{}'.format(current_gradient))

                #net.fc1.weight.grad = current_gradient.repeat(1,4)
                # much worse than above
                loss.backward(gradient=torch.argmax(current_gradient))
                '''												 
				print('Calculated grad \n {}'.format(current_gradient))
				j = 1
				for p in net.parameters():
					print('Gradient of parameter {} \n {}'.format(j, p.grad))
					j +=1
				j = 0
				'''

                # store metrics for printing
                NLList.append(sum(loss).item())
                iterations.append(i)
                evdList.append(evd.item())
                finaloutput = output
                return loss  # .max().detach().numpy()
            for i in range(500):
                optimizer.step(closure)

        # Normalise data
        #NLList = [float(i)/sum(NLList) for i in NLList]
        #evdList = [float(i)/sum(evdList) for i in evdList]

        # plot
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        ax1.plot(iterations, NLList)
        ax1.plot(iterations, NLList, 'r+')
        ax1.set_title('NLL')

        ax2.plot(iterations, evdList)
        ax2.plot(iterations, evdList, 'r+')
        ax2.set_title('Expected Value Diff')
        plt.show()

        # calculate metrics for printing
        v, q, logp, thisp = linearvalueiteration(
            mdp_data, output.view(4, 1))  # to get policy under out R
        thisoptimal_policy = np.argmax(thisp.detach().cpu().numpy(), axis=1)

        print(
            '\nTrue R: \n{}\n - with optimal policy {}'.format(r[:, 0].view(4, 1), optimal_policy))
        print('\nFinal Estimated R after 100 optim steps: \n{}\n - with optimal policy {}\n - avg EVD of {}'.format(
            finaloutput.view(4, 1), thisoptimal_policy, sum(evdList)/len(evdList)))

    def torchbasic(self, lh, type_optim):

        # Initalise params

        countlist = []
        NLLlist = []
        gradList = []
        estRlist = []
        evdList = []
        lr = 1
        n_epochs = 1000
        NLL = 0
        prev = 0
        diff = 1
        threshhold = 0.1
        i = 0
        # initial estimated R)
        estR = torch.randn(mdp_data['states'], 1,
                           dtype=torch.float64, requires_grad=True)
        if(type_optim == 'LBFGS'):
            optimizer = torch.optim.LBFGS([estR], lr=lr, max_iter=20, max_eval=None,
                                          tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                NLL = lh.negated_likelihood(estR)
                if NLL.requires_grad:
                    estR.grad = lh.calc_gradient(estR)
                return NLL
            print("... minimising likelihood with LBFGS...\n")
            while (diff >= threshhold):
                i += 1
                prev = NLL
                NLL = optimizer.step(closure)
                diff = abs(prev-NLL)
                print('Optimiser iteration {} with NLL {}, estR values of \n{} and gradient of \n{} and abs diff of {}\n'.format(
                    i, NLL, estR.data, estR.grad, diff))
                # store values for plotting
                evd = lh.calculate_EVD(truep)
                evdList.append(evd)
                gradList.append(torch.sum(estR.grad))
                NLLlist.append(NLL)
                countlist.append(i)
                estRlist.append(torch.sum(estR.data))

        else:
            optimizer = torch.optim.Adam([estR], lr=lr)
            print("... minimising likelihood with Adam...\n")
            while (diff >= threshhold):
                optimizer.zero_grad()
                i += 1
                prev = NLL
                NLL = lh.negated_likelihood(estR)
                estR.grad = lh.calc_gradient(estR)
                optimizer.step()
                diff = abs(prev-NLL)
                print('Optimiser iteration {} with NLL {}, estR values of \n{} and gradient of \n{} and abs diff of {}\n'.format(
                    i, NLL, estR.data, estR.grad, diff))  # store values for plotting
                evd = lh.calculate_EVD(truep)
                evdList.append(evd)
                gradList.append(torch.sum(estR.grad))
                NLLlist.append(NLL)
                countlist.append(i)
                estRlist.append(torch.sum(estR.data))

        # Normalise data for plotting
        NLLlist = [float(i)/sum(NLLlist) for i in NLLlist]
        gradList = [float(i)/sum(gradList) for i in gradList]
        estRlist = [float(i)/sum(estRlist) for i in estRlist]

        # plot
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True)
        ax1.plot(countlist, NLLlist)
        ax1.set_title('Likelihood')
        # ax1.xlabel('Iterations')
        ax2.plot(countlist, gradList)
        ax2.set_title('grad')
        # ax2.xlabel('Iterations')
        ax3.plot(countlist, estRlist)
        ax3.set_title('estR')
        # ax3.xlabel('Iterations')
        ax4.plot(countlist, evdList)
        ax4.set_title('Expected Value Diff')
        # ax4.xlabel('Iterations')
        plt.show()

        # reshape foundR & find it's likelihood
        foundR = torch.reshape(torch.tensor(estR.data), (4, 1))
        foundR = foundR.repeat(1, 5)
        print(foundR.dtype)
        foundLH = lh.negated_likelihood(foundR)

        # solve MDP with foundR for optimal policy
        v, q, logp, foundp = linearvalueiteration(mdp_data, foundR)
        found_optimal_policy = np.argmax(foundp.detach().cpu().numpy(), axis=1)

        # print
        print("\nTrue R is \n{}\n with negated likelihood of {}\n and optimal policy {}\n".format(
            r, trueNLL, optimal_policy))
        foundRprintlist = [foundR, foundLH, found_optimal_policy]
        print("\nFound R is \n{}\n with negated likelihood of {}\n and optimal policy {}\n".format(
            *foundRprintlist))

    def scipy(self, lh):

        estR = np.random.randn(mdp_params['n']**2, 1)  # initial estimated R
        res = minimize(lh.negated_likelihood_with_grad, estR, jac=True, method="L-BFGS-B", options={
                       'disp': True, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})
        # reshape foundR & find it's likelihood
        foundR = torch.reshape(torch.tensor(res.x), (4, 1))
        foundR = foundR.repeat(1, 5)
        print(foundR.dtype)
        foundLH = lh.negated_likelihood(foundR)

        # solve MDP with foundR for optimal policy
        v, q, logp, foundp = linearvalueiteration(mdp_data, foundR)
        found_optimal_policy = np.argmax(foundp.detach().cpu().numpy(), axis=1)

        print("\nTrue R is \n{}\n with negated likelihood of {}\n and optimal policy {}\n".format(
            *trueRprintlist))

        # Print found R stats
        foundRprintlist = [foundR, foundLH, found_optimal_policy]
        print("\nFound R is \n{}\n with negated likelihood of {}\n and optimal policy {}\n".format(
            *foundRprintlist))

    def getNNpreds(minimise, mynet, num_nets):
        wb_vals = {}
        X = torch.Tensor([[0, 0],
                        [1, 0],
                        [2, 0],
                        [3, 0]])

        preds = torch.empty(num_nets, mdp_params['n']**2)

        for i in range(num_nets):
            mynet = minimise.nonLinearNN(
                evdThreshold=0.02, optim_type='Adam', net=mynet)

            preds[i] = testNN(mynet, X)  # save predicted R from this net

            params = {}  # save weights and biases
            params['fc1'] = {'weight': mynet.fc1.weight, 'bias': mynet.fc1.bias}
            params['fc2'] = {'weight': mynet.fc1.weight, 'bias': mynet.fc1.bias}
            wb_vals['net' + str(i)] = params

            for layer in mynet.children():  # reset net params
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        return preds

    def model(x_data, y_data):

        fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight),
                            scale=torch.ones_like(net.fc1.weight))
        fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias),
                            scale=torch.ones_like(net.fc1.bias))

        fc2w_prior = Normal(loc=torch.zeros_like(net.fc2.weight),
                            scale=torch.ones_like(net.fc2.weight))
        fc2b_prior = Normal(loc=torch.zeros_like(net.fc2.bias),
                            scale=torch.ones_like(net.fc2.bias))

        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
                'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}

        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", net, priors)
        # sample a regressor (which also samples w and b)
        lifted_reg_model = lifted_module()

        lhat = lifted_reg_model(x_data)

        #print('Lhat', lhat)

        # change from binomial as reward estimate is NOT bionmial dis
        pyro.sample("obs", Binomial(logits=lhat), obs=y_data)

    def guide(x_data, y_data):
        # First layer weight distribution priors
        fc1w_mu = torch.randn_like(net.fc1.weight)
        fc1w_sigma = torch.randn_like(net.fc1.weight)
        fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
        fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
        fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
        # First layer bias distribution priors
        fc1b_mu = torch.randn_like(net.fc1.bias)
        fc1b_sigma = torch.randn_like(net.fc1.bias)
        fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
        fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
        fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
        # Output layer weight distribution priors
        fc2w_mu = torch.randn_like(net.fc2.weight)
        fc2w_sigma = torch.randn_like(net.fc2.weight)
        fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
        fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", fc2w_sigma))
        fc2w_prior = Normal(loc=fc2w_mu_param,
                            scale=fc2w_sigma_param).independent(1)
        # Output layer bias distribution priors
        fc2b_mu = torch.randn_like(net.fc2.bias)
        fc2b_sigma = torch.randn_like(net.fc2.bias)
        fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
        fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", fc2b_sigma))
        fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)
        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
                'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}

        lifted_module = pyro.random_module("module", net, priors)

        return lifted_module()

    def variationalweightsBNN(r, X, net):

        optim = Adam({"lr": 0.01})
        svi = SVI(model, guide, optim, loss=Trace_ELBO())

        num_iterations = 5
        loss = 0
        r = r[:, 0].view(len(r), 1)  # make r column vector to match X
        for j in range(num_iterations):
            loss = 0
            output = torch.empty(len(X))
            indexer = 0
            for f in X:
                # calculate the loss and take a gradient step
                loss += svi.step(f.view(-1, len(f)), r[(X == f)])
            print("Iter ", j, " Loss ", loss)

        # insert code to test how accurate BNN is i.e make predictions. last section of code from https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd


#Get which benchmark
if len(sys.argv) > 1:
    worldtype = str(sys.argv[1]) #benchmark type from cmd line
    user_input = True
else:
    user_input = False

#Get number of paths
if len(sys.argv) > 2:
    N = int(str(sys.argv[2])) #number of sampled from cmd line
else:
    N = 32 #default value


T = 16 #number of actions in each trajectory


#set trigger params
final_figures = True
NLL_EVD_plots = False 
heatmapplots = False
new_paths = True
normalise = False

if new_paths:
    print('\n... new paths will be sampled and dumped to NNIRL_param_list.pkl this run ...\n')
else:
    print('\n... loading paths and params from NNIRL_param_list.pkl ...\n')

# Generate mdp and R
# Will always produce constant results due to constant random seed param
if(user_input):
    if worldtype == "gridworld" or worldtype == "gw" or worldtype == "grid":
        print('\n... Creating GridWorld ...\n')
        mdp_data, r, feature_data, mdp_params = create_gridworld()
    elif worldtype == "objectworld" or worldtype == "ow" or worldtype == "obj":
        print('\n... Creating ObjectWorld ...\n')
        mdp_data, r, feature_data, true_feature_map, mdp_params = create_objectworld()
else:
    worldtype = 'gridworld'
    print('\n... Creating GridWorld ...\n')
    mdp_data, r, feature_data, mdp_params = create_gridworld()

if normalise:
    scaler = MinMaxScaler()
    r = torch.tensor(scaler.fit_transform(r.data.cpu().numpy()))

#Solve MDP
print("\n... performing value iteration for v, q, logp and truep ...")
v, q, logp, truep = linearvalueiteration(mdp_data, r)
mdp_solution = {'v': v, 'q': q, 'p': truep, 'logp': logp}
optimal_policy = torch.argmax(truep, axis=1)
print("\n... done ...")

#Sample paths
if new_paths:
    print("\n... sampling paths from true R ...")
    example_samples = sampleexamples(N, T, mdp_solution, mdp_data)
    print("\n... done sampling", N, "paths ...")


NLL = NLLFunction()  # initialise NLL
if new_paths:
    initD, mu_sa, muE, F, mdp_data = NLL.calc_var_values(mdp_data, N, T, example_samples, feature_data)  # calculate required variables
else:
    print("\n... using pre-loaded sampled paths ...")
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

# assign constant class variable
NLL.F = F
NLL.muE = muE
NLL.mu_sa = mu_sa
NLL.initD = initD
NLL.mdp_data = mdp_data


configuration_dict = {'number_of_epochs': 200, 'base_lr': 0.1, 'i2': 30, 'h1_out': 15, 'h2_out': 6} #set config params for clearml
mynet = None
#print('\n using neural network with parameters: ', configuration_dict)

trueNLL = NLL.apply(r, initD, mu_sa, muE, F, mdp_data)  # NLL for true R

#run single NN 
#single_net, feature_weights, run_time = run_single_NN()

if new_paths:
    print('\n... Saving new variables ...\n')
    #save params for NNIRL to file
    NNIRL_param_list = [0.01, "Adam", mynet, initD, mu_sa, muE, mdp_data, truep, NLL_EVD_plots, example_samples, mdp_params, r, mdp_solution, feature_data, trueNLL, normalise, user_input, worldtype]
    file_name = "NNIRL_param_list.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(NNIRL_param_list, open_file)
    open_file.close()

num_first_half_states = 0
num_second_half_states = 0
num_moves = 0
for path in example_samples:
    for move in path:
        num_moves +=1
        if move[0] in np.arange(0,127,1):
            num_first_half_states +=1
        if move[0] in np.arange(128,255,1):
            num_second_half_states +=1


print('No states from paths between 1-128: ', num_first_half_states)
print('No states from paths between 128-256: ', num_second_half_states)
print('Percent of states from paths between 1-128: ', (num_first_half_states/num_moves)*100)
print('Percent of states from paths between 128-256: ', (num_second_half_states/num_moves)*100)



'''
# run NN ensemble
models_to_train = 10  # train this many models
max_epochs = 3      # for this many epochs
iters_per_epoch = 50
learning_rate = 0.1
models, model_weights =  run_NN_ensemble(models_to_train, max_epochs, iters_per_epoch, learning_rate, feature_data['splittable'])

#get ensemble model predictions
ensemble_models = []
for _, row in model_weights.iterrows():
    # Compute test prediction for this iteration of ensemble weights
    tmp_y_hat = np.array(
        [models[model_name] * weight
            for model_name, weight in row.items()]
    ).sum(axis=0)
    ensemble_models.append(tmp_y_hat)
ensemble_models = pd.Series(ensemble_models)


print('for model in ensemble models line')
for model in ensemble_models:
    print(model)


#print metrics for true R 
print("\nTrue R: {}\n - negated likelihood: {}\n - optimal policy: {}".format(r[:, 0], trueNLL, optimal_policy))  # Printline if LH is scalar

#calculate metrics for final R from ensemble
v, q, logp, ensemblep = linearvalueiteration(mdp_data, ensemble_models.iloc[-1].view(4, 1))
ensembleoptimal_policy = np.argmax(ensemblep.detach().cpu().numpy(), axis=1)

#print metrics for final R from ensemble
print('\nFinal Estimated R from ensemble NN: \n{}\n - with optimal policy {}\n - EVD of {}\n - negated likelihood: {}'.format(ensemble_models.iloc[-1], ensembleoptimal_policy, NLL.calculate_EVD(truep, torch.matmul(X, ensemble_models.iloc[-1])), NLL.apply(ensemble_models.iloc[-1], initD, mu_sa, muE, F, mdp_data)))

#stack all predicted rewards
predictions = torch.empty(len(ensemble_models), mdp_params['n']**2)
i = 0
for predictedR in ensemble_models:
    predictions[i] = predictedR
    i += 1

#get average predicted reward and uncertainty of all models
average_predictions = torch.empty(mdp_params['n']**2)
predictions_uncertainty = torch.empty(mdp_params['n']**2)
for column in range(predictions.size()[1]):
    average_predictions[column] = torch.mean(predictions[:, column]) #save avg predicted R for each state
    predictions_uncertainty[column] = torch.var(predictions[:, column]) #save variance for each states prediciton as uncertainty

#calculate metrics for avg R from ensemble
v, q, logp, avgensemblep = linearvalueiteration(mdp_data, average_predictions.view(4, 1))
avgensembleoptimal_policy = np.argmax(avgensemblep.detach().cpu().numpy(), axis=1)
print('\nAverage Estimated R from ensemble NN: \n{}\n - with optimal policy {}\n - EVD of {}\n - negated likelihood: {}'.format(average_predictions, avgensembleoptimal_policy, NLL.calculate_EVD(truep, torch.matmul(X, average_predictions)), NLL.apply(average_predictions, initD, mu_sa, muE, F, mdp_data)))
print('\nPredictions Uncertainty {}'.format(predictions_uncertainty))


#plot uncertainty
x = [1,2,3,4]
y = ensemble_models.iloc[-1].detach().numpy()
yerr = predictions_uncertainty.detach().numpy()
fig, ax = plt.subplots()
ax.errorbar(x, y,
            yerr=yerr,
            fmt='o')
ax.set_xlabel('State')
ax.set_ylabel('Predicted R')
ax.set_title('Predicted R w/ Error Bars')
plt.show()




'''





"""
# set true R equal matlab impl w/ random seed 0
# not a reward func ... a look up table
r = torch.Tensor(np.array(
    [
        [3., 3., 3., 3., 3.],
        [6., 6., 6., 6., 6.],
        [5., 5., 5., 5., 5.],
        [2., 2., 2., 2., 2.]
    ], dtype=np.float64))


# Solve MDP
v, q, logp, truep = linearvalueiteration(mdp_data, r)
mdp_solution = {'v': v, 'q': q, 'p': truep, 'logp': logp}
optimal_policy = np.argmax(truep.detach().cpu().numpy(), axis=1)

# Sample paths
print("\n... sampling paths from true R ... \n")
example_samples = sampleexamples(N, T, mdp_solution, mdp_data)
print("... done ...\n")

NLL = NLLFunction()  # initialise NLL
initD, mu_sa, muE, F, mdp_data = NLL.calc_var_values(
    mdp_data, N, T, example_samples)  # calculate required variables

# assign constant class variable
NLL.F = F
NLL.muE = muE
NLL.mu_sa = mu_sa
NLL.initD = initD
NLL.mdp_data = mdp_data

trueNLL = NLL.apply(r, initD, mu_sa, muE, F, mdp_data)  # NLL for true R

print("\nTrue R: {}\n - negated likelihood: {}\n - optimal policy: {}\n".format(r[:, 0], trueNLL, optimal_policy))  # Printline if LH is scalar

"""

# run single NN
"""
mynet = NonLinearNet()
single_net = run_single_NN(0.003, "Adam", mynet)
"""

'''
# run NN ensemble
models_to_train = 10  # train this many models
max_epochs = 3       # for this many epochs
iters_per_epoch = 50
learning_rate = 0.1
models, model_weights =  run_NN_ensemble(models_to_train, max_epochs, iters_per_epoch, learning_rate)

#get ensemble model predictions
ensemble_models = []
for _, row in model_weights.iterrows():
    # Compute test prediction for this iteration of ensemble weights
    tmp_y_hat = np.array(
        [models[model_name] * weight
            for model_name, weight in row.items()]
    ).sum(axis=0)
    ensemble_models.append(tmp_y_hat)
ensemble_models = pd.Series(ensemble_models)


print('for model in ensemble models line')
for model in ensemble_models:
    print(model)


#print metrics for true R 
print("\nTrue R: {}\n - negated likelihood: {}\n - optimal policy: {}".format(r[:, 0], trueNLL, optimal_policy))  # Printline if LH is scalar

#calculate metrics for final R from ensemble
v, q, logp, ensemblep = linearvalueiteration(mdp_data, ensemble_models.iloc[-1].view(4, 1))
ensembleoptimal_policy = np.argmax(ensemblep.detach().cpu().numpy(), axis=1)

#print metrics for final R from ensemble
print('\nFinal Estimated R from ensemble NN: \n{}\n - with optimal policy {}\n - EVD of {}\n - negated likelihood: {}'.format(ensemble_models.iloc[-1], ensembleoptimal_policy, NLL.calculate_EVD(truep, ensemble_models.iloc[-1]), NLL.apply(ensemble_models.iloc[-1], initD, mu_sa, muE, F, mdp_data)))

#stack all predicted rewards
predictions = torch.empty(len(ensemble_models), mdp_params['n']**2)
i = 0
for predictedR in ensemble_models:
    predictions[i] = predictedR
    i += 1

#get average predicted reward and uncertainty of all models
average_predictions = torch.empty(mdp_params['n']**2)
predictions_uncertainty = torch.empty(mdp_params['n']**2)
for column in range(predictions.size()[1]):
    average_predictions[column] = torch.mean(predictions[:, column]) #save avg predicted R for each state
    predictions_uncertainty[column] = torch.var(predictions[:, column]) #save variance for each states prediciton as uncertainty

#calculate metrics for avg R from ensemble
v, q, logp, avgensemblep = linearvalueiteration(mdp_data, average_predictions.view(4, 1))
avgensembleoptimal_policy = np.argmax(avgensemblep.detach().cpu().numpy(), axis=1)
print('\nAverage Estimated R from ensemble NN: \n{}\n - with optimal policy {}\n - EVD of {}\n - negated likelihood: {}'.format(average_predictions, avgensembleoptimal_policy, NLL.calculate_EVD(truep, average_predictions), NLL.apply(average_predictions, initD, mu_sa, muE, F, mdp_data)))
print('\nPredictions Uncertainty {}'.format(predictions_uncertainty))


#plot uncertainty
x = [1,2,3,4]
y = ensemble_models.iloc[-1].detach().numpy()
yerr = predictions_uncertainty.detach().numpy()
fig, ax = plt.subplots()
ax.errorbar(x, y,
            yerr=yerr,
            fmt='o')
ax.set_xlabel('State')
ax.set_ylabel('Predicted R')
ax.set_title('Predicted R w/ Error Bars')
plt.show()



'''







# variation weights BNN
'''
X = torch.Tensor([[0],
				[1],
				[2],
				[3]]) #1 feature to work with nonLinearBNN config 
net = NonLinearBNN() #maps 1 input feature to 1 output to comply with pyro.svi expectations
variationalweightsBNN(r, X, net)
'''

'''
minimise = minimise()
mynet = NonLinearNet()
preds = getNNpreds(minimise = minimise, mynet = mynet, num_nets = 10)



#for testing to avoid running getNNpreds
preds = torch.Tensor(np.array(
	[
		[-2.316e+77, 2.687e+154,  3.000e+00,  3.000e+00],
        [ 3.000e+00,  6.000e+00,  6.000e+00,  6.000e+00],
        [ 6.000e+00,  6.000e+00,  5.000e+00,  5.000e+00],
        [ 5.000e+00,  5.000e+00,  5.000e+00,  2.000e+00],
        [ 2.000e+00,  2.000e+00,  2.000e+00,  2.000e+00]
	], dtype=np.float64))






#get each states reward predictions
state0preds = preds[:, 0]
state1preds = preds[:, 1]
state2preds = preds[:, 2]
state3preds = preds[:, 3]
predictedRewards = torch.empty(4, dtype=torch.float64)
predictedRewards[0] = torch.mean(state0preds)
predictedRewards[1] = torch.mean(state1preds)
predictedRewards[2] = torch.mean(state2preds)
predictedRewards[3] = torch.mean(state3preds)
rewardUncertanties = torch.empty(4)
rewardUncertanties[0] = torch.var(state0preds)
rewardUncertanties[1] = torch.var(state1preds)
rewardUncertanties[2] = torch.var(state2preds)
rewardUncertanties[3] = torch.var(state3preds)


#print("\nTrue R is \n{}\n with negated tensor likelihood of \n{}\n total: {}\n and optimal policy: {}\n".format(r, trueNLL.view(4).detach().numpy(), sum(trueNLL).detach().numpy(), optimal_policy)) #Printline if LH is tensor
print("\nTrue R: {}\n - negated likelihood: {}\n - optimal policy: {}\n".format(r[:, 0], trueNLL, optimal_policy)) #Printline if LH is scalar
print('NN predicted state 0 has R of {} with an uncertainty of {}'.format(torch.mean(state0preds), torch.var(state0preds)))
print('NN predicted state 1 has R of {} with an uncertainty of {}'.format(torch.mean(state1preds), torch.var(state1preds)))
print('NN predicted state 2 has R of {} with an uncertainty of {}'.format(torch.mean(state2preds), torch.var(state2preds)))
print('NN predicted state 3 has R of {} with an uncertainty of {}'.format(torch.mean(state3preds), torch.var(state3preds)))

#calculate metrics for printing
v, q, logp, thisp = linearvalueiteration(mdp_data, predictedRewards.view(4,1)) #to get policy under out R
thisoptimal_policy = np.argmax(thisp.detach().cpu().numpy(), axis=1) 
estNLL = NLL.apply(predictedRewards,initD, mu_sa, muE, F, mdp_data) #NLL for true R

print('\nNN Estimated R: {}\n- negated likelihood: {}\n- optimal policy {}'.format(predictedRewards, estNLL, thisoptimal_policy))

'''


# plot predicted reward w/ uncertainties
'''
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
ax1.plot(r[:,0].view(4), np.linspace(-10,10,4), xerr=rewardUncertanties, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
ax1.set_title('True R')
ax2.errorbar(predictedRewards, np.linspace(-10,10,4), xerr=rewardUncertanties, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
ax2.set_title('Predicted Rewards')
plt.errorbar(np.linspace(-10,10,4), predictedRewards, xerr=0.2, yerr=0.4, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
plt.show()
'''

"""
# playground from here 
# -------------------------------

[states,actions,transitions] = mdp_data['sa_p'].size()
D = torch.zeros(states,1)
diff = 1.0
threshold = 0.00001
features = F.shape[2-1]
r = torch.rand(features,1)
for j in range(1):
    # Dp = torch.tensor(D.clone().detach().requires_grad_(True), dtype=torch.float64)
    Dp = D
    #left_hold = torch.tensor(np.tile(p[...,None], (1,1,transitions)))
    #LHS = torch.mul(left_hold, mdp_data['sa_p'])
    #RHS = torch.tensor(np.tile(Dp[...,None], (1,actions,transitions)))
    #RHS = RHS0*mdp_data['discount']

  
    Dpi = torch.mul(torch.mul(torch.tensor(np.tile(mdp_solution['p'][...,None], (1,1,transitions))), mdp_data['sa_p'].type('torch.DoubleTensor')), torch.tensor(np.tile(Dp[...,None], (1,actions,transitions)))*mdp_data['discount'])
    
    '''
    print(Dpi.size())
    print('sa_p\n', Dpi[:,:,0])
    print('sa_p\n', Dpi[:,:,1])
    print('sa_p\n', Dpi[:,:,2])
    print('sa_p\n', Dpi[:,:,3])
    print('sa_p\n', Dpi[:,:,4])
    '''

    D_CSR = sps.csc_matrix((Dpi.transpose(1,0).flatten(), mdp_data['sa_s'].transpose(1,0).flatten().type('torch.DoubleTensor'), torch.arange(states*actions*transitions+1)), shape=(states,states*actions*transitions)) 
    D_CSR.eliminate_zeros()    
    D_mx = torch.tensor(D_CSR.todense()) @ torch.ones(states*actions*transitions,1)
    D_s = torch.sum(D_mx,1)
    D_s = D_s.view(len(D_s), 1)
    D = initD + D_s

    print(D)
    
    diff = torch.max(torch.abs(torch.subtract(D,Dp)))


'''
print('sa_p\n', mdp_data['sa_s'][:,:,0])
print('sa_p\n', mdp_data['sa_s'][:,:,1])
print('sa_p\n', mdp_data['sa_s'][:,:,2])
print('sa_p\n', mdp_data['sa_s'][:,:,3])
print('sa_p\n', mdp_data['sa_s'][:,:,4])
'''
"""