from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions import Gamma, Poisson, Normal, Binomial
import pyro.distributions as dist
import pyro
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
from scipy.optimize import minimize, check_grad
import os
from NLLFunction import *
from myNLL import *
from likelihood import *
from gridworld import *
from objectworld import *
from linervalueiteration import *
import pprint
from sampleexamples import *
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)

torch.set_printoptions(precision=3)
# default type to torch.float64
torch.set_default_tensor_type(torch.DoubleTensor)
np.set_printoptions(precision=3)

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

#Functions to execute diff methods

def testNN(net, X):
    # build output vector as reward for each state w.r.t its features
    output = torch.empty(len(X))
    indexer = 0
    for f in X:
        thisR = net(f.view(-1, len(f)))
        output[indexer] = thisR
        indexer += 1
    return output

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

def ensemble_selector(loss_function, optim_for_loss, y_hats, init_size=1,
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
            losses[model] = loss_function.calculate_EVD(truep, y_hat).item()

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
            init_loss = loss_function.calculate_EVD(truep, y_hat).item()

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
                    truep, tmp_y_avg[mod]).item()

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

def run_NN_ensemble(models_to_train, max_epochs, iters_per_epoch, learning_rate):

    # use ensemble selector to generate ensemble of NN's optimised by min loss or min evd depending on "opitim_for_loss" variable
    # state features
    X = torch.Tensor([[0, 0],
                      [1, 0],
                      [2, 0],
                      [3, 0]])

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
        net = NonLinearNet()
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
                yhat = torch.empty(len(X))
                indexer = 0
                for f in X:
                    this_state_r = net(f.view(-1, len(f)))
                    yhat[indexer] = this_state_r
                    indexer += 1

                # compute loss and EVD
                loss = NLL.apply(yhat, initD, mu_sa, muE, F, mdp_data)
                evd = NLL.calculate_EVD(truep, yhat)

                print('{}: output: {} | EVD: {} | loss: {} '.format(
                    i, yhat.detach().numpy(), evd, loss.detach().numpy()))

                # Backpropogate and update weights
                loss.backward()
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
    x_test = torch.Tensor([[0, 0],
                           [1, 0],
                           [2, 0],
                           [3, 0]])  # features for test

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
        y_hat_test = torch.empty(len(X))
        indexer = 0
        for f in x_test:
            this_state_r = net(f.view(-1, len(f)))
            y_hat_test[indexer] = this_state_r
            indexer += 1

        # compute evd & loss
        test_loss = NLL.apply(y_hat_test, initD, mu_sa,
                              muE, F, mdp_data).item()
        test_evd = NLL.calculate_EVD(truep, y_hat_test)

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
        loss_function=NLL, optim_for_loss=True, y_hats=y_hats_test, init_size=1, replacement=True, max_iter=10)

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
        loss_function=NLL, optim_for_loss=False, y_hats=y_hats_test, init_size=1, replacement=True, max_iter=10)

    # Compute evd of the equally weighted ensemble
    evds = []
    for model, predictedR in y_hats_test.items():
        evds.append(NLL.calculate_EVD(truep, predictedR))
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

def run_single_NN(self, evdThreshold, optim_type, net):

        #net = NonLinearNet()
        tester = testers()
        # initialise rewards by finding true weights for NN. feed features through NN using true Weights to get ground truth reward.
        # initalise with some noise? can we still uncover sensible reward
        # put an l2 regulisariton weight decay on the network weights. fine tune the lambda value
        # bias = false on weight params seems to work when inital R is 0
        # check gradients with torch.gradcheck

        X = torch.Tensor([[0, 0],
                          [1, 0],
                          [2, 0],
                          [3, 0]
                          ])  # for NN(state feature vector) = reward

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
            # weight decay for l2 regularisation
            optimizer = torch.optim.Adam(
                net.parameters(), lr=lr, weight_decay=1e-2)
            while(evd > evdThreshold):
                # for i in range(50):
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
            i = 0

            def closure():
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
                print('{}: output: {} | EVD: {} | loss: {} '.format(
                    i, output.detach().numpy(), evd, loss.detach().numpy()))

                # store metrics for printing
                NLList.append(loss.item())
                iterations.append(i)
                evdList.append(evd.item())
                finaloutput = output
                i += 1

                return loss

            while(evd > evdThreshold):
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
                thisoptimal_policy = np.argmax(
                    thisp.detach().cpu().numpy(), axis=1)

                print(
                    '\nTrue R: \n{}\n - with optimal policy {}'.format(r[:, 0].view(4, 1), optimal_policy))
                print('\nFinal Estimated R after 100 optim steps: \n{}\n - with optimal policy {}\n - avg EVD of {}'.format(
                    finaloutput.view(4, 1), thisoptimal_policy, sum(evdList)/len(evdList)))

        return net

N = 2000 #number of sampled trajectories
T = 8 #number of actions in each trajectory

print("\n ... generating MDP and intial R ... \n")
#generate mdp and R
mdp_params = {'n': 2, 'b': 1, 'determinism': 1.0, 'discount': 0.99, 'seed': 0}
mdp_data, r = gridworldbuild(mdp_params)

'''
mdp_params = {'n': 32, 'placement_prob': 0.05, 'c1': 2.0, 'c2': 2.0, 'determinism': 1.0, 'discount': 0.99, 'seed': 0}
mdp_data = objectworldbuild(mdp_params)
'''


print("... done ...")
pprint.pprint(mdp_data)



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


# run single NN
"""
mynet = NonLinearNet()
single_net = run_single_NN(0.003, "Adam", mynet)
"""

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
