from gridworld import *
from linervalueiteration import *
import pprint
from sampleexamples import *
import numpy as np
np.set_printoptions(suppress=True) 
from likelihood import *
from scipy.optimize import minimize, check_grad
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
torch.set_printoptions(precision=3)

class testers:
	def checkgradients(lh, mdp_params, k):
		#checking gradient for k reward function points

		rewardsToCheck = []
		for i in range(k):
			checkR = np.random.randn(mdp_params['n']**2,1) #initial estimated R
			rewardsToCheck.append(checkR)

		print("... checking gradient ...\n")

		total_rmse = []
		true_gradients = []
		expected_gradients = []

		for reward in rewardsToCheck:
			rmse, true_grad, expected_grad = check_grad(lh.negated_likelihood, lh.calc_gradient, [reward], epsilon = 1e-4)
			total_rmse.append(rmse)
			true_gradients.append(true_grad.item())
			expected_gradients.append(expected_grad)

		#print and plot RMSE
		print('\nGradient check terminated with a RMSE of {} from {} gradient checks\n'.format(sum(total_rmse)/len(total_rmse), len(rewardsToCheck)))
		plt.plot(true_gradients, expected_gradients, 'bo')
		plt.title('check_grad() gradient comparison')
		plt.xlabel('True Gradients')
		plt.ylabel('Expected Gradients')
		plt.grid(b='True', which='minor')
		plt.show()

	def test_gradient(lh, testr):
		print('Gradient for test r is \n{}'.format(lh.calc_gradient(testr)))
		return(lh.calc_gradient(testr))

	def test_likelihood(lh, testr):
		print('Likelihood for test r is {}'.format(lh.negated_likelihood(testr)))
		return(lh.negated_likelihood(testr))

	def compare_gandLH_with_matlab(lh):
		torchG = test_gradient(lh, testr)
		torchL =test_likelihood(lh, testr)

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

		print('Elementwise diff torch gradient - matlab gradient is \n {}'.format(np.subtract(torchG.detach().cpu().numpy(),matlabG)))
		print('Likelihood diff is {}'.format(torchL - matlabL))

class minimisers:
	def minimise_likelihood_with_torch(self, lh, type_optim):
		
		#Initalise params
		estR = torch.randn(mdp_params['n']**2,1, dtype=torch.float64, requires_grad=True) #initial estimated R
		countlist = []
		NLLlist = []
		gradList = []
		estRlist = []
		lr = 1
		n_epochs = 1000
		NLL = 0
		prev = 0
		diff = 1
		threshhold =  0.1
		i = 0
		if(type_optim == 'LBFGS'):
			optimizer = torch.optim.LBFGS([estR], lr=lr, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
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
				print('Optimiser iteration {} with NLL {}, estR values of \n{} and gradient of \n{} and abs diff of {}\n'.format(i, NLL, estR.data, estR.grad, diff))
				#store values for plotting
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

				print('Optimiser iteration {} with NLL {}, estR values of \n{} and gradient of \n{} and abs diff of {}\n'.format(i, NLL, estR.data, estR.grad, diff))				#store values for plotting
				gradList.append(torch.sum(estR.grad))
				NLLlist.append(NLL)
				countlist.append(i)
				estRlist.append(torch.sum(estR.data))

		#Normalise data for plotting
		NLLlist = [float(i)/sum(NLLlist) for i in NLLlist]
		gradList = [float(i)/sum(gradList) for i in gradList]
		estRlist = [float(i)/sum(estRlist) for i in estRlist]

		#plot
		f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
		ax1.plot(countlist, NLLlist)
		ax1.set_title('Likelihood')
		ax2.plot(countlist, gradList)
		ax2.set_title('grad')
		ax3.plot(countlist, estRlist)
		ax3.set_title('estR')
		plt.show()


		#reshape foundR & find it's likelihood
		foundR = torch.reshape(torch.tensor(estR.data), (4,1))
		foundR = foundR.repeat(1, 5)
		print(foundR.dtype)
		foundLH = lh.negated_likelihood(foundR)

		#solve MDP with foundR for optimal policy
		v, q, logp, foundp = linearvalueiteration(mdp_data, foundR)
		found_optimal_policy = np.argmax(foundp.detach().cpu().numpy(), axis=1) 

		#print
		print("\nTrue R is \n{}\n with negated likelihood of {}\n and optimal policy {}\n".format(*trueRprintlist))
		foundRprintlist = [foundR, foundLH, found_optimal_policy]
		print("\nFound R is \n{}\n with negated likelihood of {}\n and optimal policy {}\n".format(*foundRprintlist))

	def minimise_likelihood_with_scipy(self, lh):

		estR = np.random.randn(mdp_params['n']**2,1) #initial estimated R
		res = minimize(lh.negated_likelihood_with_grad, estR, jac=True, method="L-BFGS-B", options={'disp': True, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})
		#reshape foundR & find it's likelihood
		foundR = torch.reshape(torch.tensor(res.x), (4,1))
		foundR = foundR.repeat(1, 5)
		print(foundR.dtype)
		foundLH = lh.negated_likelihood(foundR)

		#solve MDP with foundR for optimal policy
		v, q, logp, foundp = linearvalueiteration(mdp_data, foundR)
		found_optimal_policy = np.argmax(foundp.detach().cpu().numpy(), axis=1) 


		print("\nTrue R is \n{}\n with negated likelihood of {}\n and optimal policy {}\n".format(*trueRprintlist))

		#Print found R stats
		foundRprintlist = [foundR, foundLH, found_optimal_policy]
		print("\nFound R is \n{}\n with negated likelihood of {}\n and optimal policy {}\n".format(*foundRprintlist))


#set mdp params
mdp_params = {'n':2, 'b':1, 'determinism':1.0, 'discount':0.99, 'seed': 0}
N = 1000
T = 8

print("\n... generating MDP and intial R ... \n")
#generate mdp and R
mdp_data, r = gridworldbuild(mdp_params)
print("\n... done ... \n")


'''
#set true R equal matlab impl w/ random seed 0
#not a reward func ... a look up table
r = torch.Tensor(np.array(
	[
		[0.0005  ,  0.0005  ,  0.0005  ,  0.0005  ,  0.0005],
	    [0.0000   , 0.0000   , 0.0000   , 0.0000   , 0.0000],
	    [4.5109  ,  4.5109  ,  4.5109  ,  4.5109   , 4.5109],
	    [4.5339  ,  4.5339  ,  4.5339 ,   4.5339 ,   4.5339]
	 ], dtype=np.float64))
'''

#Solve MDP
v, q, logp, truep = linearvalueiteration(mdp_data, r)
mdp_solution = {'v': v, 'q':q, 'p':truep, 'logp': logp}
optimal_policy = np.argmax(truep.detach().cpu().numpy(), axis=1) 

#Sample paths
print("\n... sampling paths from true R ... \n")
example_samples = sampleexamples(N,T, mdp_solution, mdp_data)
print("\n... done ... \n")

#Set likelihood variables
lh = Likelihood()
lh.set_variables_for_likehood(mdp_data, N, T, example_samples)

print("\nTrue R is \n{}\n with negated likelihood of {}\n and optimal policy {}\n".format(r, lh.negated_likelihood(r), optimal_policy))

mnms = minimisers()
#mnms.minimise_likelihood_with_torch(lh, 'LBFGS') #use this when on MBP M1
mnms.minimise_likelihood_with_torch(lh, 'Adam')
  
























