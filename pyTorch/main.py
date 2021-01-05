from gridworld import *
from linervalueiteration import *
import pprint
from sampleexamples import *
import numpy as np
np.set_printoptions(suppress=True) 
from likelihood import *
from myNLL import *
from NLLFunction import *
from scipy.optimize import minimize, check_grad
import torch
from torch.autograd import Variable
from torch.autograd import gradcheck
import matplotlib.pyplot as plt
from linearnn import *
from nonlinearnn import *
import torch.nn.functional as F
torch.set_printoptions(precision=3)
torch.set_default_tensor_type(torch.DoubleTensor) #default type to torch.float64
np.set_printoptions(precision=3)

class testers:

	def checkgradients_NN(self, input, linear):
		# gradcheck takes a tuple of tensors as input, check if your gradient
		# evaluated with these tensors are close enough to numerical
		# approximations and returns True if they all verify this condition.
		test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
		print('Gradient check; {}'.format(test))

	def checkgradients(self, lh, mdp_params, k):
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

	def test_gradient(self, lh, testr):
		print('Gradient for test r is \n{}'.format(lh.calc_gradient(testr)))
		return(lh.calc_gradient(testr))

	def test_likelihood(self, lh, testr):
		print('Likelihood for test r is {}'.format(lh.negated_likelihood(testr)))
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

		print('Elementwise diff torch gradient - matlab gradient is \n {}'.format(np.subtract(torchG.detach().cpu().numpy(),matlabG)))
		print('Likelihood diff is {}'.format(torchL - matlabL))

class minimise:

	def nonLinearNN(self, evdThreshold, optim_type, net):

		#net = NonLinearNet()
		tester = testers()		
		
		#initialise rewards by finding true weights for NN. feed features through NN using true Weights to get ground truth reward.
		#initalise with some noise? can we still uncover sensible reward
		#put an l2 regulisariton weight decay on the network weights. fine tune the lambda value
		#bias = false on weight params seems to work when inital R is 0 
		#check gradients with torch.gradcheck


		X = torch.Tensor([[0, 0],
						  [1, 0],
						  [2, 0],
						  [3, 0]
						  ]) #for NN(state feature vector) = reward 

		evd = 10 
		lr = 0.1
		finaloutput = None

		#lists for printing
		NLList = []
		iterations = []
		evdList = []
		i = 0
		
		if (optim_type == 'Adam'):
			print('\nOptimising with torch.Adam\n')
			optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-2) #weight decay for l2 regularisation
			#while(evd > evdThreshold):
			for i in range(50):
				net.zero_grad()
				
				#build output vector as reward for each state w.r.t its features
				output = torch.empty(len(X))
				indexer = 0
				for f in X:
					thisR = net(f.view(-1,len(f)))
					output[indexer] = thisR
					indexer += 1

				loss = NLL.apply(output, initD, mu_sa, muE, F, mdp_data) #get loss from curr output
				#check gradients
				#tester.checkgradients_NN(output, NLL)
				#print('Output {} with grad fn {}'.format(output, output.grad_fn))
				#print('Loss {} with grad fn {}'.format(loss, loss.grad_fn))
				loss.backward() #propagate grad through network
				evd = NLL.calculate_EVD(truep, output) #calc EVD
				optimizer.step()

				#Printline when LH is vector
				#print('{}: output: {} | EVD: {} | loss: {} | {}'.format(i, output.detach().numpy(), evd,loss.detach().numpy(), sum(loss).detach().numpy()))
				#Printline when LH scalar
				print('{}: output: {} | EVD: {} | loss: {} '.format(i, output.detach().numpy(), evd,loss.detach().numpy()))

				#store metrics for printing 
				NLList.append(loss.item())
				iterations.append(i)
				evdList.append(evd.item())
				finaloutput = output
				i += 1

		else:
			print('\nOptimising with torch.LBFGS\n')
			optimizer = torch.optim.LBFGS(net.parameters(), lr=lr)
			i=0
			def closure():
				net.zero_grad()
				
				#build output vector as reward for each state w.r.t its features
				output = torch.empty(len(X))
				indexer = 0
				for f in X:
					thisR = net(f.view(-1,len(f)))
					output[indexer] = thisR
					indexer += 1


				loss = NLL.apply(output, initD, mu_sa, muE, F, mdp_data) #get loss from curr output
				
				#check gradients
				#tester.checkgradients_NN(output, NLL)
				#print('Output {} with grad fn {}'.format(output, output.grad_fn))
				#print('Loss {} with grad fn {}'.format(loss, loss.grad_fn))

				loss.backward() #propagate grad through network
				evd = NLL.calculate_EVD(truep, output) #calc EVD
				print('{}: output: {} | EVD: {} | loss: {} '.format(i, output.detach().numpy(), evd,loss.detach().numpy()))

				#store metrics for printing 
				NLList.append(loss.item())
				iterations.append(i)
				evdList.append(evd.item())
				finaloutput = output
				i += 1

				return loss

			while(evd > evdThreshold):
				optimizer.step(closure)
				


		'''
		#Normalise data
		#NLList = [float(i)/sum(NLList) for i in NLList]
		#evdList = [float(i)/sum(evdList) for i in evdList]
		
		#plot
		f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
		ax1.plot(iterations, NLList)
		ax1.plot(iterations, NLList, 'r+')
		ax1.set_title('NLL')

		ax2.plot(iterations, evdList)
		ax2.plot(iterations, evdList, 'r+')
		ax2.set_title('Expected Value Diff')
		plt.show()

		#calculate metrics for printing
		v, q, logp, thisp = linearvalueiteration(mdp_data, output.view(4,1)) #to get policy under out R
		thisoptimal_policy = np.argmax(thisp.detach().cpu().numpy(), axis=1) 

		print('\nTrue R: \n{}\n - with optimal policy {}'.format(r[:,0].view(4,1), optimal_policy))
		print('\nFinal Estimated R after 100 optim steps: \n{}\n - with optimal policy {}\n - avg EVD of {}'.format(finaloutput.view(4,1),thisoptimal_policy, sum(evdList)/len(evdList)))
		'''
		return net 

	def linearNN(self, evdThreshold, optim_type):
		net = LinearNet()
		tester = testers()		
		
		#initialise rewards by finding true weights for NN. feed features through NN using true Weights to get ground truth reward.
		
		#initalise with some noise? can we still uncover sensible reward


		#put an l2 regulisariton weight decay on the network weights. fine tune the lambda value
		#  bias = false on weight params seems to work when inital R is 0 

		#check gradients with torch.gradcheck

		X = torch.Tensor([[0, 0],
						  [1, 0],
						  [2, 0],
						  [3, 0]]) #for NN(state feature vector) = reward 

		  
		


		'''
		X = torch.Tensor([[0],
				  [1],
				  [2],
				  [3]]) #for (4,4) NN
		'''

		evd = 10 
		lr = 0.1
		finaloutput = None
		#lists for printing
		NLList = []
		iterations = []
		evdList = []
		i = 0
		
		if (optim_type == 'Adam'):
			print('\nOptimising with torch.Adam\n')
			optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-2) #inital adam optimiser, weight decay for l2 regularisation
			while(evd > evdThreshold):
				net.zero_grad()
				
				#build output vector as reward for each state w.r.t its features
				output = torch.empty(len(X))
				indexer = 0
				for f in X: 
					thisR = net(f.view(-1,len(f)))
					output[indexer] = thisR
					indexer += 1


				loss = NLL.apply(output, initD, mu_sa, muE, F, mdp_data) #get loss from curr output
				

				#check gradients
				#tester.checkgradients_NN(output, NLL)

				#print('Output {} with grad fn {}'.format(output, output.grad_fn))
				#print('Loss {} with grad fn {}'.format(loss, loss.grad_fn))

				loss.backward() #propagate grad through network
				evd = NLL.calculate_EVD(truep, output) #calc EVD
				'''
				j = 1
				for p in net.parameters():
					print('Gradient of parameter {} with shape {} is {}'.format(j, p.shape, p.grad))
					j +=1
				j = 0
				'''

				optimizer.step()

				#Printline when LH is vector
				#print('{}: output: {} | EVD: {} | loss: {} | {}'.format(i, output.detach().numpy(), evd,loss.detach().numpy(), sum(loss).detach().numpy()))
				#Printline when LH scalar
				print('{}: output: {} | EVD: {} | loss: {} '.format(i, output.detach().numpy(), evd,loss.detach().numpy()))

				#store metrics for printing 
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
				output = net(X.view(-1,4)) #when NLL layer is (4,4)
				loss = NLL.negated_likelihood(output)
				loss = sum(loss)
				evd = NLL.calculate_EVD(truep)
				print('{}: output: {} | EVD: {} | loss: {}'.format(i, output.detach().numpy(), evd,loss.detach().numpy()))
				current_gradient = NLL.calc_gradient(output)
				#print('Current gradient \n{}'.format(current_gradient))

				#net.fc1.weight.grad = current_gradient.repeat(1,4) 
				loss.backward(gradient=torch.argmax(current_gradient)) #much worse than above
				'''												 
				print('Calculated grad \n {}'.format(current_gradient))
				j = 1
				for p in net.parameters():
					print('Gradient of parameter {} \n {}'.format(j, p.grad))
					j +=1
				j = 0
				'''

				#store metrics for printing 
				NLList.append(sum(loss).item())
				iterations.append(i)
				evdList.append(evd.item())
				finaloutput = output
				return loss #.max().detach().numpy()
			for i in range(500):
				optimizer.step(closure)

		#Normalise data
		#NLList = [float(i)/sum(NLList) for i in NLList]
		#evdList = [float(i)/sum(evdList) for i in evdList]
		
		#plot
		f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
		ax1.plot(iterations, NLList)
		ax1.plot(iterations, NLList, 'r+')
		ax1.set_title('NLL')

		ax2.plot(iterations, evdList)
		ax2.plot(iterations, evdList, 'r+')
		ax2.set_title('Expected Value Diff')
		plt.show()

		#calculate metrics for printing
		v, q, logp, thisp = linearvalueiteration(mdp_data, output.view(4,1)) #to get policy under out R
		thisoptimal_policy = np.argmax(thisp.detach().cpu().numpy(), axis=1) 

		print('\nTrue R: \n{}\n - with optimal policy {}'.format(r[:,0].view(4,1), optimal_policy))
		print('\nFinal Estimated R after 100 optim steps: \n{}\n - with optimal policy {}\n - avg EVD of {}'.format(finaloutput.view(4,1),thisoptimal_policy, sum(evdList)/len(evdList)))

	def torchbasic(self, lh, type_optim):
		
		#Initalise params
		
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
		threshhold =  0.1
		i = 0
		estR = torch.randn(mdp_data['states'],1, dtype=torch.float64, requires_grad=True) #initial estimated R)
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
				print('Optimiser iteration {} with NLL {}, estR values of \n{} and gradient of \n{} and abs diff of {}\n'.format(i, NLL, estR.data, estR.grad, diff))				#store values for plotting
				evd = lh.calculate_EVD(truep)
				evdList.append(evd)
				gradList.append(torch.sum(estR.grad))
				NLLlist.append(NLL)
				countlist.append(i)
				estRlist.append(torch.sum(estR.data))

		#Normalise data for plotting
		NLLlist = [float(i)/sum(NLLlist) for i in NLLlist]
		gradList = [float(i)/sum(gradList) for i in gradList]
		estRlist = [float(i)/sum(estRlist) for i in estRlist]

		#plot
		f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True)
		ax1.plot(countlist, NLLlist)
		ax1.set_title('Likelihood')
		#ax1.xlabel('Iterations')
		ax2.plot(countlist, gradList)
		ax2.set_title('grad')
		#ax2.xlabel('Iterations')
		ax3.plot(countlist, estRlist)
		ax3.set_title('estR')
		#ax3.xlabel('Iterations')
		ax4.plot(countlist, evdList)
		ax4.set_title('Expected Value Diff')
		#ax4.xlabel('Iterations')
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
		print("\nTrue R is \n{}\n with negated likelihood of {}\n and optimal policy {}\n".format(r, trueNLL, optimal_policy))
		foundRprintlist = [foundR, foundLH, found_optimal_policy]
		print("\nFound R is \n{}\n with negated likelihood of {}\n and optimal policy {}\n".format(*foundRprintlist))

	def scipy(self, lh):

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


def testNN(net, X):
	#build output vector as reward for each state w.r.t its features
	output = torch.empty(len(X))
	indexer = 0
	for f in X:
		thisR = net(f.view(-1,len(f)))
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
		mynet = minimise.nonLinearNN(evdThreshold = 0.02, optim_type = 'Adam', net = mynet)

		preds[i] = testNN(mynet, X) #save predicted R from this net

		params = {} #save weights and biases
		params['fc1'] = {'weight': mynet.fc1.weight, 'bias': mynet.fc1.bias}
		params['fc2'] = {'weight': mynet.fc1.weight, 'bias': mynet.fc1.bias}
		wb_vals['net' + str(i)] = params

		for layer in mynet.children(): 	#reset net params
			if hasattr(layer, 'reset_parameters'):
					layer.reset_parameters()

	return preds


#set mdp params
mdp_params = {'n':2, 'b':1, 'determinism':1.0, 'discount':0.99, 'seed': 0}
N = 2000
T = 8

print("\n... generating MDP and intial R ... \n")
#generate mdp and R
mdp_data, r = gridworldbuild(mdp_params)
print("... done ...")

#set true R equal matlab impl w/ random seed 0
#not a reward func ... a look up table

"""
r = torch.Tensor(np.array(
	[
		[0.0000 ,  0.0000  ,  0.0000  ,  0.0000  ,  0.0000],
	    [1.   , 1.   , 1.   , 1.   , 1.],
	    [2.,2.,2.,2.,2.],
	    [3.  , 3.  ,  3.,   3.,   3.0]], dtype=np.float64))
"""

r = torch.Tensor(np.array(
	[
		[3.,3.,3.,3.,3.],
		[6.,6.,6.,6.,6.],
	    [5.,5.,5.,5.,5.],
	    [2.,2.,2.,2.,2.]
	], dtype=np.float64))


#Solve MDP
v, q, logp, truep = linearvalueiteration(mdp_data, r)
mdp_solution = {'v': v, 'q':q, 'p':truep, 'logp': logp}
optimal_policy = np.argmax(truep.detach().cpu().numpy(), axis=1) 

#Sample paths
print("\n... sampling paths from true R ... \n")
example_samples = sampleexamples(N,T, mdp_solution, mdp_data)
print("... done ...")

NLL = NLLFunction() #initialise NLL
initD, mu_sa, muE, F, mdp_data = NLL.calc_var_values(mdp_data, N, T, example_samples) #calculate required variables 

#assign constant class variable
NLL.F = F
NLL.muE = muE
NLL.mu_sa = mu_sa
NLL.initD = initD
NLL.mdp_data = mdp_data

trueNLL = NLL.apply(r,initD, mu_sa, muE, F, mdp_data) #NLL for true R

#'''
minimise = minimise()
mynet = NonLinearNet()
preds = getNNpreds(minimise = minimise, mynet = mynet, num_nets = 10)


'''
#for testing to avoid running getNNpreds
preds = torch.Tensor(np.array(
	[
		[-2.316e+77, 2.687e+154,  3.000e+00,  3.000e+00],
        [ 3.000e+00,  6.000e+00,  6.000e+00,  6.000e+00],
        [ 6.000e+00,  6.000e+00,  5.000e+00,  5.000e+00],
        [ 5.000e+00,  5.000e+00,  5.000e+00,  2.000e+00],
        [ 2.000e+00,  2.000e+00,  2.000e+00,  2.000e+00]
	], dtype=np.float64))
'''


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

#plot predicted reward w/ uncertainties
'''
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
ax1.plot(r[:,0].view(4), np.linspace(-10,10,4), xerr=rewardUncertanties, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
ax1.set_title('True R')
ax2.errorbar(predictedRewards, np.linspace(-10,10,4), xerr=rewardUncertanties, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
ax2.set_title('Predicted Rewards')
plt.errorbar(np.linspace(-10,10,4), predictedRewards, xerr=0.2, yerr=0.4, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
plt.show()
'''

