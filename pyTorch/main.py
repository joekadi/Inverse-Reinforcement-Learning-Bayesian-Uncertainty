from gridworld import *
from linervalueiteration import *
import pprint
from sampleexamples import *
import numpy as np
np.set_printoptions(suppress=True) 
from likelihood import *
from scipy.optimize import minimize
import torch
torch.set_printoptions(precision=3)



#set mdp params
mdp_params = {'n':2, 'b':1, 'determinism':1.0, 'discount':0.99, 'seed': 0}
N = 1000
T = 8

#generate mdp and R
mdp_data, r = gridworldbuild(mdp_params)



#set true R equal matlab impl w/ random seed 0
r = torch.Tensor(np.array([[0.0005  ,  0.0005  ,  0.0005  ,  0.0005  ,  0.0005],
                [0.0000   , 0.0000   , 0.0000   , 0.0000   , 0.0000],
                [4.5109  ,  4.5109  ,  4.5109  ,  4.5109   , 4.5109],
                [4.5339  ,  4.5339  ,  4.5339 ,   4.5339 ,   4.5339]], dtype=np.float64))


#solve MDP
v, q, logp, truep = linearvalueiteration(mdp_data, r)
mdp_solution = {'v': v, 'q':q, 'p':truep, 'logp': logp}
optimal_policy = np.argmax(truep.detach().cpu().numpy(), axis=1) 

#sample paths
print("\n... sampling paths from true R ... \n")
example_samples = sampleexamples(N,T, mdp_solution, mdp_data)

#set likelihood variables
lh = Likelihood()
lh.set_variables_for_likehood(mdp_data, N, T, example_samples)


#minimsing likelihood for true R
print("... minimising likelihood for R ...\n")
guess = torch.randn(mdp_params['n']**2,1)


res = minimize(lh.negated_likelihood, guess, jac=True, method="L-BFGS-B", options={'disp': True})



#reshape foundR & find it's likelihood
foundR = torch.reshape(torch.tensor(res.x), (4,1))
foundR = foundR.repeat(1, 5)
print(foundR.dtype
	)
foundLH, founddr = lh.negated_likelihood(foundR)

#solve MDP with foundR for optimal policy
v, q, logp, foundp = linearvalueiteration(mdp_data, foundR)
found_optimal_policy = np.argmax(foundp.detach().cpu().numpy(), axis=1) 


#Print true R stats
trueLH, trueDR = lh.negated_likelihood(r)
trueRprintlist = [r, trueLH, optimal_policy]
print("\nTrue R is \n{}\n with negated likelihood of {}\n and optimal policy {}\n".format(*trueRprintlist))

#Print found R stats
foundRprintlist = [foundR, foundLH, found_optimal_policy]
print("\nFound R is \n{}\n with negated likelihood of {}\n and optimal policy {}\n".format(*foundRprintlist))




#uncomment all prints above to debug

"""
#code to evaluate consistency when scaling R
print("\n ... Doubling R & recalcuating ... \n")
doubledR = r*2
#solve MDP
v, q, logp, p = linearvalueiteration(mdp_data, doubledR)
mdp_solution = {'v': v, 'q':q, 'p':p, 'logp': logp}
optimal_policy = Torch.argmax(p, dim=1)   

#set likelihood variables
lh = Likelihood()
lh.set_variables_for_likehood(mdp_data, N, T, example_samples)


doubledLH = lh.negated_likelihood(doubledR)
doubledRprintlist = [doubledR, trueLH,optimal_policy]
print("\nDouble R is \n{}\n with negated likelihood of {}\n and optimal policy {}".format(*doubledRprintlist))


print("\n ... Quadrupling R & recalculating ... \n")
doubledR = r*4
#solve MDP
v, q, logp, p = linearvalueiteration(mdp_data, doubledR)
mdp_solution = {'v': v, 'q':q, 'p':p, 'logp': logp}
optimal_policy = Torch.argmax(p, dim=1)   


#set likelihood variables
lh = Likelihood()
lh.set_variables_for_likehood(mdp_data, N, T, example_samples)


doubledLH = lh.negated_likelihood(doubledR)
doubledRprintlist = [doubledR, trueLH,optimal_policy]
print("\nQuadrupled R is \n{}\n with negated likelihood of {}\n and optimal policy {}".format(*doubledRprintlist))

"""


























