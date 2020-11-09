from gridworld import *
from linervalueiteration import *
import pprint
from sampleexamples import *
import numpy as np
np.set_printoptions(suppress=True) 
from likelihood import *
from scipy.optimize import minimize

#set mdp params
mdp_params = {'n':2, 'b':1, 'determinism':1.0, 'discount':0.99, 'seed': 0}
N = 1000
T = 8

#generate mdp and R
mdp_data, r = gridworldbuild(mdp_params)




#set R equal matlab random seed 0
r = np.array([[0.0005  ,  0.0005  ,  0.0005  ,  0.0005  ,  0.0005],
                [0.0000   , 0.0000   , 0.0000   , 0.0000   , 0.0000],
                [4.5109  ,  4.5109  ,  4.5109  ,  4.5109   , 4.5109],
                [4.5339  ,  4.5339  ,  4.5339 ,   4.5339 ,   4.5339]], dtype=np.float32)


#solve MDP
v, q, logp, p = linearvalueiteration(mdp_data, r)
mdp_solution = {'v': v, 'q':q, 'p':p, 'logp': logp}

"""
print("P \n {}".format(p))
print("Q \n {}".format(q))
print("V \n {}".format(v))
"""

optimal_policy = np.argmax(p, axis=1)   

#sample paths
example_samples = sampleexamples(N,T, mdp_solution, mdp_data)
print("\n... Paths sampled with true R ... \n")

#set likelihood variables
lh = Likelihood()
lh.set_variables_for_likehood(mdp_data, N, T, example_samples)

#minimsing likelihood for true R
bnds = [(0,10), (0,10), (0,10), (0,10)]
guess = np.random.randn(mdp_params['n']**2,1)
print("... minimising likelihood for R ...")
res = minimize(lh.negated_likelihood, guess, jac=True, method="L-BFGS-B", options={"disp": True})

#reshape & find likelihood
foundR = np.reshape(res.x, (4,1))
foundR = np.tile(foundR, (1, 5))
foundLH = lh.negated_likelihood(foundR)[0]

#solve MDP for optimal policy
v, q, logp, p = linearvalueiteration(mdp_data, foundR)
found_optimal_policy = np.argmax(p, axis=1) 
foundRprintlist = [foundR, foundLH, found_optimal_policy]


#Print true R stats
trueLH = lh.negated_likelihood(r)[0]


trueRprintlist = [r, trueLH, optimal_policy]
print("\nTrue R is \n{}\n with negated likelihood of {}\n and optimal policy {}".format(*trueRprintlist))

#Print found R stats
print("\nFound R is \n{}\n with negated likelihood of {}\n and optimal policy {}".format(*foundRprintlist))




#uncomment all prints above to debug

"""
#code to evaluate consistency when scaling R
print("\n ... Doubling R & recalcuating ... \n")
doubledR = r*2
#solve MDP
v, q, logp, p = linearvalueiteration(mdp_data, doubledR)
mdp_solution = {'v': v, 'q':q, 'p':p, 'logp': logp}
optimal_policy = np.argmax(p, axis=1)   

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
optimal_policy = np.argmax(p, axis=1)   


#set likelihood variables
lh = Likelihood()
lh.set_variables_for_likehood(mdp_data, N, T, example_samples)


doubledLH = lh.negated_likelihood(doubledR)
doubledRprintlist = [doubledR, trueLH,optimal_policy]
print("\nQuadrupled R is \n{}\n with negated likelihood of {}\n and optimal policy {}".format(*doubledRprintlist))

"""


























