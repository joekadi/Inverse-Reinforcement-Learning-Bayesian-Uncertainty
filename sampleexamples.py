import numpy as np
import copy
import math
"""
 def step(self, state, action):
        
        outcomes = self.T[state,action].values() #get T(s'|s,a)
        probs = [] #empty list
        for item in outcomes:
            probs.append(item) #populate probs list with outcomes 
        probs = np.array(probs) #cast probs list to np.array
        newstatearray = np.random.multinomial(1, probs).tolist() #get new state array e.g [0,1,0]
        new_state = newstatearray.index(1) #get new state sample(T(s'|s,a))
        return new_state
"""

def step(mdp_data, s, action):
    #Random sample for stochastic step.
    r = np.random.rand(1,1)
    sm = 0
    for k in range(mdp_data['sa_p'].shape[2]):
        sm = sm+mdp_data['sa_p'][s,action,k]

        """
        printlist = [sm, r]
        print("Sm {} vs R {}".format(*printlist))
        """

        if sm >= r:
            s = mdp_data['sa_s'][s,action,k]
            
            return s

    print("Something's wrong, step returned -1")
    return -1

def optimal_action(mdp_data, mdp_solution, s):
    samp = np.random.rand(1,1)
    total = 0
    for a in range(mdp_data['actions']):
        total = total+mdp_solution['p'][s,a]
        if total >= samp:
            return a
    print("Something's wrong, optimal_action() returned -1")
    return -1

def sampleexamples(N,T, mdp_solution, mdp_data):
    example_samples =  [None] * N
    action = None
    for i in range(N):
        s = np.ceil(np.random.rand(1,1)*mdp_data['states']) #sample intial states
        s = int(s-1) #for 0 index
        path = [None] * T
        #print("initial state {}".format(s))
        #run sample trajectory
        for t in range(T):
            #if t != 0:
                #print("New state {}".format(s))
            #get optimal action given state
            staterow = mdp_solution['q'][s,:]
            action = optimal_action(mdp_data, mdp_solution, s)
            #store sample
            path[t] = (s,action)
            #next_state
            s = step(mdp_data, s, action)
        example_samples[i] = path
    return example_samples