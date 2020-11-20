import numpy as np
import copy
import torch
import math

def maxentsoftmax(q):

    maxx = q.max(1)[0]
    maxx = maxx.view(len(maxx),1) #change 4 to be number of states
    inside = torch.sum(torch.exp(q - maxx.repeat([1, q.shape[1]])),1)
    inside = torch.reshape(inside, (len(inside), 1))
    v = maxx + torch.log(inside)
    
    return v

def linearvalueiteration(mdp_data, r):
    v = torch.zeros((mdp_data['states'], 1), dtype=float)
    sa_s = copy.copy(mdp_data['sa_s'])
    sa_s = sa_s.type(torch.float64)
    diff = 1.0
    count = 0
    while diff >= 0.00001:
        count +=1 
        vp = copy.copy(v)

        # q = r + mdp_data.discount*sum(mdp_data.sa_p.*vp(mdp_data.sa_s),3);
       
        for i in range(len(vp)): 
            sa_s[mdp_data['sa_s'] == i] = vp[i] #vp(mdp_data.sa_s)

        q = r + mdp_data['discount']* torch.sum(mdp_data['sa_p']*sa_s, 2)
        v = maxentsoftmax(q)

        diff = max(abs(v-vp))
       


    #compute policy
    logp = q - v.repeat([1,mdp_data['actions']])
    p = torch.exp(logp)

    return v, q, logp, p


