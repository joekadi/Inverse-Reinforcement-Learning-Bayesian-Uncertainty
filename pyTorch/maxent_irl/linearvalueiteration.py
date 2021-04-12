import numpy as np
import copy
import torch
import math

def maxentsoftmax(q):

    maxx = q.max(1)[0]
    maxx = maxx.view(len(maxx),1) #make column vector

    #line using tile -- working!
    #inside = torch.sum(torch.exp(q - torch.tile(maxx, (1, q.shape[1]))),1)

    #experimental line using torch.repeat
    
    inside = torch.sum(torch.exp(q - maxx.repeat(1, q.shape[1])   ),1)

    inside = torch.reshape(inside, (len(inside), 1))
    v = maxx + torch.log(inside)
    return v


def linearvalueiteration(mdp_data, r):

    v = torch.zeros((int(mdp_data['states']), 1), dtype=float)
    diff = 1.0
    count = 0
    didwork = 0
    didntwork = 0
    while diff >= 0.00001:
        count +=1 
        vp = v.detach().clone()

        '''
        for i in range(len(vp)): 
            sa_s_indexer = (mdp_data['sa_s'].type(torch.LongTensor)== float(i)).type(torch.LongTensor)
            #ensure no value in sa_s_indexer is greater than max no. of states i.e the length of dim 0 in sa_s
            sa_s_indexer[sa_s_indexer > sa_s.size()[0]-1 ] = sa_s.size()[0]-1
            sa_s[sa_s_indexer]   = vp[i] #vp(mdp_data.sa_s)
        '''        
        #q = r + mdp_data.discount*sum(mdp_data.sa_p.* vp(mdp_data.sa_s) ,3); - matlab
        q = r + mdp_data['discount']*torch.sum(mdp_data['sa_p'] * vp[mdp_data['sa_s'], 0], 2)
        #q = r + mdp_data['discount']* torch.sum(mdp_data['sa_p']*sa_s, 2)
        #q = r + mdp_data['discount']* torch.sum(mdp_data['sa_p']*vp[mdp_data['sa_s'].type(torch.LongTensor)], 2)
        v = maxentsoftmax(q)
        diff = max(abs(v-vp))


    #compute policy
    #print('q\n', q)
    #logp = q - v.repeat([1,int(mdp_data['actions'])])

    #line using tile -- working!
    #logp = q - torch.tensor(torch.tile(v, (1, int(mdp_data['actions']))))

    #experimental line using torch.repeat
    logp = q - v.repeat(1,int(mdp_data['actions']) )   


    p = torch.exp(logp)

    return v, q, logp, p

