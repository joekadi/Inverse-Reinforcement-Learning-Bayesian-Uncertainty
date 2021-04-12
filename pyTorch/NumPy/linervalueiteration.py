import numpy as np
import copy
import math

def maxentsoftmax(q):
    
    maxx = q.max(1)
    maxx = np.reshape(maxx, (len(maxx),1)) #change 4 to be number of states
    inside = np.sum(np.exp(q - np.tile(maxx,[1, np.shape(q)[1]])),1)
    inside = np.reshape(inside, (len(inside), 1))
    v = maxx + np.log(inside)
    
    return v

def linearvalueiteration(mdp_data, r):
    v = np.zeros((mdp_data['states'], 1), dtype=float)
    sa_s = copy.copy(mdp_data['sa_s'])
    sa_s = sa_s.astype(np.float64)
    diff = 1.0
    count = 0
    while diff >= 0.0001:
        count +=1 
        vp = copy.copy(v)

        # q = r + mdp_data.discount*sum(mdp_data.sa_p.*vp(mdp_data.sa_s),3);
       
        for i in range(len(vp)): 
            sa_s[mdp_data['sa_s'] == i] = vp[i] #vp(mdp_data.sa_s)

        """
        print("sa_s post transform")
        print('\nsa_s(:,:,0)\n\n{}'.format(sa_s[:,:,0]))
        print('\nsa_s(:,:,1)\n{}'.format(sa_s[:,:,1]))
        print('\nsa_s(:,:,2)\n{}'.format(sa_s[:,:,2]))
        print('\nsa_s(:,:,3)\n{}'.format(sa_s[:,:,3]))
        print('\nsa_s(:,:,4)\n{}'.format(sa_s[:,:,4]))
        """
    


        q = r + mdp_data['discount']* np.sum(mdp_data['sa_p']*sa_s, 2)
        v = maxentsoftmax(q)

        diff = max(abs(v-vp))
       


    #compute policy
    logp = q - np.tile(v,[1,mdp_data['actions']])
    p = np.exp(logp)

    return v, q, logp, p


