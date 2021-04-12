import numpy as np
import scipy.sparse as sps
import sys

def linearmdpfrequency(mdp_data,p,initD):
    [states,actions,transitions] = mdp_data['sa_p'].shape
    D = np.zeros((states,1))
    diff = 1.0
    threshold = 0.001

    while(diff >= threshold):
        Dp = np.array(D, dtype=np.float64)
        LHS = np.multiply(np.tile(p[...,None], (1,1,transitions)), mdp_data['sa_p'])
        RHS0 = np.tile(Dp[...,None], (1,actions,transitions))
        RHS = RHS0*mdp_data['discount']
        Dpi = np.multiply(LHS, RHS)
        D_CSR = sps.csc_matrix((Dpi.flatten(1), mdp_data['sa_s'].flatten(1),np.arange(states*actions*transitions+1)), shape=(states,states*actions*transitions)) 
        D_CSR.eliminate_zeros()
        D_mx = np.matmul(D_CSR.todense(),np.ones((states*actions*transitions,1)))
        D_s = np.sum(D_mx,1)
        D = initD + D_s
        diff = max(abs(D-Dp))

    return D


