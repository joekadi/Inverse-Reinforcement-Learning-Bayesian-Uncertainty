import numpy as np
import scipy.sparse as sps
import torch

"""
def linearmdpfrequency(mdp_data,p,initD):
    [states,actions,transitions] = mdp_data['sa_p'].shape
    D = torch.zeros((states,1))
    diff = 1.0
    threshold = 0.001

    while(diff >= threshold):
        Dp = D.clone().detach()
        LHS = torch.mul(p[...,None].repeat((1,1,transitions)), mdp_data['sa_p'])
        RHS0 = Dp[...,None].repeat((1,actions,transitions))
        RHS = RHS0*mdp_data['discount']
        Dpi = torch.mul(LHS, RHS)
        D_CSR = sps.csc_matrix((torch.flatten(Dpi.permute(1,0,2)), torch.flatten(mdp_data['sa_s'].permute(1,0,2)),torch.arange(states*actions*transitions+1)), shape=(states,states*actions*transitions)) 
        D_CSR.eliminate_zeros()
        D_mx = torch.matmul(torch.tensor(D_CSR.todense()), torch.ones(states*actions*transitions,1, dtype=torch.double))
        D_s = torch.sum(D_mx,1)
        D = initD + D_s
        D = torch.reshape(D, (len(D),1))
        diff = torch.max(torch.abs(D-Dp))

    return D
"""


#x.flatten(1) becomes torch.flatten(torch.t(x))

def linearmdpfrequency(mdp_data,p,initD):
    [states,actions,transitions] = mdp_data['sa_p'].size()
    D = torch.zeros(states,1)
    diff = 1.0
    threshold = 0.00001

    while(diff >= threshold):
        Dp = torch.tensor(D.clone().detach().requires_grad_(True), dtype=torch.float64)
        left_hold = torch.tensor(np.tile(p[...,None], (1,1,transitions)))
        LHS = torch.mul(left_hold, mdp_data['sa_p'])
        RHS0 = torch.tensor(np.tile(Dp[...,None], (1,actions,transitions)))
        RHS = RHS0*mdp_data['discount']
        Dpi = torch.mul(LHS, RHS)
        D_CSR = sps.csc_matrix((Dpi.transpose(1,0).flatten(), mdp_data['sa_s'].transpose(1,0).flatten(), torch.arange(states*actions*transitions+1)), shape=(states,states*actions*transitions)) 
        D_CSR.eliminate_zeros()
        D_mx = torch.tensor(D_CSR.todense()) @ torch.ones(states*actions*transitions,1)
        #D_s = torch.sum(D_mx,1)
        D = torch.from_numpy(initD) + D_mx
  

        diff = torch.max(torch.abs(torch.subtract(D,Dp)))

    return D

