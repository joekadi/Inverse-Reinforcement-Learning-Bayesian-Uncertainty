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
        diff = max(abs(D-Dp))

    return D
"""


#x.flatten(1) becomes torch.flatten(torch.t(x))

def linearmdpfrequency(mdp_data,p,initD):
    [states,actions,transitions] = mdp_data['sa_p'].detach().cpu().numpy().shape
    D = np.zeros((states,1))
    diff = 1.0
    threshold = 0.001

    while(diff >= threshold):
        Dp = np.array(D, dtype=np.float64)
        LHS = np.multiply(np.tile(p[...,None], (1,1,transitions)), mdp_data['sa_p'].detach().cpu().numpy())
        #print('LHS \n {}'.format(LHS))
        RHS0 = np.tile(Dp[...,None], (1,actions,transitions))
        #print('RHS0 \n {}'.format(RHS0))
        RHS = RHS0*mdp_data['discount']

        Dpi = np.multiply(LHS, RHS)
        D_CSR = sps.csc_matrix((Dpi.flatten(1), mdp_data['sa_s'].detach().cpu().numpy().flatten(1),np.arange(states*actions*transitions+1)), shape=(states,states*actions*transitions)) 
        D_CSR.eliminate_zeros()
        D_mx = np.matmul(D_CSR.todense(),np.ones((states*actions*transitions,1)))
        D_s = np.sum(D_mx,1)
        D = initD + D_s
        diff = np.max(np.abs(np.subtract(D,Dp)))

    return D