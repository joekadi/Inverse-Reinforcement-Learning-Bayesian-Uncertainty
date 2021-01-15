import numpy as np
from linervalueiteration import *
from linearmdpfrequency import *
import torch
from torch.autograd import Variable
import torch.nn as nn
from gridworld import *
from scipy import sparse as sps
torch.set_printoptions(precision=3)

class NLLFunction(torch.autograd.Function):

    mdp_data = {} 
    N, T  = 0,0 
    example_samples = []
    F = None
    muE = None
    mu_sa = None
    initD = None
    p = None

    def calc_var_values(self, mdp_data, N, T, example_samples):
        mdp_data = mdp_data
        N = N
        T = T
        example_samples = example_samples
        transitions = len(mdp_data['sa_p'][0][0])
        
        '''
        #set mu_sa & initD = matlab to test with same sampled paths
        #for normal run keep below code commented
        
        self.mu_sa = torch.tensor([[1  , 260   ,  1  ,   0   ,  0],
                        [1  , 253  ,   0   ,  1   ,  0],
                        [906 ,  843  , 931   ,  2  , 928],
                       [991  , 968  , 972  ,   3 ,  939]], dtype=torch.float64)

        self.muE = torch.tensor([[262.],
                        [255.],
                        [3610.],
                        [3873.]], dtype=torch.float64)

        self.initD = torch.tensor([[259.0300],
                                [249.0600],
                                [-227.2400],
                                [-200.8500]], dtype=torch.float64)

        self.F = torch.tensor([ [1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]], dtype=torch.float64)
        '''
        
        
        #Compute feature expectations.
        F = torch.eye(int(mdp_data['states']), dtype=torch.float64)
        features = F.shape[2-1]
        #print('Features {}'.format(self.F))

        ex_s = torch.zeros((N,T))
        ex_a = torch.zeros((N,T))
        muE = torch.zeros((features,1), dtype=torch.float64)


        mu_sa = torch.zeros((int(mdp_data['states']),int(mdp_data['actions'])))

        for i in range(N):
            for t in range(T):
                ex_s[i,t] = example_samples[i][t][0]
                ex_a[i,t] = example_samples[i][t][1]
                mu_sa[int(ex_s[i,t]),int(ex_a[i,t])] = mu_sa[int(ex_s[i,t]),int(ex_a[i,t])] + 1.0 #maybe minus 1 since numpy index -1 to matlab
                state_vec = torch.zeros((int(mdp_data['states']),1), dtype=torch.float64)
                state_vec[int(ex_s[i,t])] = 1.0
                muE = muE + torch.matmul(torch.t(F),state_vec)
                            
        #initlaise params to construct sparse matrix
        ex_s_reshaped = torch.flatten(torch.t(ex_s)) #flatten in column order
        ex_s_reshaped = ex_s_reshaped.type(torch.int) #cast to int since indicies array
        po = torch.arange(N*T+1)
        Rones = torch.ones((N*T))
        ones = torch.ones((N*T,1))
     
        #Generate initial state distribution for infinite horizon.
        initD_CSR = sps.csc_matrix((Rones, ex_s_reshaped, po), shape=(int(mdp_data['states']),T*N))
        initD_CSR.eliminate_zeros() 
        initD_mx = torch.matmul(torch.tensor(initD_CSR.todense()), ones)
        initD = torch.sum(initD_mx,1)

        for i in range(N):
            for t in range(T):
                s = int(ex_s[i,t])
                a = int(ex_a[i,t])
                for k in range(transitions):
                    sp = mdp_data['sa_s'][s,a,k]
                    initD[sp] = initD[sp] - mdp_data['discount']*mdp_data['sa_p'][s,a,k]
        '''
        print('mu_sa \n{}\n'.format(self.mu_sa))
        print('muE \n{}\n'.format(self.muE))
        print('initD \n{}\n'.format(self.initD))
        print('F \n{}\n'.format(self.F))
        print('Features \n{}\n'.format(features))
        '''
        self.initD = torch.reshape(initD, (int(mdp_data['states']),1))
        initD = torch.reshape(initD, (int(mdp_data['states']),1)) 

        self.mu_sa = mu_sa
        self.muE = muE
        self.F = F
        self.mdp_data = mdp_data

        return initD, mu_sa, muE, F, mdp_data

    @staticmethod
    def forward(ctx, r, initD, mu_sa, muE, F, mdp_data):
        #Returns NLL w.r.t input r

        ctx.save_for_backward(r, initD, mu_sa, muE, F, mdp_data['sa_p'], mdp_data['sa_s'], torch.tensor(mdp_data['states']), torch.tensor(mdp_data['actions']), torch.tensor(mdp_data['discount']), torch.tensor(mdp_data['determinism']))
        if(torch.is_tensor(r) == False):
            r = torch.tensor(r) #cast to tensor
        if(r.shape != (mdp_data['states'],5)):
            #reformat to be in shape (states,actions)
            r = torch.reshape(r, (int(mdp_data['states']),1))
            r = r.repeat((1, 5))

        #Solve MDP with current reward
        v, q, logp, p = linearvalueiteration(mdp_data, r) 
   
        #Calculate likelihood from logp
        likelihood = torch.empty(mu_sa.shape, requires_grad=True)
        likelihood = torch.sum(torch.sum(logp*mu_sa)) #for scalar likelihood

        #LH for each state as tensor size (states,1)
        #mul = logp*mu_sa #hold
        #likelihood = torch.sum(mul, dim=1)
        #likelihood.requires_grad = True
        return -likelihood

    @staticmethod
    def backward(ctx, grad_output):
        #print('Grad output {}'.format(grad_output))
        #Should return as many gradient tesnors w.r.t to inputs

        r, initD, mu_sa, muE, F, sa_p, sa_s, states, actions, discount, determinism = ctx.saved_tensors

        #reconsrtuct mdp_data since ctx can't save dicts
        mdp_data = {
            'states':states.item(), 
            'actions':actions.item(), 
            'discount':discount.item(), 
            'determinism':determinism.item(),
            'sa_s':sa_s,
            'sa_p':sa_p
            }


        if(torch.is_tensor(r) == False):
            r = torch.tensor(r) #cast to tensor
        if(r.shape != (mdp_data['states'],5)):
            #reformat to be in shape (states,actions)
            r = torch.reshape(r, (int(mdp_data['states']),1))
            r = r.repeat((1, 5))

        #Solve MDP with current reward
        v, q, logp, p = linearvalueiteration(mdp_data, r) 
        

        #Calc gradient w.r.t to forward inputs 
        D = grad_output.clone()
        D = linearmdpfrequency(mdp_data,p.detach().cpu().numpy(),initD.detach().cpu().numpy())#Compute state visitation count D
        D = torch.tensor(D) #Cast to tensor
        dr = muE - torch.matmul(torch.t(F),D) #Compute gradient
        dr = dr.view(len(dr)) #Make row vector
        return -dr, None, None, None, None, None #+dr, return -dr for descent 

    def calculate_EVD(self, trueP, currR):
        v, q, logp, currP = linearvalueiteration(self.mdp_data, currR.view(int(self.mdp_data['states']),1))
        #Expected Value Diff = diff in policies since exact True R values never actually learned, only it's structure
        evd=torch.max(torch.abs(currP-trueP))
        return evd



