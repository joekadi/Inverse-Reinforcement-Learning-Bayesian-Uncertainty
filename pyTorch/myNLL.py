import numpy as np
from linervalueiteration import *
from linearmdpfrequency import *
import torch
from torch.autograd import Variable
import torch.nn as nn
from gridworld import *
torch.set_printoptions(precision=3)

class myNLL(nn.Module):

    #make class variables to enable access ctx
    mdp_data = {} 
    N, T  = 0,0 
    example_samples = []
    F = None
    muE = None
    mu_sa = None
    initD = None
    p = None

    def calc_var_values(self, mdp_data, N, T, example_samples):
        self.mdp_data = mdp_data
        self.N = N
        self.T = T
        self.example_samples = example_samples
        self.transitions = len(self.mdp_data['sa_p'][0][0])
        
        
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
        self.F = torch.eye(self.mdp_data['states'], dtype=torch.float64)
        features = self.F.shape[2-1]
        #print('Features {}'.format(self.F))

        ex_s = torch.zeros((self.N,self.T))
        ex_a = torch.zeros((self.N,self.T))
        self.muE = torch.zeros((features,1), dtype=torch.float64)


        self.mu_sa = torch.zeros((self.mdp_data['states'],self.mdp_data['actions']))

        for i in range(self.N):
            for t in range(self.T):
                ex_s[i,t] = self.example_samples[i][t][0]
                ex_a[i,t] = self.example_samples[i][t][1]
                self.mu_sa[int(ex_s[i,t]),int(ex_a[i,t])] = self.mu_sa[int(ex_s[i,t]),int(ex_a[i,t])] + 1.0 #maybe minus 1 since numpy index -1 to matlab
                state_vec = torch.zeros((self.mdp_data['states'],1), dtype=torch.float64)
                state_vec[int(ex_s[i,t])] = 1.0
                self.muE = self.muE + torch.matmul(torch.t(self.F),state_vec)
                            
        #initlaise params to construct sparse matrix
        ex_s_reshaped = torch.flatten(torch.t(ex_s)) #flatten in column order
        ex_s_reshaped = ex_s_reshaped.type(torch.int) #cast to int since indicies array
        po = torch.arange(self.N*self.T+1)
        Rones = torch.ones((self.N*self.T))
        ones = torch.ones((self.N*self.T,1))
     
        #Generate initial state distribution for infinite horizon.
        initD_CSR = sps.csc_matrix((Rones, ex_s_reshaped, po), shape=(self.mdp_data['states'],self.T*self.N))
        initD_CSR.eliminate_zeros() 
        initD_mx = torch.matmul(torch.tensor(initD_CSR.todense()), ones)
        self.initD = torch.sum(initD_mx,1)

        for i in range(self.N):
            for t in range(self.T):
                s = int(ex_s[i,t])
                a = int(ex_a[i,t])
                for k in range(self.transitions):
                    sp = self.mdp_data['sa_s'][s,a,k]
                    self.initD[sp] = self.initD[sp] - self.mdp_data['discount']*self.mdp_data['sa_p'][s,a,k]
        '''
        print('mu_sa \n{}\n'.format(self.mu_sa))
        print('muE \n{}\n'.format(self.muE))
        print('initD \n{}\n'.format(self.initD))
        print('F \n{}\n'.format(self.F))
        print('Features \n{}\n'.format(features))
        '''

    def forward(self, r):
        print('Init D {}'.format(initD))
        #Returns NLL w.r.t input r
        r = self.reshapeReward(r)
        self.initD = torch.reshape(self.initD, (self.mdp_data['states'],1)) 
        v, q, logp, p = linearvalueiteration(self.mdp_data, r) #Solve MDP with current reward
        self.p = p #set policy w.r.t current r for backward

        #Calculate likelihood from logp
        #torch.sum
        likelihood = sum(sum(logp*self.mu_sa)) #for scalar likelihood

        #mul = logp*self.mu_sa #hold
        #likelihood = torch.sum(mul, dim=1)#likelihood for each state as tensor size (states,1)
        #likelihood = likelihood.view(self.mdp_data['states'],1) #make column vector
        #likelihood.requires_grad = True
        return -likelihood


    def backward(self, r):
        #Should return as many gradient te w.r.t to inputs
        r = self.reshapeReward(r)
        D = linearmdpfrequency(self.mdp_data,self.p.detach().cpu().numpy(),self.initD.detach().cpu().numpy())#Compute state visitation count D
        D = torch.tensor(D) #Cast to tensor
        dr = self.muE - torch.matmul(torch.t(self.F),D) #Compute gradient
        dr = dr.view(len(dr)) #Make row vector
        print('-dr inside myNLL.backward {}'.format(-dr))
        return -dr #+dr, return -dr for descent 

    def reshapeReward(self, r):
        #Reshapes R to be in format (states,actions)
        if(torch.is_tensor(r) == False):
            r = torch.tensor(r) #cast to tensor
        if(r.shape != (self.mdp_data['states'],5)):
            #reformat to be in shape (states,actions)
            r = torch.reshape(r, (self.mdp_data['states'],1))
            r = r.repeat((1, 5))
        return r 

    def calculate_EVD(self, trueP):
        #Expected Value Diff = diff in policies since exact True R values never actually learned, only it's structure
        evd=torch.max(torch.abs(self.p-trueP))
        return evd



