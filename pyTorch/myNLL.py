import numpy as np
import scipy.sparse as sps
from linervalueiteration import *
from linearmdpfrequency import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(precision=3)

class myNLL(torch.autograd.Function):

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
    @staticmethod
    def forward(ctx, self, r):
        #ctx.save_for_backward(r)
        #Reformat R
        if(torch.is_tensor(r) == False):
            r = torch.tensor(r)
        if(r.shape != (self.mdp_data['states'],5)):
            #print('Reward in reshape \n{}'.format(r))
            r = torch.reshape(r, (self.mdp_data['states'],1))
            r = r.repeat((1, 5))
        self.initD = torch.reshape(self.initD, (self.mdp_data['states'],1))
        #Solve MDP with current reward
        v, q, logp, p = linearvalueiteration(self.mdp_data, r)
        self.p = p
        #Calculate likelihood from logp
        likelihood = sum(sum(logp*self.mu_sa))
        return -likelihood

    @staticmethod
    def backward(ctx, self, grad_output, weightVector):
        grad_input = grad_output.mm(weightVector)
        return grad_input

    def gradient(self, r):
        #r, F = ctx.saved_tensors
        #Reformat R
        if(torch.is_tensor(r) == False):
            r = torch.tensor(r)
        if(r.shape != (self.mdp_data['states'],5)):
            r = torch.reshape(r, (self.mdp_data['states'],1))
            r = r.repeat((1, 5))

        #Compute state visitation count D
        D = linearmdpfrequency(self.mdp_data,self.p.detach().cpu().numpy(),self.initD.detach().cpu().numpy())
        D = torch.tensor(D)
        #Compute gradient.

        dr = self.muE - torch.matmul(torch.t(self.F),D)
       
        return -dr




    def calculate_EVD(self, trueP, guessR):
        v, q, logp, guessP = linearvalueiteration(self.mdp_data, guessR)
        #EVD = diff in policies since exact True R values never actually learned, only it's structure
        evd=torch.max(torch.abs(guessP-trueP))
        return evd

