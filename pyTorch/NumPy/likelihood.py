import numpy as np
import scipy.sparse as sps
from linervalueiteration import *
from linearmdpfrequency import *


class Likelihood:
    mdp_data = {} 
    N, T  = 0,0 
    example_samples = []
    F = None
    muE = None
    mu_sa = None
    initD = None

    def set_variables_for_likehood(self, mdp_data, N, T, example_samples):
        self.mdp_data = mdp_data
        self.N = N
        self.T = T
        self.example_samples = example_samples
        self.transitions = len(mdp_data['sa_p'][0][0])

        #uncomment below to set mu_sa, muE, initD & f = matlab in order to test with same sampled paths
        
        self.mu_sa = np.array([[1  , 260   ,  1  ,   0   ,  0],
                        [1  , 253  ,   0   ,  1   ,  0],
                        [906 ,  843  , 931   ,  2  , 928],
                       [991  , 968  , 972  ,   3 ,  939]])

        self.muE = np.array([[262.],
                        [255.],
                        [3610.],
                        [3873.]])
        self.initD = np.array([[259.0300],
                                [249.0600],
                               [-227.2400],
                               [ -200.8500]])
        self.F = np.array([ [1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])

        #uncomment below to calculate initD for this
        '''
        #Compute feature expectations.
        self.F = np.eye(self.mdp_data['states'])
        features = self.F.shape[2-1]

        ex_s = np.zeros((self.N,self.T))
        ex_a = np.zeros((self.N,self.T))
        self.muE = np.zeros((features,1))


        self.mu_sa = np.zeros((self.mdp_data['states'],self.mdp_data['actions']))

        for i in range(self.N):
            for t in range(self.T):
                ex_s[i,t] = self.example_samples[i][t][0]
                ex_a[i,t] = self.example_samples[i][t][1]
                self.mu_sa[int(ex_s[i,t]),int(ex_a[i,t])] = self.mu_sa[int(ex_s[i,t]),int(ex_a[i,t])] + 1.0 #maybe minus 1 since np index -1 to matlab
                state_vec = np.zeros((self.mdp_data['states'],1))
                state_vec[int(ex_s[i,t])] = 1.0
                self.muE = self.muE + np.dot(self.F.T,state_vec)

        #initlaise params to construct sparse matrix
        print(ex_s)
        ex_s_reshaped = ex_s.flatten(1)
        print(ex_s_reshaped)
        print(ex_s_reshaped.shape)
        ex_s_reshaped.astype(int) #cast to int since indicies array
        po = np.arange(self.N*self.T+1)
        Rones = np.ones((self.N*self.T))
        ones = np.ones((self.N*self.T,1))
     
        #Generate initial state distribution for infinite horizon.
        initD_CSR = sps.csc_matrix((Rones, ex_s_reshaped, po), shape=(self.mdp_data['states'],self.T*self.N))
        initD_CSR.eliminate_zeros() 
        initD_mx = np.matmul(initD_CSR.todense(), ones)
        self.initD = np.sum(initD_mx,1)

        for i in range(self.N):
            for t in range(self.T):
                s = int(ex_s[i,t])
                a = int(ex_a[i,t])
                for k in range(self.transitions):
                    sp = self.mdp_data['sa_s'][s,a,k]
                    self.initD[sp] = self.initD[sp] - self.mdp_data['discount']*self.mdp_data['sa_p'][s,a,k]
        '''

    def negated_likelihood(self, r):
        

        

        #Reshape R to expected
        if(np.shape(r) != (4,5)):
            r = np.reshape(r, (4,1))
            r = np.tile(r, (1, 5))
            
        #Solve MDP with current reward
        v, q, logp, p = linearvalueiteration(self.mdp_data, r)

        #Calculate likelihood from logp
        likelihood = sum(sum(logp*self.mu_sa))
        print(likelihood)
        '''
        D = linearmdpfrequency(self.mdp_data,p,self.initD) 

        #Compute gradient.
        dr = self.muE - np.matmul(self.F.T,D)
        '''
        return -likelihood#, -dr

    
    def calc_gradient(self, r):


        if(np.shape(r) != (4,5)):
            r = np.reshape(r, (4,1))
            r = np.tile(r, (1,5))

        #Solve MDP with current reward
        v, q, logp, p = linearvalueiteration(self.mdp_data, r)

        #Compute state visitation count D
        D = linearmdpfrequency(self.mdp_data,p,self.initD) 

        #Compute gradient.
        dr = self.muE - np.matmul(self.F.T,D)
       
        #return -dr for descent
        #print('Gradient \n{}'.format(-dr))
        return -dr

    def negated_likelihood_with_gradient(self, r):
        
        #Reshape R to expected
        if(np.shape(r) != (4,5)):
            r = np.reshape(r, (4,1))
            r = np.tile(r, (1, 5))
            
        #Solve MDP with current reward
        v, q, logp, p = linearvalueiteration(self.mdp_data, r)

        #Calculate likelihood from logp
        likelihood = sum(sum(logp*self.mu_sa))
        print(likelihood)
        
        D = linearmdpfrequency(self.mdp_data,p,self.initD) 

        #Compute gradient.
        dr = self.muE - np.matmul(self.F.T,D)
        
        return -likelihood, -dr

    
