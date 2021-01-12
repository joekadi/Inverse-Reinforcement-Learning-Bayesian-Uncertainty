import torch
import numpy as np
import math as math

torch.set_printoptions(precision=5)

def objectworldbuild(mdp_params):
    '''
    % mdp_params - parameters of the objectworld:
%       seed (0) - initialization for random seed
%       n (32) - number of cells along each axis
%       placement_prob (0.05) - probability of placing object in each cell
%       c1 (2) - number of primary "colors"
%       c2 (2) - number of secondary "colors"
%       determinism (1.0) - probability of correct transition
%       discount (0.9) - temporal discount factor to use
% mdp_data - standard MDP definition structure with object-world details:
%       states - total number of states in the MDP
%       actions - total number of actions in the MDP
%       discount - temporal discount factor to use
%       sa_s - mapping from state-action pairs to states
%       sa_p - mapping from state-action pairs to transition probabilities
%       map1 - mapping from states to c1 colors
%       map2 - mapping from states to c2 colors
%       c1array - array of locations by c1 colors
%       c2array - array of locations by c2 colors
% r - mapping from state-action pairs to rewards
    '''
    #Construct the Gridworld MDP structures. 
    torch.manual_seed(mdp_params['seed'])
    np.random.seed(seed=mdp_params['seed'])
    #Build action mappings

    sa_s = torch.zeros((mdp_params['n']**2,5,5), dtype=torch.int8)
    sa_p = torch.zeros((mdp_params['n']**2,5,5), dtype=torch.int8)
    for y in range(mdp_params['n']):
        for x in range(mdp_params['n']):    
                   
            s = y*mdp_params['n']+x
            successors = torch.zeros((1,1,5))
            successors[0,0,0] = s
            #print(successors[0,0,0])
            successors[0,0,1] = ((min(mdp_params['n'],y+2)-1)*mdp_params['n']+x+1)-1
            successors[0,0,2] = ((y)*mdp_params['n']+min(mdp_params['n'],x+2))-1
            successors[0,0,3] = ((max(1,y)-1)*mdp_params['n']+x+1)-1
            successors[0,0,4] = ((y)*mdp_params['n']+max(1,x))-1  
            sa_s[s,:,:] = successors.repeat([1,5,1])
            sa_p[s,:,:] = torch.reshape(torch.eye(5,5)*mdp_params['determinism'] + (torch.ones((5,5)) - torch.eye(5,5)) * ((1.0-mdp_params['determinism'])/4.0), (1,5,5))

    #construct map
    map1 = torch.zeros(mdp_params['n']**2,1)
    map2 = torch.zeros(mdp_params['n']**2,1)
    c1array = []
    c2array = []
    for i in range(int(mdp_params['c1'])):
        c1array.append(torch.rand(1,1))
        c2array.append(torch.rand(1,1))


    #c1array = np.empty((int(mdp_params['c1']),1))
    #c2array = np.empty((int(mdp_params['c1']),1))



    '''
    Place objects in "rounds", with 2 colors each round.
    This ensures, for example, that increasing c1 from 2 to 4 results in all
    of the objects from c1=2 being placed, plus additional "distractor"
    objects. This prevents the situation when c1 is high of not placing any
    objects with c1=1 or c1=2 (which makes the example useless for trying to
    infer any meaningful reward).
    '''


    for round in range(int(mdp_params['c1']*0.5)):
        initc1 = (round-1)*2
        if initc1+1 == mdp_params['c1']:
            #Always choose the leftover c1
            prob = mdp_params['placement_prob']*0.5
            maxc1 = 1
        else:
            #choose from two c1 colours
            prob = mdp_params['placement_prob']
            maxc1 = 1
        
        for s in range(mdp_params['n']**2):
            if(torch.rand(1,1) < prob and map1[s] == 0):
                #place object
                c1 = initc1+math.ceil(torch.rand(1,1)*maxc1)-1
                c2 = math.ceil(torch.rand(1,1)*mdp_params['c2'])-1
                map1[s] = c1
                map2[s] = c2
                c1array[c1] = [c1array[c1], s]
                c2array[c2] = [c2array[c2], s]

    #Create MDP data structure.
    mdp_data = {'states':mdp_params['n']**2.0, 
                'actions':5.0, 
                'discount':mdp_params['discount'], 
                'determinism':mdp_params['determinism'],
                'sa_s':sa_s,
                'sa_p':sa_p,
                'map1':map1,
                'map2':map2,
                'c1array':c1array,
                'c2array':c2array}

    return mdp_data