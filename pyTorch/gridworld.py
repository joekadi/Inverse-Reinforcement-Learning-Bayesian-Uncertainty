import torch
import numpy as np
torch.set_printoptions(precision=5)

def gridworldbuild(mdp_params):
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

    
    #Create MDP data structure.
    mdp_data = {'states':mdp_params['n']**2, 
                'actions':5, 
                'discount':mdp_params['discount'], 
                'determinism':mdp_params['determinism'],
                'sa_s':sa_s,
                'sa_p':sa_p}

    #Fill in the reward function.
    R_SCALE = 100
    r = torch.zeros((mdp_params['n']**2,5), dtype=torch.float64)
    for yc in range(int(mdp_params['n']/mdp_params['b'])):
        for xc in range(int(mdp_params['n']/mdp_params['b'])):
            #Select a reward for macro-cell
            macro_reward0= (np.random.rand(1,1)**8)*R_SCALE
            macro_reward = torch.tensor(macro_reward0)
            #Assign reward to all state-action pairs in macro-cell.
            ys = yc*mdp_params['b']
            ye = yc*mdp_params['b']+1
            for y in range(ys, ye):
                xs = (xc)*mdp_params['b']
                xe = xc*mdp_params['b']+1
                for x in range(xs, xe):
                    r[y*mdp_params['n']+x, :] = macro_reward.repeat([1,5])

    return mdp_data, r







