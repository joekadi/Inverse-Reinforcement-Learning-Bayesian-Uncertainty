import torch
import numpy as np
from scipy import sparse as sps
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


torch.set_printoptions(precision=5)

def gridworldbuild(mdp_params):
    #Construct the Gridworld MDP structures. 
    torch.manual_seed(mdp_params['seed'])
    np.random.seed(seed=mdp_params['seed'])
    #Build action mappings
    sa_s = torch.zeros((mdp_params['n']**2,5,5), dtype=torch.int64)
    sa_p = torch.zeros((mdp_params['n']**2,5,5), dtype=torch.int64)
    for y in range(mdp_params['n']):
        for x in range(mdp_params['n']):    
                
            s = y*mdp_params['n']+x
            successors = torch.zeros((1,1,5))
            successors[0,0,0] = s
            successors[0,0,1] = ((min(mdp_params['n'],y+2)-1)*mdp_params['n']+x+1)-1
            successors[0,0,2] = ((y)*mdp_params['n']+min(mdp_params['n'],x+2))-1
            successors[0,0,3] = ((max(1,y)-1)*mdp_params['n']+x+1)-1
            successors[0,0,4] = ((y)*mdp_params['n']+max(1,x))-1  
            sa_s[s,:,:] = torch.tensor(np.tile(successors, (1,5,1)))
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
    for yc in range( int(mdp_params['n']/mdp_params['b'])    ):
        for xc in range(int(mdp_params['n']/mdp_params['b'])):
            
        
            #Select a reward for macro-cell
            macro_reward0= (np.random.rand(1,1)**8)*R_SCALE
            macro_reward = torch.tensor(macro_reward0)

            #Assign reward to all state-action pairs in macro-cell.
            for y in range(yc*mdp_params['b'], (yc+1)*mdp_params['b']):
                for x in range(xc*mdp_params['b'], (xc+1)*mdp_params['b']):
                    r[(y-1)*mdp_params['n']+x, :] = torch.tensor(np.tile(macro_reward, (1,5)))
                
    feature_data = gridworldfeatures(mdp_params,mdp_data)
    return mdp_data, r, feature_data

def gridworldfeatures(mdp_params, mdp_data):

    #Construct adjacency table.
    indptr = np.zeros((int(mdp_data['states'])+1))
    stateadjacency = sps.csc_matrix(([], [], indptr), shape=(int(mdp_data['states']),  int(mdp_data['states'])))
    for s in range(int(mdp_data['states'])):
        for a in range(int(mdp_data['actions'])):
            stateadjacency[s, mdp_data['sa_s'][s,a,0].item()] = 1

    #Construct split table
    splittable = torch.zeros(int(mdp_data['states']), int((mdp_params['n']-1)*2), dtype=torch.double)
    for y in range(mdp_params['n']):
        for x in range(mdp_params['n']):
            #Compute x and y split tables
            #xtable = torch.hstack( (torch.zeros(1,x), torch.ones(1, mdp_params['n']-(x+1))))
            xtable = np.hstack( (np.zeros((1,x)), np.ones((1, mdp_params['n']-(x+1)))) )
            
            #ytable = torch.hstack((torch.zeros(1,y), torch.ones(1, mdp_params['n']-(y+1))))
            ytable = np.hstack((np.zeros((1,y)), np.ones((1, mdp_params['n']-(y+1)))))
           
            hold = np.hstack((xtable, ytable))
            hold = torch.tensor(hold)
            
            splittable[(y)*mdp_params['n']+x, :] = hold

    #Return feature data structure.
    feature_data = {
        'stateadjacency': stateadjacency,
        'splittable': splittable
    }

    return feature_data

def gwVisualise(test_result):

    # Compute visible examples.
    Eo = torch.zeros(test_result['mdp_data']['states'],1)
    for i in range(len(test_result['example_samples'][0])):
        for t in range(len(test_result['example_samples'][0][0])):
            Eo[test_result['example_samples'][0][i][t][0]] = 1
    g = torch.ones(test_result['mdp_data']['states'],1)*0.5+Eo*0.5

    #Create figure.
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
    #Draw reward for ground truth.
    gridworlddraw(test_result['true_r'],test_result['mdp_solution']['p'],g,test_result['mdp_params'],test_result['mdp_data'], f, ax1)

    # Draw reward for IRL result.
    gridworlddraw(test_result['irl_result']['r'],test_result['irl_result']['p'],g,test_result['mdp_params'],test_result['mdp_data'], f, ax2)
    ax1.set_title('Ground Truth Reward & Policy')
    ax2.set_title('Estimated Reward & Policy')
    plt.show()

def gridworlddraw(r,p,g,mdp_params, mdp_data, f, ax):

    #Set up the axes.
    n = mdp_params['n']

    maxr = torch.max(torch.max(r))
    minr = torch.min(torch.min(r))
    rngr = maxr-minr
    rngr = rngr.item()

    if isinstance(g, list):
        crop = p
    else:
        crop = np.array([[1, n], [1,n]])

    ax.set_xlim(0, crop[0,1]-crop[0,0]+1)
    ax.set_ylim(0, crop[1,1]-crop[1,0]+1)

    if isinstance(g, list):
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xticks(np.arange(0, crop[0,1]-crop[0,0]+2))
        ax.set_yticks(np.arange(0, crop[1,1]-crop[1,0]+2))

    #daspect[1 1 1]

    #Draw the reward function
    for y in range(crop[1,0], crop[1,1]+1):
        for x in range(crop[0,0], crop[0,1]+1):
            
            if rngr == 0:
                v = 0.0
            else:
                v = (torch.mean( r[(y-2)*n+x,:])-minr)/rngr
                v = v.item()
            colour = np.array([v,v,v])
            colour = np.minimum( np.ones((1,3))   ,  np.maximum(np.zeros((1,3)) , colour))
            rect = patches.Rectangle(xy = (x-crop[0,0],y-crop[1,0]),width=1,height=1,facecolor=colour[0]) #create rectangle patches
            ax.add_patch(rect) # Add the patch to the Axes

    if isinstance(g, list):
        print('g contains example traces - just draw those.')
        print('needs implementeed')
    else: 
        #Convert p to action mode.
        if p.size()[1] != 1:
            #max(a,[],2) = a.max(1)
            p = torch.argmax(p, dim=1)
            p = p.view(len(p), 1)

        #Draw paths
        for y in range(1,n+1):
            for x in range(1,n+1):
                s = ((y-1)*n+x)-1
                a = p[s].item()
                gridworlddrawagent(x,y,a,np.array([g[s].item(),g[s].item(),g[s].item()]), f, ax)
        
def gridworlddrawagent(x,y,a,colour,f,ax):
    overlap = 0
    w = 1+(overlap)*0.5
    if (a == 0):
       circle = patches.FancyBboxPatch(xy = (x-0.6,y-0.6),width=0.2,height=0.2, boxstyle='circle', facecolor=colour, linewidth= w) #create circle patches
       ax.add_patch(circle) # Add the patch to the Axes
    else:
        if a == 4:
            nx = x-1
            ny = y
        elif a == 3:
            nx = x
            ny = y-1
        elif a == 2:
            nx = x+1
            ny=y
        elif a == 1:
            nx = x
            ny = y+1
        vec = np.array([[(nx-x)*0.25],[(ny-y)*0.25]])
        norm1 = np.array([vec[1], -vec[0]])
        norm2 = -norm1
        xv = np.array([x-0.5+vec[0], x-0.5+norm1[0], x-0.5+norm2[0]])
        yv = np.array([y-0.5+vec[1], y-0.5+norm1[1], y-0.5+norm2[1]])
        xv = np.append(xv, xv[0])
        yv = np.append(yv, yv[0])
        ax.fill(xv, yv, colour, linewidth=w)

def create_gridworld():
    print("\n... generating gridworld MDP and intial R ...")
    #mdp_params = {'n': 32, 'b': 4, 'determinism': 1.0, 'discount': 0.9, 'seed': 0}
    #mdp_params = {'n': 16, 'b': 2, 'determinism': 1.0, 'discount': 0.99, 'seed': 0}
    mdp_params = {'n': 8, 'b': 1, 'determinism': 1.0, 'discount': 0.99, 'seed': 0}
    mdp_data, r, feature_data = gridworldbuild(mdp_params)
    print("\n... done ...")
    return mdp_data, r, feature_data, mdp_params