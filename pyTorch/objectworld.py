import torch as torch
import numpy as np
import math as math
import random
from scipy import sparse as sps
from gridworld import *

torch.set_printoptions(precision=5)

def objectworldbuild(mdp_params):
    
    '''
    mdp_params - parameters of the objectworld:
    seed (0) - initialization for random seed
    n (16) - number of cells along each axis
    placement_prob (0.05) - probability of placing object in each cell
    c1 (2) - number of primary "colors"
    c2 (2) - number of secondary "colors"
    determinism (1.0) - probability of correct transition
    discount (0.9) - temporal discount factor to use
    mdp_data - standard MDP definition structure with object-world details:
    states - total number of states in the MDP
    actions - total number of actions in the MDP
    discount - temporal discount factor to use
    sa_s - mapping from state-action pairs to states
    sa_p - mapping from state-action pairs to transition probabilities
    map1 - mapping from states to c1 colors
    map2 - mapping from states to c2 colors
    c1array - array of locations by c1 colors
    c2array - array of locations by c2 colors
    r - mapping from state-action pairs to rewards
    '''

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

    #construct map
    map1 = torch.zeros(mdp_params['n']**2,1)
    map2 = torch.zeros(mdp_params['n']**2,1)

    #initalise empty lists
    c1array = []
    c2array = []
    for i in range(int(mdp_params['c1'])):
        c1array.append([])
        c2array.append([])


    '''
    Place objects in "rounds", with 2 colors each round.
    This ensures, for example, that increasing c1 from 2 to 4 results in all
    of the objects from c1=2 being placed, plus additional "distractor"
    objects. This prevents the situation when c1 is high of not placing any
    objects with c1=1 or c1=2 (which makes the example useless for trying to
    infer any meaningful reward).
    '''


    for round in range(int(mdp_params['c1']*0.5)):
        initc1 = (round)*2
        
        if initc1+1 == mdp_params['c1']:
            #Always choose the leftover c1
            prob = mdp_params['placement_prob']*0.5
            
            maxc1 = 1
        else:
            #choose from two c1 colours
            prob = mdp_params['placement_prob']
            maxc1 = 2

        for s in range(mdp_params['n']**2):
            if(torch.rand(1,1) < prob and map1[s] == 0):

                #place object
                c1 = initc1+math.ceil(torch.rand(1,1)*maxc1)-1
                c2 = math.ceil(torch.rand(1,1)*mdp_params['c2'])-1
                map1[s] = c1
                map2[s] = c2
                
                #make copies then append new state
                #for lists
                c1inserter = c1array[c1] 
                c2inserter = c2array[c2] 
                c1inserter.append([s]) 
                c2inserter.append([s]) 
                
                c1array[c1] = c1inserter 
                c2array[c2] = c2inserter

    #convert into list of column vectors
    for arr in c1array:
        index = c1array.index(arr)
        newarr = np.array([arr]).T.tolist()
        c1array[index] = newarr

    #convert into list of column vectors
    for arr in c2array:
        index = c2array.index(arr)
        newarr = np.array([arr]).T.tolist()
        c2array[index] = newarr
    
    

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

    r, feature_data, true_feature_map = objectworldfeatures(mdp_params, mdp_data)

    return mdp_data, r, feature_data, true_feature_map

def objectworldfeatures(mdp_params, mdp_data):

    '''
    mdp_params - definition of MDP domain
    mdp_data - generic definition of domain
    feature_data - generic feature data:
    splittable - matrix of states to features
    stateadjacency - sparse state adjacency matrix
    '''
    
    #Construct adjacency table.
    indptr = np.zeros((int(mdp_data['states'])+1))
    stateadjacency = sps.csc_matrix(([], [], indptr), shape=(int(mdp_data['states']),  int(mdp_data['states'])))
    for s in range(int(mdp_data['states'])):
        for a in range(int(mdp_data['actions'])):
            stateadjacency[s, mdp_data['sa_s'][s,a,0].item()] = 1


    #construct discrete and continous split tables
    splittable = torch.zeros(int(mdp_data['states']), int((mdp_params['n']-1)*(mdp_params['c1']+mdp_params['c2'])))
    splittablecont = torch.zeros(int(mdp_data['states']),int((mdp_params['c1']+mdp_params['c2']))) 

    for s in range(int(mdp_data['states'])):
        #get x and y positions
        if s % 10 == 0:
            y = math.ceil(s/int(mdp_params['n']))
        else:
            y = math.ceil(s/int(mdp_params['n']))-1
        x = s-y*int(mdp_params['n'])

        #Determine distances to each type of object.
        c1dsq = math.sqrt(2*(mdp_params['n'])**2)*torch.ones(int(mdp_params['c1']),1)
        c2dsq = math.sqrt(2*(mdp_params['n'])**2)*torch.ones(int(mdp_params['c2']),1)
        
        for i in range(int(mdp_params['c1'])):
            for j in range(len(mdp_data['c1array'][i])):

                cy = math.ceil(mdp_data['c1array'][i][0][j][0]/mdp_params['n'])
                cx = mdp_data['c1array'][i][0][j][0]-(cy-1)*mdp_params['n']
                d = math.sqrt( (cx-x)**2 + (cy-y)**2)     
                c1dsq[i] = min(c1dsq[i], d)

        for i in range(int(mdp_params['c2'])):
            for j in range(len(mdp_data['c2array'][i])):

                cy = math.ceil(mdp_data['c2array'][i][0][j][0]/mdp_params['n'])
                cx = mdp_data['c2array'][i][0][j][0]-(cy-1)*mdp_params['n']
                d = math.sqrt( (cx-x)**2 + (cy-y)**2)     
                c2dsq[i] = min(c2dsq[i], d)

        #Build corresponding feature table (discrete).
        for d in range(int(mdp_params['n'])-1):
            strt = d * (int(mdp_params['c1'])+int(mdp_params['c2']))
            for i in range(int(mdp_params['c1'])):
                splittable[s,strt+i] = c1dsq[i] < d
            strt = d * (int(mdp_params['c1'])+int(mdp_params['c2']))+int(mdp_params['c1'])
            for i in range(int(mdp_params['c2'])):
                splittable[s,strt+i] = c2dsq[i] < d

        #Build corresponding feature table (continuous).
        splittablecont[s,torch.arange(0,int(mdp_params['c1']))] = c1dsq.view(len(c1dsq))
        splittablecont[s, torch.arange(int(mdp_params['c1']),int(mdp_params['c1'])+int(mdp_params['c2']))] = c2dsq.view(len(c2dsq))
    
    #Return feature data structure.
    feature_data = {
        'stateadjacency': stateadjacency,
        'splittable': splittable
    }

    #If continuous flag True, replace splittable
    if mdp_params['continuous']:
        feature_data['splittable'] = splittablecont

    #Construct true feature map
    fm_indptr = np.zeros((int(mdp_params['r_tree']['total_leaves']+1)))


    true_feature_map = sps.csc_matrix(([], [], fm_indptr), shape=(int(mdp_data['states']),  int(mdp_params['r_tree']['total_leaves'])))
    for s in range(int(mdp_data['states'])):
        #Determine which leaf state belongs to
        leaf, val = cartcheckleaf(mdp_params['r_tree'],s,feature_data)
        true_feature_map[s,leaf] = 1
    #Fill in the reward function.
    R_SCALE = 5
    #print('feature data', feature_data['splittable'][0,:])
    
    r = cartaverage(mdp_params['r_tree'],feature_data)*R_SCALE

    return r, feature_data, true_feature_map

def cartcheckleaf(tree, s, feature_data):
    #Check which leaf of tree contains leaf.

    #Check if this is a leaf.
    if (tree['type'] == 0):
        #Return result.
        leaf = tree['index']
        val = tree['mean']
    else:
        #Recurse.
        if feature_data['splittable'][s,int(tree['test'])] == 0:
            branch = tree['ltTree']
        else:
            branch = tree['gtTree']
        leaf, val = cartcheckleaf(branch, s, feature_data)
    
    return leaf, val

def cartaverage(tree,feature_data):
    #Return average reward for given regression tree
    if (tree['type'] == 0):
        #Simply return the average.
        R = np.tile(tree['mean'], (feature_data['splittable'].shape[0], 1))
        R = torch.tensor(R)
        return R
        
    else:
        #Compute reward on each side.
        ltR = cartaverage(tree['ltTree'],feature_data)
        
        gtR = cartaverage(tree['gtTree'],feature_data)
        #print('gtR', gtR)

    #Combine.

    ind = np.tile(feature_data['splittable'][:, int(tree['test'])].view(len(feature_data['splittable'][:, int(tree['test'])]),1), (1, ltR.shape[1]))
    ind = np.reshape(ind, ltR.size())
    ind = torch.tensor(ind)
    hold= (1-ind)*ltR 
    R = hold + ind*gtR
    return R

#def draw(r,p,g,mdp_params,mdp_data,feature_data,model):

def create_objectworld():
    print("\n ... generating objectworld MDP and intial R ...")
    mdp_params = {'n': 16, 'placement_prob': 0.05, 'c1': 2.0, 'c2': 2.0, 'continuous': False, 'determinism': 1.0, 'discount': 0.9, 'seed': 0, 'r_tree': None}
    step = mdp_params['c1'] + mdp_params['c2']
    
    r_tree = {'type': 1, 'test':1+step*2, 'total_leaves':3,           # Test distance to c1 1 shape
        'ltTree':{'type':0, 'index': 0,'mean':[0,0,0,0,0]},           # Neutral reward for being elsewhere
        'gtTree': {'type':1,'test':2+step*1,'total_leaves':2,         # Test distance to c1 2 shape
            'gtTree':{'type':0, 'index':1,'mean':[1,1,1,1,1]},        # Reward for being close
            'ltTree':{'type':0,'index':2,'mean':[-2,-2,-2,-2,-2]}}}   # Penalty otherwise
   
    mdp_params['r_tree'] = r_tree
   
    mdp_data, r, feature_data, true_feature_map = objectworldbuild(mdp_params)
    print("\n... done ...")
    return mdp_data, r, feature_data, true_feature_map, mdp_params

def owVisualise(test_result):

    # Compute visible examples.
    Eo = torch.zeros(int(test_result['mdp_data']['states']),1)
    for i in range(len(test_result['example_samples'][0])):
        for t in range(len(test_result['example_samples'][0][0])):
            Eo[test_result['example_samples'][0][i][t][0]] = 1
    g = torch.ones(int(test_result['mdp_data']['states']),1)*0.5+Eo*0.5

    #Create figure.
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
        
    #Draw reward for ground truth.
    objectworlddraw(test_result['true_r'],test_result['mdp_solution']['p'],g,test_result['mdp_params'],test_result['mdp_data'], f, ax1)

    # Draw reward for IRL result.
    objectworlddraw(test_result['irl_result']['r'],test_result['irl_result']['p'],g,test_result['mdp_params'],test_result['mdp_data'], f, ax2)
    
    # Draw uncertainty for IRL result
    objectworlddraw(test_result['irl_result']['uncertainty'],test_result['irl_result']['p'],g,test_result['mdp_params'],test_result['mdp_data'], f, ax3)

    ax1.set_title(test_result['irl_result']['truth_figure_title'])
    ax2.set_title(test_result['irl_result']['pred_reward_figure_title'])
    ax3.set_title(test_result['irl_result']['uncertainty_figure_title'])

    plt.show()

def objectworlddraw(r,p,g,mdp_params, mdp_data, f, ax):

    #Use gridworld drawing function to draw paths and reward function
    gridworlddraw(r,p,g,mdp_params, mdp_data, f, ax)

    #Initialize colors.
    shapeColours = plt.cm.jet( np.linspace(0,1, int(mdp_params['c1']+mdp_params['c2']))   )    

    n = mdp_params['n']
    if isinstance(g, list):
        #This means p is crop
        crop = p
    else:
        crop = np.array([[1, n], [1,n]])

    #cast to np array for easier indexing
    np_c1array = np.array(mdp_data['c1array'], dtype=object)

    #Draw objects
    for i in range(1, len(mdp_data['c1array'])+1):
        for j in range(1, len(np_c1array[i-1][0])+1):
            #Get colours and position of object
            s = mdp_data['c1array'][i-1][0][j-1][0]
            c1 = i-1
            c2 = int(mdp_data['map2'][s][0].item())
            y = math.ceil(s/n)
            x = s-(y-1)*n

            if x<crop[0,0] or x>crop[0,1] or y<crop[1,0] or y > crop[1,1]:
                continue

            x = x-crop[0,0]+1
            y = y-crop[1,0]+1

            #Draw the object
            circle = patches.FancyBboxPatch(xy = (x-0.55,y-0.55),width=0.000000008,height=0.000000008, boxstyle='circle', facecolor=shapeColours[int(mdp_params['c1']+c2),:], edgecolor=shapeColours[int(c1),:], linewidth=2.0) #create rectangle patches
            ax.add_patch(circle) # Add the patch to the Axes
