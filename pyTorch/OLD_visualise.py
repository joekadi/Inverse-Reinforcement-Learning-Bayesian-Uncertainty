import torch 
import torch.nn as nn
from torch.nn.functional import softplus
import torch.nn.functional as Functional
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import gradcheck
from torch.autograd import Variable
import torch
from torch.utils.data import random_split
import os
from sklearn.preprocessing import MinMaxScaler
import sys
from NLLFunction import *
from gridworld import *
from objectworld import *
from linearvalueiteration import *
import pprint
import numpy as np
import pandas as pd
import time
import math as math
import random
import torchvision
import torchvision.transforms as transforms
from clearml import Task
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from torch.utils.tensorboard import SummaryWriter
import pickle
import pytorch_lightning as pl

tensorboard_writer = SummaryWriter('./tensorboard_logs')
torch.set_printoptions(precision=5, sci_mode=False, threshold=1000)
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":

    features_corrupted = ['20 Percent of Features Randomized', '40 Percent of Features Randomized', '60 Percent of Features Randomized', '80 Percent of Features Randomized']
    num_states_to_remove = [np.arange(0, 32, 1), np.arange(40, 100, 1), np.arange(100, 200, 1), np.arange(200,255,1)]
    states_to_remove = ['States 0-32 Removed From Paths', 'States 40-100 Removed From Paths', 'States 100-200 Removed From Paths', 'States 200-255 Removed From Paths']
    
    # Get which experiment
    if len(sys.argv) > 1:
        experiment_type = str(sys.argv[1])
        print('\n... got experiment type from cmd line ...\n')
    else:
        raise Exception('\n... no experiment type provided ...\n')
    
    # Validate input
    if experiment_type not in ['regular', 'noisey_paths', 'noisey_features']:
         raise Exception('\n...invalid experiment type provided. Expected one of: 1. regular, 2. noisey_paths, 3. noisey_features ...\n')

    # Load variables from main
    open_file = open("NNIRL_param_list.pkl", "rb")
    NNIRL_param_list = pickle.load(open_file)
    open_file.close()
    threshold = NNIRL_param_list[0]
    optim_type = NNIRL_param_list[1]
    net = NNIRL_param_list[2]
    initD = NNIRL_param_list[3]
    mu_sa = NNIRL_param_list[4]
    muE = NNIRL_param_list[5]
    mdp_data = NNIRL_param_list[6]
    truep = NNIRL_param_list[7] 
    NLL_EVD_plots = NNIRL_param_list[8]
    example_samples = NNIRL_param_list[9]
    mdp_params = NNIRL_param_list[10] 
    r = NNIRL_param_list[11] 
    mdp_solution = NNIRL_param_list[12] 
    feature_data = NNIRL_param_list[13] 
    trueNLL = NNIRL_param_list[14]
    normalise = NNIRL_param_list[15]
    user_input = NNIRL_param_list[16]
    worldtype = NNIRL_param_list[17]

    # Compute visible examples.
    Eo = torch.zeros(int(mdp_data['states']),1)
    for i in range(len(example_samples[0])):
        for t in range(len(example_samples[0][0])):
            Eo[[example_samples][0][i][t][0]] = 1
    g = torch.ones(int(mdp_data['states']),1)*0.5+Eo*0.5

    #Get directory for results
    RESULTS_PATH = "./" +experiment_type+ "/results/"

    #Create path for graphs 
    GRAPHS_PATH = RESULTS_PATH + "/graphs/"
    for path in [GRAPHS_PATH]:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    if experiment_type == "regular":

        #Init results dict - real
        results = {'ow': {
                      12: {0.0: None, 0.2: None, 0.4: None, 0.8: None}, 
                      24 : {0.0: None, 0.2: None, 0.4: None, 0.8: None}, 
                      48: {0.0: None, 0.2: None, 0.4: None, 0.8: None},
                      64: {0.0: None, 0.2: None, 0.4: None, 0.8: None}
                     },
                'gw': {
                      12: {0.0: None, 0.2: None, 0.4: None, 0.8: None}, 
                      24 : {0.0: None, 0.2: None, 0.4: None, 0.8: None}, 
                      48: {0.0: None, 0.2: None, 0.4: None, 0.8: None},
                      64: {0.0: None, 0.2: None, 0.4: None, 0.8: None}
                      }
            }
        

        #Get lists of variants
        #worlds = results.keys()
        worlds = ['ow']
        paths = results['ow'].keys()
        dropout_vals = results['ow'][12].keys()

        #Populate results dict
        for world in worlds:
            for no_paths in paths:
                for dropout_val in dropout_vals:
                    file_name = RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results.pkl'
                    open_file = open(file_name, "rb")
                    open_results = pickle.load(open_file)
                    open_file.close()

                    result_values = {
                    'y_mc_relu': open_results[0], 
                    'y_mc_std_relu': open_results[1],
                    'y_mc_relu_reward': open_results[2],
                    'y_mc_relu_v': open_results[3],
                    'y_mc_relu_P': open_results[4],
                    'y_mc_relu_q': open_results[5],
                    'evd': open_results[6],
                    'run_time': open_results[7],
                    'num_preds': open_results[8]
                    }

                    results[str(world)][no_paths][dropout_val] = result_values


        optimal_paths = 48
        optimal_p_value = 0.0

        #Plot 1: True R and Optimal R

        #Print EVD's
        evds = []
        for path in paths:
            for p_value in dropout_vals:
                evd = results['ow'][path][p_value]['evd']
                print('Paths= ' + str(path) + ' P= ' + str(p_value) + ' EVD = ' + str(evd.item()))

        #Get optimal IRL results
        irl_result_relu = { 
            'r': results['ow'][optimal_paths][optimal_p_value]['y_mc_relu_reward'],
            'v': results['ow'][optimal_paths][optimal_p_value]['y_mc_relu_v'],
            'p': results['ow'][optimal_paths][optimal_p_value]['y_mc_relu_P'],
            'q': results['ow'][optimal_paths][optimal_p_value]['y_mc_relu_q'],
            'r_itr': [results['ow'][optimal_paths][optimal_p_value]['y_mc_relu_reward']],
            'model_r_itr': [results['ow'][optimal_paths][optimal_p_value]['y_mc_relu_reward']],
            'p_itr': [results['ow'][optimal_paths][optimal_p_value]['y_mc_relu_P']],
            'model_p_itr':[results['ow'][optimal_paths][optimal_p_value]['y_mc_relu_P']],
            #'time': run_time,
            #'uncertainty': y_mc_std_relu_resized, #leave out so no uncertainty plotted
            'truth_figure_title': 'Truth R & P',
            'pred_reward_figure_title': 'Pred R & P w/ ReLU non-linearities',
            'uncertainty_figure_title': 'Uncertainty w/ ReLU non-linearities'
        }

        # Ground truth dict for predicitons with ReLU non-linearities
        test_result_relu = { 
            'irl_result': irl_result_relu,
            'true_r': r,
            'example_samples': [example_samples],
            'mdp_data': mdp_data,
            'mdp_params': mdp_params,
            'mdp_solution': mdp_solution,
            'feature_data': feature_data
        }

        #Plot True R and P
        fig1, ax1 = plt.subplots(1)
        objectworlddraw(test_result_relu['true_r'],test_result_relu['mdp_solution']['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig1, ax1)
        ax1.set_title('True Reward')
        fig1.savefig(GRAPHS_PATH + "true_reward.png")

        #Plot optimal Estimated R and P
        fig1, ax1 = plt.subplots(1)
        objectworlddraw(test_result_relu['irl_result']['r'],test_result_relu['irl_result']['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig1, ax1)
        ax1.set_title('Optimal Estimated Reward w/ Dropout = ' + str(optimal_p_value) + '& Paths = ' + str(optimal_paths))
        fig1.savefig(GRAPHS_PATH + "optimal_est_reward.png")

        
        #Plot 2: 4 tiles reward figure. Tiles = P values & paths = optimal

        for p_value in dropout_vals:
            fig2, ax1 = plt.subplots(1)
            irl_result_relu = { 
                'r': results['ow'][optimal_paths][p_value]['y_mc_relu_reward'],
                'v': results['ow'][optimal_paths][p_value]['y_mc_relu_v'],
                'p': results['ow'][optimal_paths][p_value]['y_mc_relu_P'],
                'q': results['ow'][optimal_paths][p_value]['y_mc_relu_q'],
                'r_itr': [results['ow'][optimal_paths][p_value]['y_mc_relu_reward']],
                'model_r_itr': [results['ow'][optimal_paths][p_value]['y_mc_relu_reward']],
                'p_itr': [results['ow'][optimal_paths][p_value]['y_mc_relu_P']],
                'model_p_itr':[results['ow'][optimal_paths][p_value]['y_mc_relu_P']],
                #'time': run_time,
                #'uncertainty': y_mc_std_relu_resized, #leave out so no uncertainty plotted
                'truth_figure_title': 'Truth R & P',
                'pred_reward_figure_title': 'Pred R & P w/ ReLU non-linearities',
                'uncertainty_figure_title': 'Uncertainty w/ ReLU non-linearities'
            }

            # Ground truth dict for predicitons with ReLU non-linearities
            test_result_relu = { 
                'irl_result': irl_result_relu,
                'true_r': r,
                'example_samples': [example_samples],
                'mdp_data': mdp_data,
                'mdp_params': mdp_params,
                'mdp_solution': mdp_solution,
                'feature_data': feature_data
            }

            objectworlddraw(test_result_relu['irl_result']['r'],test_result_relu['irl_result']['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig2, ax1)
           
            ax1.set_title('Dropout = ' + str(p_value) + ' & Paths = ' + str(optimal_paths))
            fig2.savefig(GRAPHS_PATH + str(p_value) +"_reward_predictions.png")


        #Plot 3: EVD (Y) vs Paths (X) line graph figure
        
        #Get all EVD values

        evd_vals = [[], [], [], []]
        
        i = 0
        for p_value in dropout_vals:
            for path in paths:
                curr_evd = results['ow'][path][p_value]['evd']
                evd_vals[i].append(curr_evd)
            i += 1


        # Plot lines ng
        fig3, ax1 = plt.subplots()
        ax1.plot(paths, evd_vals[0], alpha=0.8, label=list(dropout_vals)[0])
        ax1.plot(paths, evd_vals[1], alpha=0.8, label=list(dropout_vals)[1])
        ax1.plot(paths, evd_vals[2], alpha=0.8, label=list(dropout_vals)[2])
        ax1.plot(paths, evd_vals[3], alpha=0.8, label=list(dropout_vals)[3])
        ax1.legend(title='Dropout', fontsize='small')
        ax1.set_title('EVD vs Paths vs Dropout')
        ax1.set_xlabel('Number Of Paths')
        ax1.set_ylabel('EVD')

        #Save figure
        fig3.savefig(GRAPHS_PATH + "evd_vs_no_paths.png")

    else:
        
        #Init results dict - real
        results = {
                'ow': {
                      12: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}}, 
                      24: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}}, 
                      48: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}},
                      64: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}}
                      },
                     
                'gw': {
                      12: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}}, 
                      24: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}}, 
                      48: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}},
                      64: {0.0: {0: None, 1: None, 2: None, 3: None}, 0.2: {0: None, 1: None, 2: None, 3: None}, 0.4: {0: None, 1: None, 2: None, 3: None}, 0.8: {0: None, 1: None, 2: None, 3: None}}
                      }
            }
    

        # Get lists of variants
        # Worlds = results.keys()
        worlds = ['ow']
        paths = results['ow'].keys()
        dropout_vals = results['ow'][12].keys()
        variant_vals = results['ow'][12][0.0].keys()

        # Populate results dictionary
        for world in worlds:
            for no_paths in paths:
                for dropout_val in dropout_vals:
                    for variant in variant_vals:
                        file_name = RESULTS_PATH+str(world)+'_'+str(dropout_val)+'_'+ str(no_paths)+ '_results_'+str(variant)+'.pkl'
                        
                        open_file = open(file_name, "rb")
                        open_results = pickle.load(open_file)
                        open_file.close()
                        y_mc_std_relu_resized = torch.from_numpy(open_results[1])
                        y_mc_std_relu_resized = y_mc_std_relu_resized.reshape(len(y_mc_std_relu_resized), 1)
                        y_mc_std_relu_resized = y_mc_std_relu_resized.repeat((1, 5))
                        result_values = {'y_mc_relu': open_results[0], 
                                         'y_mc_std_relu': open_results[1],
                                         'y_mc_relu_reward': open_results[2],
                                         'y_mc_relu_v': open_results[3],
                                         'y_mc_relu_P': open_results[4],
                                         'y_mc_relu_q': open_results[5],
                                         'evd': open_results[6],
                                         'run_time': open_results[7],
                                         'num_preds': open_results[8],
                                         'y_mc_std_relu_resized': y_mc_std_relu_resized}
                        
                        results[str(world)][no_paths][dropout_val][variant] = result_values

        optimal_paths = 64
        optimal_p_value = 0.8
        optimal_variant = 0

        # Plot 4: 4 tiled uncertainty shading. Tile = variant, paths = optimal, P = optimal
        
        for variant in variant_vals:
            fig4, ax1 = plt.subplots(1)
            irl_result_relu = { 
                'r': results['ow'][optimal_paths][optimal_p_value][variant]['y_mc_std_relu_resized'], #make R uncertainty matrix for shading
                'v': results['ow'][optimal_paths][optimal_p_value][variant]['y_mc_relu_v'],
                'p': results['ow'][optimal_paths][optimal_p_value][variant]['y_mc_relu_P'],
                'q': results['ow'][optimal_paths][optimal_p_value][variant]['y_mc_relu_q'],
                'r_itr': [results['ow'][optimal_paths][optimal_p_value][variant]['y_mc_relu_reward']],
                'model_r_itr': [results['ow'][optimal_paths][optimal_p_value][variant]['y_mc_relu_reward']],
                'p_itr': [results['ow'][optimal_paths][optimal_p_value][variant]['y_mc_relu_P']],
                'model_p_itr':[results['ow'][optimal_paths][optimal_p_value][variant]['y_mc_relu_P']],
                #'time': run_time,
                #'uncertainty': y_mc_std_relu_resized, #leave out so no uncertainty plotted
                'truth_figure_title': 'Truth R & P',
                'pred_reward_figure_title': 'Uncertainty w/ ReLU non-linearities',
                #'uncertainty_figure_title': 'Uncertainty w/ ReLU non-linearities'
            }

            # Ground truth dict for predicitons with ReLU non-linearities
            test_result_relu = { 
                'irl_result': irl_result_relu,
                'true_r': r,
                'example_samples': [example_samples],
                'mdp_data': mdp_data,
                'mdp_params': mdp_params,
                'mdp_solution': mdp_solution,
                'feature_data': feature_data
            }

            objectworlddraw(test_result_relu['irl_result']['r'],test_result_relu['irl_result']['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig4, ax1)
            if experiment_type == 'noisey_paths':
                title = states_to_remove[variant]
            if experiment_type == 'noisey_features':
                title = features_corrupted[variant]

            ax1.set_title(title + ' w/ Dropout = ' + str(optimal_p_value)+ ' & Paths = ' + str(optimal_paths))
            fig4.savefig(GRAPHS_PATH + str(variant) +"_uncertainty_predictions.png")


        # Plot 5: 4 tiled uncertainty/reward line graph. Tile = variant, paths = optimal, P = optimal

        for variant in variant_vals:
            fig5, ax1 = plt.subplots(1, sharex=True)

            # Plot regression line w/ uncertainty shading
            ax1.plot(np.arange(1,len(feature_data['splittable'])+1,1), results['ow'][optimal_paths][optimal_p_value][variant]['y_mc_relu'], alpha=0.8)
            ax1.fill_between(np.arange(1,len(feature_data['splittable'])+1,1), (results['ow'][optimal_paths][optimal_p_value][variant]['y_mc_relu']-2*results['ow'][optimal_paths][optimal_p_value][variant]['y_mc_std_relu']).squeeze(), (results['ow'][optimal_paths][optimal_p_value][variant]['y_mc_relu']+2*results['ow'][optimal_paths][optimal_p_value][variant]['y_mc_std_relu']).squeeze(), alpha=0.3)
           
            if experiment_type == 'noisey_paths':
                ax1.axvline(num_states_to_remove[variant][0], color='g',linestyle='--')
                ax1.axvline(num_states_to_remove[variant][-1], color='g',linestyle='--')
                
            ax1.set_title('w/ ReLU non-linearities')
            ax1.set_xlabel('State')
            ax1.set_ylabel('Reward')

            if experiment_type == 'noisey_paths':
                title = states_to_remove[variant]
            if experiment_type == 'noisey_features':
                title = features_corrupted[variant]

            ax1.set_title(title + ' w/ Dropout = ' + str(optimal_p_value)+ ' & Paths = ' + str(optimal_paths))

            fig5.savefig(GRAPHS_PATH + str(variant) + "_uncertainty_line.png")

        # Plot 6: 4 tiled uncertainty shading. Tile = paths, variant = optimal, P = optimal

        #Create figure
        
        for path in paths:
            fig6, ax1 = plt.subplots(1)
            irl_result_relu = { 
                'r': results['ow'][path][optimal_p_value][optimal_variant]['y_mc_std_relu_resized'], #make R uncertainty matrix for shading
                'v': results['ow'][path][optimal_p_value][optimal_variant]['y_mc_relu_v'],
                'p': results['ow'][path][optimal_p_value][optimal_variant]['y_mc_relu_P'],
                'q': results['ow'][path][optimal_p_value][optimal_variant]['y_mc_relu_q'],
                'r_itr': [results['ow'][path][optimal_p_value][optimal_variant]['y_mc_relu_reward']],
                'model_r_itr': [results['ow'][path][optimal_p_value][optimal_variant]['y_mc_relu_reward']],
                'p_itr': [results['ow'][path][optimal_p_value][optimal_variant]['y_mc_relu_P']],
                'model_p_itr':[results['ow'][path][optimal_p_value][optimal_variant]['y_mc_relu_P']],
                #'time': run_time,
                #'uncertainty': y_mc_std_relu_resized, #leave out so no uncertainty plotted
                'truth_figure_title': 'Truth R & P',
                'pred_reward_figure_title': 'Uncertainty w/ ReLU non-linearities',
                #'uncertainty_figure_title': 'Uncertainty w/ ReLU non-linearities'
            }

            # Ground truth dict for predicitons with ReLU non-linearities
            test_result_relu = { 
                'irl_result': irl_result_relu,
                'true_r': r,
                'example_samples': [example_samples],
                'mdp_data': mdp_data,
                'mdp_params': mdp_params,
                'mdp_solution': mdp_solution,
                'feature_data': feature_data
            }

            objectworlddraw(test_result_relu['irl_result']['r'],test_result_relu['irl_result']['p'],g,test_result_relu['mdp_params'],test_result_relu['mdp_data'], fig6, ax1)
            if experiment_type == 'noisey_paths':
                title = states_to_remove[variant]
            if experiment_type == 'noisey_features':
                title = features_corrupted[variant]

            ax1.set_title(title + ' w/ Dropout = ' + str(optimal_p_value)+ ' & Paths = ' + str(path))
            fig6.savefig(GRAPHS_PATH +  str(path) +"_uncertainty_predictions.png")



        #Plot 7: 4 tiled uncertainty/reward line graph. Tile = paths, variant = optimal, P = optimal
        
        for path in paths:
            fig7, ax1 = plt.subplots(1, sharex=True)
            
            # Plot regression line w/ uncertainty shading
            ax1.plot(np.arange(1,len(feature_data['splittable'])+1,1), results['ow'][path][optimal_p_value][optimal_variant]['y_mc_relu'], alpha=0.8)
            ax1.fill_between(np.arange(1,len(feature_data['splittable'])+1,1), (results['ow'][path][optimal_p_value][optimal_variant]['y_mc_relu']-2*results['ow'][path][optimal_p_value][optimal_variant]['y_mc_std_relu']).squeeze(), (results['ow'][path][optimal_p_value][optimal_variant]['y_mc_relu']+2*results['ow'][path][optimal_p_value][optimal_variant]['y_mc_std_relu']).squeeze(), alpha=0.3)
           
            if experiment_type == 'noisey_paths':
                ax1.axvline(num_states_to_remove[variant][0], color='g',linestyle='--')
                ax1.axvline(num_states_to_remove[variant][-1], color='g',linestyle='--')
                
            ax1.set_xlabel('State')
            ax1.set_ylabel('Reward')

            if experiment_type == 'noisey_paths':
                title = states_to_remove[variant]
            if experiment_type == 'noisey_features':
                title = features_corrupted[variant]

            ax1.set_title(title + ' w/ Dropout = ' + str(optimal_p_value)+ ' & Paths = ' + str(path))

            fig7.savefig(GRAPHS_PATH + str(path) + "_uncertainty_line.png")


        #Plot 8: Uncertainty ranking table

        
    




