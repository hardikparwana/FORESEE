# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

""" 
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL:	    Diego Romeres (romeres@merl.com)
"""


"""
Test MC-PILCO on a simulated cart-pole system with bimodal initial state distribution
"""

import torch
import numpy as np
import model_learning.Model_learning as ML
import policy_learning.Policy as Policy
import policy_learning.MC_PILCO as MC_PILCO
import policy_learning.Cost_function as Cost_function
import simulation_class.ode_systems as f_ode
import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import matplotlib.pyplot as plt
import pickle as pkl
import argparse

# Load random seed from command line
p = argparse.ArgumentParser('test cartpole multi init')
p.add_argument('-seed', type=int, default=1, help='seed')
locals().update(vars(p.parse_known_args()[0]))

# Set the seed
torch.manual_seed(seed)
np.random.seed(seed)

dtype=torch.float64

# Set the device
device=torch.device('cpu')
# device=torch.device('cuda:0')

# Set number of computational threads
num_threads = 1
torch.set_num_threads(num_threads)

print('---- Set environment parameters ----')
num_trials = 5
T_sampling = 0.05
T_exploration = 3.
T_control = 3.
state_dim = 4
input_dim = 1
num_gp = int(state_dim/2)
gp_input_dim = 6
ode_fun = f_ode.cartpole
u_max = 10.
std_noise = 10**(-2)
std_list = [std_noise, std_noise, std_noise, std_noise]

print('\n---- Set model learning parameters ----')
f_model_learning = ML.Speed_Model_learning_RBF_MPK_angle_state
print(f_model_learning)
model_learning_par = {}
model_learning_par['num_gp'] = num_gp
model_learning_par['angle_indeces'] = [2]
model_learning_par['not_angle_indeces'] = [0,1,3]
model_learning_par['T_sampling'] = T_sampling
model_learning_par['vel_indeces'] = [1,3]
model_learning_par['not_vel_indeces'] = [0,2]
model_learning_par['device'] = device
model_learning_par['dtype'] = dtype
# RBF initial par
init_dict_RBF = {}
init_dict_RBF['active_dims'] = np.arange(0,gp_input_dim)
init_dict_RBF['lengthscales_init'] = np.ones(init_dict_RBF['active_dims'].size)
init_dict_RBF['flg_train_lengthscales'] = True
init_dict_RBF['lambda_init'] = np.ones(1)
init_dict_RBF['flg_train_lambda'] = False
init_dict_RBF['sigma_n_init'] = 1*np.ones(1)
init_dict_RBF['sigma_n_num'] = None
init_dict_RBF['flg_train_sigma_n'] = True
init_dict_RBF['dtype'] = dtype
init_dict_RBF['device'] = device
# MPK initial par
init_dict_MPK = {}
init_dict_MPK['active_dims'] = np.arange(0,gp_input_dim)
init_dict_MPK['poly_deg'] = 2
init_dict_MPK['Sigma_pos_par_init_list'] = [np.ones(gp_input_dim+1)]+[np.ones((deg+1)*(gp_input_dim)) for deg in range(1,init_dict_MPK['poly_deg'])]
init_dict_MPK['flg_train_Sigma_pos_par_list'] = [True]*init_dict_MPK['poly_deg']
init_dict_MPK['dtype'] = dtype
init_dict_MPK['device'] = device
model_learning_par['init_dict_list'] = [[init_dict_RBF, init_dict_MPK]]*num_gp

print('\n---- Set exploration policy ----')
f_rand_exploration_policy = Policy.Random_exploration
rand_exploration_policy_par = {}
rand_exploration_policy_par['state_dim'] = state_dim
rand_exploration_policy_par['input_dim'] = input_dim
rand_exploration_policy_par['u_max'] = u_max
rand_exploration_policy_par['dtype'] = dtype
rand_exploration_policy_par['device'] = device

print('\n---- Set control policy ----')
num_basis = 200
f_control_policy = Policy.Sum_of_gaussians_with_angles
control_policy_par = {}
control_policy_par['state_dim'] = state_dim
control_policy_par['input_dim'] = input_dim
control_policy_par['angle_indices'] = np.array([2])
control_policy_par['non_angle_indices'] = np.array([0,1,3])
control_policy_par['u_max'] = u_max
control_policy_par['num_basis'] = num_basis
control_policy_par['dtype'] = dtype
control_policy_par['device'] = device
angle_centers = np.pi*2*(np.random.rand(num_basis,1)-0.5)
cos_centers = np.cos(angle_centers)
sin_centers = np.sin(angle_centers)
not_angle_centers = 2*np.array([2,2,2*np.pi])*(np.random.rand(num_basis,3)-0.5)
control_policy_par['centers_init'] = np.concatenate([not_angle_centers,cos_centers,sin_centers],1)
control_policy_par['lengthscales_init'] = 1*np.ones(state_dim+1)
control_policy_par['weight_init'] = u_max*(np.random.rand(input_dim,num_basis)-0.5)
control_policy_par['flg_squash'] = True
control_policy_par['flg_drop'] = True
policy_reinit_dict = {}
policy_reinit_dict['lenghtscales_par'] = control_policy_par['lengthscales_init']
policy_reinit_dict['centers_par'] = np.array([np.pi, np.pi, np.pi, 1., 1.])
policy_reinit_dict['weight_par'] = u_max

print('\n---- Set cost function ----')
f_cost_function = Cost_function.Cart_pole_cost
cost_function_par = {}
cost_function_par['pos_index'] = 0
cost_function_par['angle_index'] = 2
cost_function_par['target_state'] = torch.tensor([np.pi, 0.], dtype=dtype, device=device)
cost_function_par['lengthscales'] = torch.tensor([3.,1.], dtype=dtype, device=device)

print('\n---- Init policy learning object ----')
MC_PILCO_init_dict = {}
MC_PILCO_init_dict['T_sampling'] = T_sampling
MC_PILCO_init_dict['state_dim'] = state_dim
MC_PILCO_init_dict['input_dim'] = input_dim
MC_PILCO_init_dict['f_sim'] = ode_fun
MC_PILCO_init_dict['std_meas_noise'] = np.array(std_list)
MC_PILCO_init_dict['f_model_learning'] = f_model_learning
MC_PILCO_init_dict['model_learning_par'] = model_learning_par
MC_PILCO_init_dict['f_rand_exploration_policy'] = f_rand_exploration_policy
MC_PILCO_init_dict['rand_exploration_policy_par'] = rand_exploration_policy_par
MC_PILCO_init_dict['f_control_policy'] = f_control_policy
MC_PILCO_init_dict['control_policy_par'] = control_policy_par
MC_PILCO_init_dict['f_cost_function'] = f_cost_function
MC_PILCO_init_dict['cost_function_par'] = cost_function_par
MC_PILCO_init_dict['log_path'] = 'results_tmp/'+str(seed)
MC_PILCO_init_dict['dtype'] = dtype
MC_PILCO_init_dict['device'] = device
PL_obj = MC_PILCO.MC_PILCO(**MC_PILCO_init_dict)

print('\n---- Set MC-PILCO options ----')
# Model optimization options
model_optimization_opt_dict = {}
model_optimization_opt_dict['f_optimizer'] = 'lambda p : torch.optim.Adam(p, lr=0.01)'
model_optimization_opt_dict['criterion'] = Likelihood.Marginal_log_likelihood
model_optimization_opt_dict['N_epoch'] = 1501
model_optimization_opt_dict['N_epoch_print'] = 500
model_optimization_opt_list = [model_optimization_opt_dict]*num_gp
# Policy optimization options
policy_optimization_dict = {}
policy_optimization_dict['num_particles'] = 400
policy_optimization_dict['opt_steps_list'] = [2000, 4000, 4000, 4000, 4000]
policy_optimization_dict['lr_list'] = [0.01, 0.01, 0.01, 0.01, 0.01]
policy_optimization_dict['f_optimizer'] = 'lambda p, lr : torch.optim.Adam(p, lr)'
policy_optimization_dict['num_step_print'] = 100
policy_optimization_dict['p_dropout_list'] = [.25, .25, .25, .25, .25]
policy_optimization_dict['p_drop_reduction'] = 0.25/2
policy_optimization_dict['alpha_diff_cost']=0.99
policy_optimization_dict['min_diff_cost']= 0.08
policy_optimization_dict['num_min_diff_cost']=200
policy_optimization_dict['min_step']=200
policy_optimization_dict['lr_min']=0.0025
policy_optimization_dict['policy_reinit_dict'] = policy_reinit_dict
# Options for method reinforce
reinforce_param_dict = {}
reinforce_param_dict['flg_init_multi_gauss'] = True # Set True to handle multi-gaussian initial distribution
reinforce_param_dict['initial_state'] = np.array([[-1., 0., 0., 0.],[1., 0., 0., 0.]]) # Two possible mean initial state
reinforce_param_dict['initial_state_var'] = np.array([[0.0001, 0.0001, 0.0001, 0.0001],[0.0001, 0.0001, 0.0001, 0.0001]]) # Two associated variances
reinforce_param_dict['T_exploration'] = T_exploration
reinforce_param_dict['T_control'] = T_control
reinforce_param_dict['num_trials'] = num_trials
reinforce_param_dict['model_optimization_opt_list'] = model_optimization_opt_list
reinforce_param_dict['policy_optimization_dict'] = policy_optimization_dict

print('\n---- Save test configuration ----')
config_log_dict = {}
config_log_dict['MC_PILCO_init_dict'] = MC_PILCO_init_dict
config_log_dict['reinforce_param_dict'] = reinforce_param_dict
pkl.dump(config_log_dict,open('results_tmp/'+str(seed)+'/config_log.pkl','wb'))

# Start the learning algorithm
PL_obj.reinforce(**reinforce_param_dict)
