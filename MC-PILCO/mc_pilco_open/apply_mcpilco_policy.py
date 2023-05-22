# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

""" 
Authors:    Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
            Fabio Amadio (fabioamadio93@gmail.com)
MERL:	    Diego Romeres (romeres@merl.com)
"""


""" 
File that loads the logs and test the final policy obtained with MC-PILCO in the simualted cart-pole system
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import torch
import model_learning.Model_learning as ML
import policy_learning.Policy as Policy
import policy_learning.MC_PILCO as MC_PILCO
import policy_learning.Cost_function as Cost_function
import simulation_class.ode_systems as f_ode
import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import copy
import time
from scipy import signal

torch.set_num_threads(1)

# define paths
seed = 1
folder_path = 'results_tmp/'+str(seed)+'/'
config_file_path = folder_path+'/config_log.pkl'
saving_path = folder_path+'/reproduce_policy_log.pkl'

# repeat the policy 'num_test' times
num_test = 50

# select the policy obtained at trial 'num_trial'
num_trial = 5

# initialize the object
config_dict = pkl.load(open(config_file_path,'rb'))
PL_obj = MC_PILCO.MC_PILCO(**config_dict['MC_PILCO_init_dict'])
T_control = config_dict['reinforce_param_dict']['T_control']
initial_state = config_dict['reinforce_param_dict']['initial_state']

# set the policy parameters
PL_obj.load_policy_from_log(num_trial, folder=folder_path)

# test the policy
states_list = []
input_list = []
for i in range(num_test):
    state_samples, input_samples, noiseless_samples = PL_obj.system.rollout(s0 = initial_state,
                                                                            policy = PL_obj.control_policy.get_np_policy(),
                                                                            T = T_control,
                                                                            dt = PL_obj.T_sampling,
                                                                            noise = PL_obj.std_meas_noise)
    states_list.append(noiseless_samples)
    input_list.append(input_samples)

results_dict = {}
results_dict['states_list'] = states_list
results_dict['input_list'] = input_list
pkl.dump(results_dict, open(saving_path, 'wb'))

# plot trajectories
plt.figure()
plt.subplot(3,1,1)
plt.grid()
plt.ylabel('POLE')
plt.plot(np.pi*np.ones(len(states_list[0][:,2])), 'r--')
plt.plot(-np.pi*np.ones(len(states_list[0][:,2])), 'r--')
for i in range(num_test):
    plt.plot(states_list[i][:,2])
plt.subplot(3,1,2)
plt.grid()
plt.ylabel('CART')
plt.plot(np.zeros(len(states_list[0][:,2])), 'r--')
for i in range(num_test):
    plt.plot(states_list[i][:,0])
plt.subplot(3,1,3)
plt.grid()
plt.ylabel('FORCE')
for i in range(num_test):
    plt.plot(input_list[i])
plt.show()
