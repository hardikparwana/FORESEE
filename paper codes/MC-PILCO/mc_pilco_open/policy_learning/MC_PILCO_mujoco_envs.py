# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

""" 
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL contact:	Diego Romeres (romeres@merl.com)
"""
import torch
import sys
sys.path.append('..')
import simulation_class.model_mujoco as model
from policy_learning.MC_PILCO import MC_PILCO
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import time
from scipy import signal
from scipy.linalg import block_diag
import copy

class MC_PILCO_Mujoco(MC_PILCO):
    """
    MC-PILCO implementation for Mujoco Environment
    """
    def __init__(self, T_sampling, state_dim, input_dim, f_sim, sim_timestep,
                 f_model_learning, model_learning_par,
                 f_rand_exploration_policy, rand_exploration_policy_par,
                 f_control_policy, control_policy_par,
                 f_cost_function, cost_function_par,
                 std_meas_noise = None, log_path = None,
                 dtype = torch.float64, device=torch.device('cpu')):
        super(MC_PILCO_Mujoco, self).__init__(T_sampling = T_sampling, state_dim = state_dim, input_dim = input_dim,
                                              f_sim = f_sim , f_model_learning = f_model_learning, model_learning_par = model_learning_par,
                                              f_rand_exploration_policy = f_rand_exploration_policy, rand_exploration_policy_par = rand_exploration_policy_par,
                                              f_control_policy = f_control_policy, control_policy_par = control_policy_par,
                                              f_cost_function = f_cost_function, cost_function_par = cost_function_par,
                                              std_meas_noise = std_meas_noise, log_path = log_path, dtype = dtype, device = device)

        self.system = model.Mujoco_Model(f_sim, sim_timestep) # MuJoCo-simulated system

