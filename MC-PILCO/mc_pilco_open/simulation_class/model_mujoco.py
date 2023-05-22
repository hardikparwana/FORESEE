# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

""" 
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL contact:	Diego Romeres (romeres@merl.com)
"""

import gym
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from gym import wrappers
from scipy import signal

class Mujoco_Model():
    """
    MuJoCo Gym environment
    """
    
    def __init__(self, env_name, sim_timestep):
        """
        env_name: string containing environment name
        """
        self.env = gym.make(env_name)
        self.sim_timestep = sim_timestep # simulator timestep, it must be the same defined in the .xml file in envs/assets/...

    def rollout(self, s0, policy, T, dt, noise):
        """
        Generate a rollout of length T (s) for the Mujoco environment 'env_name' with 
        control inputs computed by 'policy' and applied with a sampling time 'dt'.
        'noise' defines the standard deviation of a Gaussian measurement noise.
            s0: initial state
            policy: policy function
            T: length rollout (s)
            dt: sampling time (s)
            noise: measurement noise std
        """


        state_dim = len(s0)
        init_pos = s0[0:int(len(s0)/2)]
        init_vel = s0[int(len(s0)/2):]
        times = np.linspace(0, T, int(T/dt))

        # init MuJoCo simulation
        self.env.frame_skip = int(dt/self.sim_timestep)
        self.env.init_qpos[0:int(len(s0)/2)] = init_pos
        self.env.init_qvel[-int(len(s0)/2):] = init_vel

        states = self.env.reset().reshape(1,-1)
        
        noisy_states = states + np.random.randn(state_dim)*noise
        # get initial input
        inputs = np.array([policy(noisy_states[0,:], 0)]).reshape(1,-1)
          
        for k in range(1,len(times)):
            self.env.render()
            # apply input
            new_state = self.env.step(inputs[k-1,:])
            noisy_new_state = new_state[0] + np.random.randn(state_dim)*noise
            # append 'new_state' to 'states' 
            states = np.append(states,[new_state[0]],axis=0)
            noisy_states = np.append(noisy_states, [noisy_new_state],axis=0)
            
            # compute next inputs
            u_next = np.array([policy(noisy_states[k,:], k)]).reshape(1,-1)
            # append u_next to 'inputs'
            inputs = np.append(inputs, u_next, axis=0)
        
        return noisy_states, inputs, states
