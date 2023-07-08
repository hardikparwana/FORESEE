import time
import jax
import jax.numpy as np
from jax import random, grad, jit, lax
import optax
import jaxopt
# jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})

from utils.utils import *
from cartpole_new2.cartpole_policy import policy, Sum_of_gaussians_initialize
from cartpole_new2.ut_utils.ut_utils import *

# visualization
from robot_models.custom_cartpole_mc_pilco import CustomCartPoleEnv
from robot_models.cartpole2D_mcpilco import step
from cartpole_new2.gym_wrappers.record_video import RecordVideo

key = random.PRNGKey(2)

def get_future_reward_ideal(X, params_policy, dt):
    reward = 0   
    states = np.copy(X)
    
    for i in range(H):
        action = policy( states, params_policy ).reshape(-1,1)
        states = step(states, action, dt)
        reward = reward + compute_reward(states)     
    return reward, states

def get_future_reward_ideal_ut(X, params_policy, dt):
    states, weights, weights_cov = initialize_sigma_points(X)
    reward = 0
        
    for i in range(H):
        mean_position = get_mean( states, weights )
        next_states_expanded, next_weights_expanded, next_weights_cov_expanded = sigma_point_expand_rk4_input( states, weights, weights_cov, params_policy, dt)
        next_states, next_weights, next_weights_cov = sigma_point_compress( next_states_expanded, next_weights_expanded, next_weights_cov_expanded )
        states = next_states
        weights = next_weights
        weights_cov = next_weights_cov
        reward = reward + reward_UT_Mean_Evaluator_basic( states, weights, weights_cov )    
    return reward, states

# Set up environment
env_to_render = CustomCartPoleEnv(render_mode="human")
env = RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/Research/FORESEE/", name_prefix="cartpole_sigma_test_ideal" )
observation, info = env.reset(seed=42)

key = random.PRNGKey(100)
key, subkey = random.split(key)
policy_type = 'with angles'
key, params_policy =  Sum_of_gaussians_initialize(subkey, state_dim=4, input_dim=1, type = policy_type, lengthscale = 1)
action = policy( np.array([0,0,0.1,0]).reshape(-1,1), params_policy ).reshape(-1,1)

t = 0
dt_inner = 0.02
dt_outer = 0.02
tf = 3.0#H * dt_outer
H = int( tf/dt_inner )
# H = 10
grad_clip = 2.0

state = np.copy(env.get_state())

t0 = time.time()
reward_ideal, states_ideal = get_future_reward_ideal( state, params_policy, dt_outer)
reward_ideal_ut, states_ideal_ut = get_future_reward_ideal_ut( state, params_policy, dt_outer)
print(f"initial reward ideal: {reward_ideal}, reward_ideal_ut:{reward_ideal_ut}, time to jit:{ time.time()-t0 }")

