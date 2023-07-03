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
from robot_models.cartpole2D_mcpilco import step, get_state_dot_noisy_mc
from cartpole_new2.gym_wrappers.record_video import RecordVideo

key = random.PRNGKey(2)

def get_future_reward_mc(X, params_policy, dt, key, num_particles):
    states = np.copy(X)
    reward = 0
    for i in range(H):
        action = np.array([[0.0]])
        xdot, key = get_state_dot_noisy_mc( states[:,0].reshape(-1,1), action, key )
        states_new = (states[:,0] + xdot[:,0] * dt).reshape(-1,1)
        for j in range(1,num_particles):
            xdot, key = get_state_dot_noisy_mc( states[:,j].reshape(-1,1), action, key )
            state_temp = states[:,j] + xdot[:,0] * dt
            state_new = np.array([ state_temp[0], state_temp[1], wrap_angle(state_temp[2]), state_temp[3] ])
            states_new = np.append( states_new, state_new.reshape(-1,1), axis=1)
        states = states_new
        mean_state = np.sum( states, 1 ).reshape(-1,1) / num_particles
        reward = reward + compute_reward(mean_state)     
    return reward, states, key

def get_future_reward_ideal_ut(X, params_policy, dt, key): 
    states, weights, weights_cov = initialize_sigma_points(X)
    reward = 0
        
    for i in range(H):
        next_states_expanded, next_weights_expanded, next_weights_cov_expanded = sigma_point_expand_rk4_zero_input_noisy( states, weights, weights_cov, params_policy, dt)
        next_states, next_weights, next_weights_cov = sigma_point_compress( next_states_expanded, next_weights_expanded, next_weights_cov_expanded )
        states = next_states
        weights = next_weights
        weights_cov = next_weights_cov
        reward = reward + reward_UT_Mean_Evaluator_basic( states, weights, weights_cov )    
    return reward, states, weights, weights_cov

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
H = 30#int( tf/dt_inner )
# H = 10
grad_clip = 2.0
num_particles = 500

state = np.copy(env.get_state())

t0 = time.time()
reward_mc, states_mc, key = get_future_reward_mc( state, params_policy, dt_outer, key, num_particles)
reward_ut, states_ut, weights_ut, weights_cov_ut = get_future_reward_ideal_ut( state, params_policy, dt_outer, key)
print(f"initial reward mc: {reward_mc}, reward ut:{reward_ut}, time to jit:{ time.time()-t0 }")

mu_mc, cov_mc = get_mean_cov( states_mc, np.ones((1,num_particles))/num_particles, np.ones((1,num_particles))/num_particles )
mu_ut, cov_ut = get_mean_cov( states_ut, weights_ut, weights_cov_ut )

print(f"Final state mc: { mu_mc }, ut:{ mu_ut }")

print(f"Cov state mc: { cov_mc }, ut:{ cov_ut }")

fig, ax = plt.subplots(1)
ax.scatter(states_mc[0,:], states_mc[1,:], c = 'g')
ax.scatter(states_ut[0,:], states_ut[1,:], c = 'k')
plt.show()

