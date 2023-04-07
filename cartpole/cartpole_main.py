import time
import jax.numpy as np
from jax import random, grad, jit, vmap, vjp
from jax import jacfwd, jacrev

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})

from utils.utils import *
from cartpole_policy import policy, policy_jit
from ut_utils.ut_utils import *
from robot_models.custom_cartpole_constrained import CustomCartPoleEnv
from robot_models.cartpole2D import step
from gym_wrappers.record_video import RecordVideo

key = random.PRNGKey(0)

def generate_psd_params():
    n = 4
    N = 50
    diag = random.uniform(key, shape=(n,1))[:,0] + n
    off_diag = random.uniform(key, shape=(int( (n**2-n)/2.0 ),1))[:,0]
    params = np.append(diag, off_diag, axis = 0).reshape(1,-1)
    for i in range(1,50):
        # Diagonal elements
        params_temp = random.uniform( key, shape=( 1,int(n + (n**2 -n)/2.0)) )
        params = np.append( params, params_temp, axis = 0 )    
    return params

def get_future_reward(X, horizon, dt_outer, dynamics_params, params_policy):
    states, weights = initialize_sigma_points(X)
    reward = 0
    for i in range(3):  
        mean_position = get_mean( states, weights )
        solution = policy( params_policy, mean_position )
        next_states_expanded, next_weights_expanded = sigma_point_expand( states, weights, solution, dt_outer, dynamics_params)#, gps )        
        next_states, next_weights = sigma_point_compress( next_states_expanded, next_weights_expanded )
        states = next_states
        weights = next_weights
        reward = reward + reward_UT_Mean_Evaluator_basic( states, weights )
    return reward

# get_future_reward_grad = grad(get_future_reward)
# get_future_reward_jit = jit(get_future_reward)

get_future_reward_grad = grad(get_future_reward, 4)
get_future_reward_grad_jit = jit(get_future_reward_grad)


# Set up environment
env_to_render = CustomCartPoleEnv(render_mode="human")
env = RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/", name_prefix="cartpole_constrained_H20" )
observation, info = env.reset(seed=42)

polemass_length, gravity, length, masspole, total_mass, tau = env.polemass_length, env.gravity, env.length, env.masspole, env.total_mass, env.tau
dynamics_params = np.array([ polemass_length, gravity, length, masspole, total_mass])#, tau ])

# Initialize parameters
N = 50
H = 20
param_w = random.uniform(key, shape=(N,1))[:,0] - 0.5#+ 0.5#+ 2.0  #0.5 work with Lr: 5.0
param_mu = random.uniform(key, shape=(4,N))- 0.5 * np.ones((4,N)) #- 3.5 * np.ones((4,N))
param_Sigma = generate_psd_params() # 10,N
params_policy = np.append( param_w, np.append( param_mu.reshape(-1,1)[:,0], param_Sigma.reshape(-1,1)[:,0] ) )

t = 0
dt_inner = 0.02
dt_outer = 0.1
tf = 0.06

state = np.copy(env.get_state())

while t < tf:
    
    action = policy( params_policy, state )
    next_state = step(state, action, dynamics_params, dt_inner)
    t0 = time.time()
    
    # reward = get_future_reward_jit( state, H, dt_inner, dynamics_params, param_w, param_mu, param_Sigma )
    # w_grad, mu_grad, Sigma_grad = unconstrained_update(reward)
    
    param_policy_grad = get_future_reward_grad_jit( state, H, dt_outer, dynamics_params, params_policy )
    params_policy = params_policy + 0.01 * param_policy_grad
    
    print(f"time reward :{ time.time()-t0 }")#, reward: {reward}")
    env.set_state( (next_state[0,0].item(), next_state[1,0].item(), next_state[2,0].item(), next_state[3,0].item()) )
    env.render()  
    
    state = np.copy(next_state)
    t = t + dt_inner
    