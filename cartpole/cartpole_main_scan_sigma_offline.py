import time
import jax.numpy as np
from jax import random, grad, jit, vmap, vjp, lax, value_and_grad
from jax import jacfwd, jacrev
import jax
import scipy
from scipy.optimize import minimize as minimize_scipy
from jax.scipy.optimize import minimize
# jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})

from utils.utils import *
from cartpole_policy_scan_sigma import policy, policy_jit
from ut_utils.ut_utils import *
from robot_models.custom_cartpole_constrained import CustomCartPoleEnv
from robot_models.cartpole2D import step
from gym_wrappers.record_video import RecordVideo

key = random.PRNGKey(2)

def generate_psd_params():
    n = 4
    N = 60
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    diag = random.uniform(subkey, shape=(n,1))[:,0] + n
    key, subkey = random.split(key)
    off_diag = random.uniform(subkey, shape=(int( (n**2-n)/2.0 ),1))[:,0]
    params = np.append(diag, off_diag, axis = 0).reshape(1,-1)
    for i in range(1,N):
        # Diagonal elements
        key, subkey = random.split(key)
        params_temp = random.uniform( subkey, shape=( 1,int(n + (n**2 -n)/2.0)) )
        params = np.append( params, params_temp, axis = 0 )    
    return params

def policy_next_state1( params_policy, mean_position, states, weights, dt_outer, dynamics_params ):
    solution = policy_jit( params_policy, mean_position ).reshape(-1,1)
    next_states_expanded, next_weights_expanded = sigma_point_expand_jit( states, weights, solution, dt_outer, dynamics_params)#, gps )        
    next_states, next_weights = sigma_point_compress_jit( next_states_expanded, next_weights_expanded )
    reward = reward_UT_Mean_Evaluator_basic_jit( next_states, next_weights )
    return reward
policy_next_state1_grad = grad( policy_next_state1, 0 )

def policy_next_state2( params_policy, mean_position, states, weights, dt_outer, dynamics_params ):
    solution = policy_jit( params_policy, mean_position ).reshape(-1,1)
    next_states_expanded, next_weights_expanded = sigma_point_expand_jit( states, weights, solution, dt_outer, dynamics_params)#, gps )        
    next_states, next_weights = sigma_point_compress_jit( next_states_expanded, next_weights_expanded )
    return np.sum(next_states)
policy_next_state2_grad = grad( policy_next_state2, 0 )

def policy_next_state3( params_policy, mean_position, states, weights, dt_outer, dynamics_params ):
    solution = policy_jit( params_policy, mean_position ).reshape(-1,1)
    next_states_expanded, next_weights_expanded = sigma_point_expand_jit( states, weights, solution, dt_outer, dynamics_params)#, gps )        
    return np.sum(next_states_expanded)
policy_next_state3_grad = grad( policy_next_state3, 0 )

def get_future_reward(X, horizon, dt_outer, dynamics_params, params_policy):
    states, weights = initialize_sigma_points(X)
    reward = 0
    H = 250
    def body(t, inputs):
        reward, states, weights = inputs
        mean_position = get_mean( states, weights )
        solution = policy( params_policy, mean_position )
        # print(f"input: {solution}")
        next_states_expanded, next_weights_expanded = sigma_point_expand( states, weights, solution, dt_outer, dynamics_params)#, gps )        
        next_states, next_weights = sigma_point_compress( next_states_expanded, next_weights_expanded )
        states = next_states
        weights = next_weights
        reward = reward + reward_UT_Mean_Evaluator_basic( states, weights )
        return reward, states, weights
    
    return lax.fori_loop( 0, H, body, (reward, states, weights) )[0]

    for i in range(H):
        mean_position = get_mean( states, weights )
        solution = policy( params_policy, mean_position ).reshape(-1,1)
        # print(f"input: {solution}")
        next_states_expanded, next_weights_expanded = sigma_point_expand( states, weights, solution, dt_outer, dynamics_params)#, gps )        
        next_states, next_weights = sigma_point_compress( next_states_expanded, next_weights_expanded )
        states = next_states
        weights = next_weights
        reward = reward + reward_UT_Mean_Evaluator_basic( states, weights )
    return reward

# get_future_reward_grad = grad(get_future_reward)
get_future_reward_jit = get_future_reward#jit(get_future_reward)

get_future_reward_grad = value_and_grad(get_future_reward, 4)
get_future_reward_grad_jit = jit(get_future_reward_grad)


# Set up environment
env_to_render = CustomCartPoleEnv(render_mode="rgb_array")
env = RecordVideo( env_to_render, video_folder="/home/dasc/hardik/videos/", name_prefix="cartpole_default_inverted" )
observation, info = env.reset(seed=42)

polemass_length, gravity, length, masspole, total_mass, tau = env.polemass_length, env.gravity, env.length, env.masspole, env.total_mass, env.tau
dynamics_params = np.array([ polemass_length, gravity, length, masspole, total_mass])#, tau ])

# Initialize parameters
n = 4
N = 60
H = 20
lr_rate = 1.0#0.1#1.0##0.01
key = random.PRNGKey(100)
key, subkey = random.split(key)
param_w = 10.0*(random.uniform(subkey, shape=(N,1))[:,0] - 0.5)#+ 0.5#+ 2.0  #0.5 work with Lr: 5.0
key, subkey = random.split(key)
param_mu = random.uniform(subkey, shape=(4,N))- 0.5 * np.ones((4,N)) #- 3.5 * np.ones((4,N))
param_Sigma = generate_psd_params() # 10,N
params_policy = np.append( np.append( param_w, param_mu.reshape(-1,1)[:,0] ), param_Sigma.reshape(-1,1)[:,0]  )

t = 0
dt_inner = 0.02
dt_outer = 0.02
tf = 8.0#4.0

state = np.copy(env.get_state())
t0 = time.time()
action = policy_jit( params_policy, state)
print(f"Policy JITed in time: {time.time()-t0}")

# RUN this
get_future_reward( state, H, dt_outer, dynamics_params, params_policy )
# get_future_reward_grad( state, H, dt_outer, dynamics_params, params_policy )

t0 = time.time()
get_future_reward_grad_jit( state, H, dt_outer, dynamics_params, params_policy)
print(f"time jit for: {time.time()-t0}")

#train using scipy
t0 = time.time()
get_future_reward_minimize = lambda params: get_future_reward( state, H, dt_outer, dynamics_params, params )
get_future_reward_minimize_jit = jit(get_future_reward_minimize)
get_future_reward_minimize_jit( params_policy )
print(f"time jit for: {time.time()-t0}")
# res = minimize( get_future_reward_minimize_jit, params_policy, method='BFGS', tol=1e-6 )
# params_policy = res.x

res = minimize_scipy( get_future_reward_minimize_jit, params_policy, method='Nelder-Mead', tol=1e-6 )
params_policy = res.x

# train
# for i in range(200):
#     reward, param_policy_grad = get_future_reward_grad_jit( state, H, dt_outer, dynamics_params, params_policy)
#     param_policy_grad = np.clip( param_policy_grad, -2.0, 2.0 )

#     params_policy = params_policy - lr_rate * param_policy_grad
#     params_w_mu_temp = np.clip( params_policy[0:N*n+N], -10, 10  )
#     params_sigma_temp = np.clip( params_policy[N*n+N:], -1, 1 )
#     params_policy = np.append( params_w_mu_temp, params_sigma_temp )
#     if i%1==0:
#         print(f"i:{i}, reward:{reward}, grad: {np.max(np.abs(param_policy_grad))}")

while t < tf:

    action = policy_jit( params_policy, state ).reshape(-1,1)
    print(f"action:{action}")
    next_state = step(state, action, dynamics_params, dt_inner)
    env.set_state( (next_state[0,0].item(), next_state[1,0].item(), next_state[2,0].item(), next_state[3,0].item()) )
    env.render()  
    
    state = np.copy(next_state)
    t = t + dt_inner
env.close()