import time
import jax.numpy as np
from jax import random, grad, jit, vmap, vjp
from jax import jacfwd, jacrev

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})

from utils.utils import *
from cartpole_policy_up import policy, policy_jit
from ut_utils.ut_utils_old import *
from robot_models.custom_cartpole_constrained import CustomCartPoleEnv
from robot_models.cartpole2D import step
from gym_wrappers.record_video import RecordVideo

key = random.PRNGKey(0)

def generate_psd_matrix_inverse():
    n = 4
    N = 50
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    params_temp = random.uniform( subkey, shape=( int(n + (n**2 -n)/2.0),1) )
    Sigma = np.array([  
            [ params_temp[0,0], 0.0, 0.0, 0.0 ],
            [ params_temp[4,0], params_temp[1,0], 0.0, 0.0 ],
            [ params_temp[5,0], params_temp[6,0], params_temp[2,0], 0.0 ],
            [ params_temp[7,0], params_temp[8,0], params_temp[9,0], params_temp[3,0] ]
        ])
    Sigma = 4 * np.eye(4) + Sigma.T @ Sigma
    Sigma_inverse = np.linalg.inv( Sigma )
    Sigma_inverse = (Sigma_inverse + Sigma_inverse.T) /2.0
    Sigmas = np.copy( Sigma_inverse.reshape(-1,1) )

    for i in range(1,50):
        # Diagonal elements
        key, subkey = random.split(key)
        params_temp = random.uniform( subkey, shape=( 1,int(n + (n**2 -n)/2.0)) )
        Sigma = np.array([  
            [ params_temp[0,0], 0.0, 0.0, 0.0 ],
            [ params_temp[4,0], params_temp[1,0], 0.0, 0.0 ],
            [ params_temp[5,0], params_temp[6,0], params_temp[2,0], 0.0 ],
            [ params_temp[7,0], params_temp[8,0], params_temp[9,0], params_temp[3,0] ]
        ])
        Sigma = 4 * np.eye(4) + Sigma.T @ Sigma
        Sigma_inverse = np.linalg.inv( Sigma )
        Sigma_inverse = (Sigma_inverse + Sigma_inverse.T) /2.0
        Sigmas = np.append( Sigmas, np.copy( Sigma_inverse.reshape(-1,1) ) )
    return Sigmas

def get_future_reward(X, horizon, dt_outer, dynamics_params, params_policy, Sigma_invs):
    states, weights = initialize_sigma_points_jit(X)
    reward = 0
    for i in range(20):  
        mean_position = get_mean_jit( states, weights )
        solution = policy_jit( params_policy, Sigma_invs, mean_position )
        next_states_expanded, next_weights_expanded = sigma_point_expand_jit( states, weights, solution, dt_outer, dynamics_params)#, gps )        
        next_states, next_weights = sigma_point_compress_jit( next_states_expanded, next_weights_expanded )
        states = next_states
        weights = next_weights
        reward = reward + reward_UT_Mean_Evaluator_basic_jit( states, weights )
    return reward

# get_future_reward_grad = grad(get_future_reward)
# get_future_reward_jit = jit(get_future_reward)

get_future_reward_grad = grad(get_future_reward, 4)
get_future_reward_grad_jit = jit(get_future_reward_grad)


# Set up environment
env_to_render = CustomCartPoleEnv(render_mode="human")
env = RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/", name_prefix="cartpole_test_ideal" )
observation, info = env.reset(seed=42)

polemass_length, gravity, length, masspole, total_mass, tau = env.polemass_length, env.gravity, env.length, env.masspole, env.total_mass, env.tau
dynamics_params = np.array([ polemass_length, gravity, length, masspole, total_mass])#, tau ])

# Initialize parameters
N = 30
H = 20
lr_rate = 0.01
key = random.PRNGKey(100)
key, subkey = random.split(key)
param_w = random.uniform(subkey, shape=(N,1))[:,0] - 0.5#+ 0.5#+ 2.0  #0.5 work with Lr: 5.0
key, subkey = random.split(key)
param_mu = random.uniform(subkey, shape=(4,N))- 0.5 * np.ones((4,N)) #- 3.5 * np.ones((4,N))
# param_Sigma = generate_psd_params() # 10,N
Sigma_invs = generate_psd_matrix_inverse()
params_policy = np.append( param_w, param_mu.reshape(-1,1)[:,0] )

t = 0
dt_inner = 0.1
dt_outer = 0.1
tf = 4.0#4.0

state = np.copy(env.get_state())
t0 = time.time()
action = policy_jit( params_policy, Sigma_invs, state )
print(f"Policy JITed in time: {time.time()-t0}")
get_future_reward( state, H, dt_outer, dynamics_params, params_policy, Sigma_invs )
get_future_reward_grad( state, H, dt_outer, dynamics_params, params_policy, Sigma_invs )

while t < tf:
    
    action = policy_jit( params_policy, Sigma_invs, state )
    next_state = step(state, action, dynamics_params, dt_inner)
    t0 = time.time()
    
    # reward = get_future_reward_jit( state, H, dt_inner, dynamics_params, param_w, param_mu, param_Sigma )
    # w_grad, mu_grad, Sigma_grad = unconstrained_update(reward)
    
    param_policy_grad = get_future_reward_grad_jit( state, H, dt_outer, dynamics_params, params_policy, Sigma_invs )
    print(f"time reward :{ time.time()-t0 }")#, reward: {reward}")
    param_policy_grad = np.clip( param_policy_grad, -2.0, 2.0 )
    params_policy = params_policy - lr_rate * param_policy_grad
    params_policy[0:4*N+N] =  np.clip( params_policy[0:4*N+N], -10, 10 )
    params_policy[5*N:5*N+10*N] =  np.clip( params_policy[0:4*N+N], -1, 1 )
    
    
    env.set_state( (next_state[0,0].item(), next_state[1,0].item(), next_state[2,0].item(), next_state[3,0].item()) )
    env.render()  
    
    state = np.copy(next_state)
    t = t + dt_inner
    