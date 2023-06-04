# Use this file for Ideal scenario: when uncertain dynamics is directly passed on to jax
# export PYDEVD_WARN_EVALUATION_TIMEOUT=100 to increase timeout with vscode debugger
import time
import jax.numpy as np
from jax import random, grad, jit, vmap, vjp, lax
from jax import jacfwd, jacrev
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})
from jax.scipy.optimize import minimize

from utils.utils import *
from cartpole_policy_scan import policy, policy_jit
from ut_utils.ut_utils_old import *
from robot_models.custom_cartpole_constrained import CustomCartPoleEnv
from robot_models.cartpole2D import step
from gym_wrappers.record_video import RecordVideo

key = random.PRNGKey(0)

def generate_psd_matrix_inverse(n, N):
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    params_temp = random.uniform( subkey, shape=( int(n + (n**2 -n)/2.0),1) )
    Sigma = np.array([  
            [ params_temp[0,0], 0.0, 0.0, 0.0 ],
            [ params_temp[4,0], params_temp[1,0], 0.0, 0.0 ],
            [ params_temp[5,0], params_temp[6,0], params_temp[2,0], 0.0 ],
            [ params_temp[7,0], params_temp[8,0], params_temp[9,0], params_temp[3,0] ]
        ])
    Sigma = n * np.eye(n) + Sigma.T @ Sigma
    Sigma_inverse = np.linalg.inv( Sigma )
    Sigma_inverse = (Sigma_inverse + Sigma_inverse.T) /2.0
    Sigmas = np.copy( Sigma_inverse.reshape(1,-1) )

    for i in range(1,N):
        # Diagonal elements
        key, subkey = random.split(key)
        params_temp = random.uniform( subkey, shape=( 1,int(n + (n**2 -n)/2.0)) )
        Sigma = np.array([  
            [ params_temp[0,0], 0.0, 0.0, 0.0 ],
            [ params_temp[4,0], params_temp[1,0], 0.0, 0.0 ],
            [ params_temp[5,0], params_temp[6,0], params_temp[2,0], 0.0 ],
            [ params_temp[7,0], params_temp[8,0], params_temp[9,0], params_temp[3,0] ]
        ])
        Sigma = n * np.eye(n) + Sigma.T @ Sigma
        Sigma_inverse = np.linalg.inv( Sigma )
        Sigma_inverse = (Sigma_inverse + Sigma_inverse.T) /2.0
        Sigmas = np.append( Sigmas, np.copy( Sigma_inverse.reshape(1,-1) ), axis=0 )
    return Sigmas

def get_future_reward(X, horizon, dt_outer, dynamics_params, params_policy, Sigma_invs):
    states, weights = initialize_sigma_points_jit(X)
    reward = 0
    H = 200#20#400
    def body(t, inputs):
        reward, states, weights = inputs
        mean_position = get_mean_jit( states, weights )
        solution = policy_jit( params_policy, Sigma_invs, mean_position )
        next_states_expanded, next_weights_expanded = sigma_point_expand_jit( states, weights, solution, dt_outer, dynamics_params)#, gps )        
        next_states, next_weights = sigma_point_compress_jit( next_states_expanded, next_weights_expanded )
        states = next_states
        weights = next_weights
        reward = reward + reward_UT_Mean_Evaluator_basic_jit( states, weights )
        return reward, states, weights
    
    return lax.fori_loop( 0, H, body, (reward, states, weights) )[0]

    # for i in range(H):
    #     mean_position = get_mean_jit( states, weights )
    #     solution = policy_jit( params_policy, Sigma_invs, mean_position )
    #     next_states_expanded, next_weights_expanded = sigma_point_expand_jit( states, weights, solution, dt_outer, dynamics_params)#, gps )        
    #     next_states, next_weights = sigma_point_compress_jit( next_states_expanded, next_weights_expanded )
    #     states = next_states
    #     weights = next_weights
    #     reward = reward + reward_UT_Mean_Evaluator_basic_jit( states, weights )
    # return reward

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
n = 4
N = 30
H = 20
lr_rate = 1.0##0.01
key = random.PRNGKey(100)
key, subkey = random.split(key)
param_w = random.uniform(subkey, shape=(N,1))[:,0] - 0.5#+ 0.5#+ 2.0  #0.5 work with Lr: 5.0
key, subkey = random.split(key)
param_mu = random.uniform(subkey, shape=(4,N))- 0.5 * np.ones((4,N)) #- 3.5 * np.ones((4,N))
# param_Sigma = generate_psd_params() # 10,N
Sigma_invs = generate_psd_matrix_inverse(n,N)
params_policy = np.append( param_w, param_mu.reshape(-1,1)[:,0] )

t = 0
dt_inner = 0.02
dt_outer = 0.06
tf = 4.0#4.0

state = np.copy(env.get_state())
t0 = time.time()
action = policy_jit( params_policy, Sigma_invs, state)
print(f"Policy JITed in time: {time.time()-t0}")
get_future_reward( state, H, dt_outer, dynamics_params, params_policy, Sigma_invs )
get_future_reward_grad( state, H, dt_outer, dynamics_params, params_policy, Sigma_invs )

t0 = time.time()
get_future_reward_grad_jit( state, H, dt_outer, dynamics_params, params_policy, Sigma_invs)
print(f"time jit for: {time.time()-t0}")
# exit()

optimize_offline = False
use_scipy = True
use_custom_gd = True

t0 = time.time()
get_future_reward_minimize = lambda params: get_future_reward( state, H, dt_outer, dynamics_params, params, Sigma_invs )
get_future_reward_minimize_jit = jit(get_future_reward_minimize)
reward = get_future_reward_minimize_jit( params_policy )
print(f"time jit for: {time.time()-t0}")
print(f"reward init:{ reward }")
if (optimize_offline):
    #train using scipy ###########################
    if use_scipy:
        t0 = time.time()
        res = minimize( get_future_reward_minimize_jit, params_policy, method='BFGS', tol=1e-8 ) #1e-8
        # params_policy = res.x
        print(f"time minimize for: {time.time()-t0}")
        print(f"reward final scipy : { get_future_reward_minimize_jit( res.x ) }")

    if use_custom_gd:
        for i in range(100):
            param_policy_grad = get_future_reward_grad_jit( state, H, dt_outer, dynamics_params, params_policy, Sigma_invs)
            param_policy_grad = np.clip( param_policy_grad, -2.0, 2.0 )
            params_policy = params_policy - lr_rate * param_policy_grad
            params_policy =  np.clip( params_policy, -10, 10 )
        print(f"reward final GD : { get_future_reward_minimize_jit( params_policy ) }")
    ##################################
# exit()

while t < tf:
    
    action = policy_jit( params_policy, Sigma_invs, state )
    next_state = step(state, action, dynamics_params, dt_inner)
    t0 = time.time()
    
    reward = get_future_reward_minimize_jit( params_policy )
    # w_grad, mu_grad, Sigma_grad = unconstrained_update(reward)
    
    param_policy_grad = get_future_reward_grad_jit( state, H, dt_outer, dynamics_params, params_policy, Sigma_invs)
    print(f"time reward :{ time.time()-t0 }, grad:{np.max(np.abs(param_policy_grad))}, reward: {reward}")
    if not optimize_offline:
        param_policy_grad = np.clip( param_policy_grad, -2.0, 2.0 )
        params_policy = params_policy - lr_rate * param_policy_grad
        params_policy =  np.clip( params_policy, -10, 10 )
   
    env.set_state( (next_state[0,0].item(), next_state[1,0].item(), next_state[2,0].item(), next_state[3,0].item()) )
    env.render()  
    
    state = np.copy(next_state)
    t = t + dt_inner
    