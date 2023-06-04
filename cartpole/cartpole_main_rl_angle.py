import time
import jax
import jax.numpy as np
from jax import random, grad, jit, vmap, vjp, lax, value_and_grad, jacfwd, jacrev
from jax.scipy.optimize import minimize
jax.config.update("jax_enable_x64", True)

# GP related libraries
jax.config.update("jax_enable_x64", True)

import scipy
from scipy.optimize import minimize as minimize_scipy

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})
from random_utils import generate_psd_params
from gp_utils import initialize_gp, train_gp, predict_gp

from utils.utils import *
from cartpole_policy_angle import policy, policy_jit, policy_grad
from ut_utils.ut_utils import *
from robot_models.custom_cartpole_constrained import CustomCartPoleEnv
# from robot_models.cartpole2D import step
from robot_models.cartpole2D_v2 import step
from gym_wrappers.record_video import RecordVideo

key = random.PRNGKey(2)

def get_future_reward(X, horizon, dynamics_params, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y):
    states, weights = initialize_sigma_points(X)
    reward = 0
    H = 50 # 300
    def body(t, inputs):
        reward, states, weights = inputs
        mean_position = get_mean( states, weights )
        solution = policy( params_policy, mean_position )
        next_states_expanded, next_weights_expanded = sigma_point_expand_with_gp( states, weights, solution, dynamics_params, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y )        
        next_states, next_weights = sigma_point_compress( next_states_expanded, next_weights_expanded )
        states = next_states
        weights = next_weights
        reward = reward + reward_UT_Mean_Evaluator_basic( states, weights )
        return reward, states, weights
    
    return lax.fori_loop( 0, H, body, (reward, states, weights) )[0]

# get_future_reward_grad = grad(get_future_reward)
get_future_reward_jit = jit(get_future_reward)
get_future_reward_grad = value_and_grad(get_future_reward, 4)
get_future_reward_grad_jit = jit(get_future_reward_grad)


# Set up environment
env_to_render = CustomCartPoleEnv(render_mode="rgb_array")
# env = RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/Research/FORESEE/videos/", name_prefix="cartpole_rl_test_full_2" )
exp_name = "cartpole_rl_tf10_angle_test4"
env = RecordVideo( env_to_render, video_folder="/home/dasc/hardik/FORESEE/videos/", name_prefix=exp_name )
observation, info = env.reset(seed=42)

polemass_length, gravity, length, masspole, total_mass, tau = env.polemass_length, env.gravity, env.length, env.masspole, env.total_mass, env.tau
dynamics_params = np.array([ polemass_length, gravity, length, masspole, total_mass])#, tau ])

# Initialize parameters
n = 5
N = 50
H = 20
lr_rate = 0.006#0.03#1.0#0.1#1.0##0.01
key = random.PRNGKey(100)
key, subkey = random.split(key)
param_w = 1.0*(random.uniform(subkey, shape=(N,1))[:,0] - 0.5)#+ 0.5#+ 2.0  #0.5 work with Lr: 5.0
key, subkey = random.split(key)
param_mu = random.uniform(subkey, shape=(n,N))- 0.5 * np.ones((n,N)) #- 3.5 * np.ones((4,N))
param_Sigma = generate_psd_params(n,N) # 10,N
params_policy = np.append( np.append( param_w, param_mu.reshape(-1,1)[:,0] ), param_Sigma.reshape(-1,1)[:,0]  )

t = 0
dt_inner = 0.06
dt_outer = 0.06
tf = 3.0#6.0#0.06#8.0#4.0

env.reset()
state = np.copy(env.get_state())
t0 = time.time()
action = policy_jit( params_policy, state)
print(f"Policy JITed in time: {time.time()-t0}")

# RUN this
# get_future_reward( state, H, dt_outer, dynamics_params, params_policy )
# # get_future_reward_grad( state, H, dt_outer, dynamics_pget_future_reward_minimizearams, params_policy )

# t0 = time.time()
# get_future_reward_grad_jit( state, H, dt_outer, dynamics_params, params_policy)
# print(f"time to JIT prediction: {time.time()-t0}")

# #train using scipy
# t0 = time.time()
# get_future_reward_minimize = lambda params: get_future_reward( state, H, dt_outer, dynamics_params, params )
# get_future_reward_minimize_jit = jit(get_future_reward_minimize)
# get_future_reward_minimize_jit( params_policy )
# print(f"time jit for: {time.time()-t0}")


# t0 = time.time()
# res = minimize( get_future_reward_minimize_jit, params_policy, method='BFGS', tol=1e-8 )
# print(f"optimized policy in time: {time.time() - t0} ")
# res = minimize_scipy( get_future_reward_minimize_jit, params_policy, method='Nelder-Mead', tol=1e-8 )
# params_policy = res.x

# train

# while t < tf:

#     action = policy_jit( params_policy, state ).reshape(-1,1)
#     # action_grad = np.max( policy_grad(params_policy, state) )
#     # print(f"action:{action}, state:{state[0,0]} grad:{np.max(action_grad)}")

#     next_state = step(state, action, dynamics_params, dt_inner)
#     env.set_state( (next_state[0,0].item(), next_state[1,0].item(), next_state[2,0].item(), next_state[3,0].item()) )
#     env.render()  
    
#     state = np.copy(next_state)
#     t = t + dt_inner
    
# plt.ioff()
# plt.show()
# exit()   

# Training Procedure
num_trials = 5
trial_horizon = 100

likelihoods = [0]*4
posteriors = [0]*4
parameter_states = [0]*4
learned_params = [0]*4
Ds = [0]*4
mus = [0]*4
stds = [0]*4
    
n = int(tf/dt_inner)
likelihoods[0], posteriors[0], parameter_states[0] = initialize_gp(num_datapoints = n)    
likelihoods[1], posteriors[1], parameter_states[1] = initialize_gp(num_datapoints = n) 
likelihoods[2], posteriors[2], parameter_states[2] = initialize_gp(num_datapoints = n) 
likelihoods[3], posteriors[3], parameter_states[3] = initialize_gp(num_datapoints = n) 

train_x = np.append(np.copy(state).reshape(1,-1), action.reshape(1,-1), axis=1)
train_y = np.copy(state).reshape(1,-1)

for k in range(num_trials):
    
    # Collect Data
    env.reset()
    state = np.copy(env.get_state())
    action = np.zeros((1,2))
    
    t = 0
    while t < tf:
        if k>0:
            action = policy_jit( params_policy, state ).reshape(-1,1)
        else:
            key, subkey = jax.random.split(key)
            action = jax.random.choice(subkey, np.array([5,-5])).reshape(-1,1)
        print(f"action :{action}")
        train_x = np.append( train_x, np.append(state.reshape(1,-1), action.reshape(1,-1), axis=1), axis=0 )
        next_state = step(state, action, dynamics_params, dt_inner)
        env.set_state( (next_state[0,0].item(), next_state[1,0].item(), next_state[2,0].item(), next_state[3,0].item()) )
        env.render()  
        state = np.copy(next_state)
        train_y = np.append(train_y, next_state.reshape(1,-1), axis=0)
        t = t + dt_inner
              
    # Learn GP
    for i in range(4):
        likelihoods[i], posteriors[i], learned_params[i], Ds[i] = train_gp( likelihoods[i], posteriors[i], parameter_states[i], train_x[1:,:], train_y[1:,i].reshape(-1,1) )      
    
    # Evaluate GPS
    plt.ioff()
    fig, ax = plt.subplots(4)
    for i in range(4):
        mus[i], stds[i] = predict_gp( likelihoods[i], posteriors[i], learned_params[i], Ds[i], train_x[1:,:] )
        ax[i].plot(np.linspace(0, train_x[1:,:].shape[0], train_x[1:,:].shape[0]), mus[i], 'g', label = 'Predicted values')
        ax[i].plot(np.linspace(0, train_x[1:,:].shape[0], train_x[1:,:].shape[0]), train_y[1:,i], 'r', label = 'True values')
        ax[i].fill_between(np.linspace(0, train_x[1:,:].shape[0], train_x[1:,:].shape[0]), mus[i] - 2* stds[i], mus[i] + 2* stds[i], alpha = 0.2, color="tab:blue", linewidth=1)
        ax[i].legend()
    plt.savefig(exp_name + "plot_gp_iter_"+str(k)+".png")
    
    # Train policy
    get_future_reward_minimize = lambda params: get_future_reward( state, H, dynamics_params, params, learned_params[0], learned_params[1], learned_params[2], learned_params[3], train_x[1:,:], train_y[1:,:] )
    get_future_reward_minimize_jit = jit(get_future_reward_minimize)
    
    t0 = time.time()
    get_future_reward_minimize_jit( params_policy )
    print(f"time to JIT prediction: {time.time()-t0}")
    
    t0 = time.time()
    res = minimize( get_future_reward_minimize_jit, params_policy, method='BFGS', tol=1e-3 )
    print(f"time to Optimize policy: {time.time()-t0}")
    params_policy = res.x
    
    # Evaluate Policy
    reward = get_future_reward_minimize_jit( params_policy )
    print(f"reward is : {reward}")


env.close()