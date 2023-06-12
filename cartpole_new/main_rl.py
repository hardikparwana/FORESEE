import time
import jax
import jax.numpy as np
from jax import random, grad, jit, lax
import optax
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})

from utils.utils import *
from gp_utils import initialize_gp, train_gp, predict_gp
from cartpole_new.cartpole_policy import policy, policy_grad, random_exploration, Sum_of_gaussians_initialize, random_policy
from cartpole_new.ut_utils.ut_utils import *

# visualization
from robot_models.custom_cartpole_mc_pilco import CustomCartPoleEnv
from robot_models.cartpole2D_mcpilco import step
from cartpole_new.gym_wrappers.record_video import RecordVideo

key = random.PRNGKey(2)

@jit
def get_future_reward(X, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y):
    states, weights, weights_cov = initialize_sigma_points(X)
    reward = 0
        
    def body(t, inputs):
        reward, states, weights, weights_cov = inputs
        mean_position = get_mean( states, weights )
        solution = policy( mean_position, params_policy )
        next_states_expanded, next_weights_expanded, next_weights_cov_expanded = sigma_point_expand_with_gp( states, weights, weights_cov, solution, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y)
        next_states, next_weights, next_weights_cov = sigma_point_compress( next_states_expanded, next_weights_expanded, next_weights_cov_expanded )
        states = next_states
        weights = next_weights
        weights_cov = next_weights_cov
        reward = reward + reward_UT_Mean_Evaluator_basic( states, weights, weights_cov )
        return reward, states, weights, weights_cov
    
    return lax.fori_loop( 0, H, body, (reward, states, weights, weights_cov) )[0]

@jit
def get_future_reward_grad(X, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y ):
    return grad(get_future_reward, 1)(X, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y)


def train_policy( key, use_custom_gd, use_adam, state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y ):

# if (optimize_offline):
#     #train using scipy ###########################

    if use_custom_gd:
        for i in range(100):
            param_policy_grad = get_future_reward_grad( state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y)
            param_policy_grad = np.clip( param_policy_grad, -2.0, 2.0 )
            params_policy = params_policy - custom_gd_lr_rate * param_policy_grad
            params_policy =  np.clip( params_policy, -10, 10 )
        print(f"reward final GD : { get_future_reward( state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y ) }")

    if use_adam:
        cost = get_future_reward( state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y )
        best_params = np.copy(params_policy)
        best_cost = np.copy(cost)
        for j in range(n_restarts):

            key, params_policy = Sum_of_gaussians_initialize(key, state_dim=4, input_dim=1, type = policy_type, lengthscale = 1)

            cost = get_future_reward( state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y )
            cost_initial = np.copy(cost)
            best_cost_local = np.copy(cost_initial)
            best_params_local = np.copy(params_policy)

            # optimizer = optax.adam(start_learning_rate)
            # opt_state = optimizer.init(params_policy)

            # Exponential decay of the learning rate.
            scheduler = optax.exponential_decay(
                init_value=adam_start_learning_rate, 
                transition_steps=1000,
                decay_rate=0.999)

            # Combining gradient transforms using `optax.chain`.
            gradient_transform = optax.chain(
                optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
                optax.scale_by_adam(),  # Use the updates from adam.
                optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
                # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
                optax.scale(-1.0)
            )

            opt_state = gradient_transform.init(params_policy)
            
            for i in range(iter_adam + 1):
                t0 = time.time()
                cost = get_future_reward( state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y)
                
                if (cost<best_cost):
                    best_cost = np.copy(cost)
                    best_params = np.copy(params_policy)
                if (cost<best_cost_local):
                    best_cost_local = np.copy(cost)
                    best_params_local = np.copy(params_policy)
                if i==iter_adam:
                    continue
                
                grads = get_future_reward_grad( state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y)
                
                # updates, opt_state = optimizer.update(grads, opt_state)
                updates, opt_state = gradient_transform.update(grads, opt_state)
                
                params_policy = optax.apply_updates(params_policy, updates)
                # if i%100==0:
                #     print(f"i:{i}, cost:{cost}, grad:{np.max(np.abs(grads))}")

                # print(f"time: {time.time()-t0}, cost:{cost}")
            print(f"run: {j}, cost initial:{cost_initial}, best cost local:{best_cost_local}, cost final:{best_cost}")
        
        params_policy = np.copy(best_params)
            
        with open('new_ideal.npy', 'wb') as f:
            np.save(f, best_params)    
        return key, params_policy


# Set up environment
exp_name = "cartpole_new_rl_test1"
env_to_render = CustomCartPoleEnv(render_mode="human")
env = RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/Research/FORESEE/", name_prefix="cartpole_sigma_test_ideal" )
observation, info = env.reset(seed=42)

key = random.PRNGKey(100)
key, subkey = random.split(key)
policy_type = 'with angles'
key, params_policy =  Sum_of_gaussians_initialize(subkey, state_dim=4, input_dim=1, type = policy_type, lengthscale = 1)

t = 0
H = 20
dt_inner = 0.05
dt_outer = 0.05
tf = H * dt_outer

state = np.copy(env.get_state())

# optimization parameters
optimize_offline = False
use_adam = True
use_custom_gd = False
n_restarts = 10#100
iter_adam = 10#4000
adam_start_learning_rate = 0.05#0.001
custom_gd_lr_rate = 1.0

# RL setup
num_trials = 5
random_threshold = np.array([0.5, 1.1, 1.1, 1.1, 1.1])

# GP setup
likelihoods = [0]*4
posteriors = [0]*4
parameter_states = [0]*4
learned_params = [0]*4
Ds = [0]*4
mus = [0]*4
stds = [0]*4
action = policy( state, params_policy ).reshape(-1,1)
train_x = np.append(np.copy(state).reshape(1,-1), action.reshape(1,-1), axis=1)
train_y = np.copy(state).reshape(1,-1)

# t0 = time.time()
# reward = get_future_reward( state, params_policy, dt_outer)
# grads = get_future_reward_grad( state, params_policy, dt_outer )
# print(f"initial reward: {reward}, grad:{np.max(np.abs(grads))}, time to jit:{ time.time()-t0 }")

for run in range(num_trials):
    
    # Collect Data
    env.reset()
    state = np.copy(env.get_state())
    action = np.zeros((1,1))
    
    # Run Policy and collect data
    t = 0
    while t < tf:
        
        key, subkey  = random.split(key)
        if (random.uniform( subkey ) < random_threshold[run] ):
            key, subkey  = random.split(key)
            action = random_policy(subkey).reshape(-1,1)
            print(f"action:{action}")
        else:
            action = policy( state, params_policy ).reshape(-1,1)
        train_x = np.append( train_x, np.append(state.reshape(1,-1), action.reshape(1,-1), axis=1), axis=0 )
        next_state = step(state, action, dt_inner)
        env.set_state( (next_state[0,0].item(), next_state[1,0].item(), next_state[2,0].item(), next_state[3,0].item()) )
        env.render()  
        state = np.copy(next_state)
        train_y = np.append(train_y, next_state.reshape(1,-1), axis=0)
        t = t + dt_inner
              
    # Learn GP
    for i in range(4):
        likelihoods[i], posteriors[i], parameter_states[i] = initialize_gp(num_datapoints = train_x.shape[0]) 
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
    plt.savefig(exp_name + "plot_gp_iter_"+str(run)+".png")
    
    # Train policy
    key, params_policy = train_policy( key, use_custom_gd = use_custom_gd, use_adam = use_adam, state = state, params_policy = params_policy, gp_params1 = learned_params[0], gp_params2 = learned_params[1], gp_params3 = learned_params[2], gp_params4 = learned_params[3], gp_train_x = train_x[1:,:], gp_train_y = train_y[1:,:] )
   
    # Evaluate Policy
    reward = get_future_reward( state, params_policy, learned_params[0], learned_params[1], learned_params[2], learned_params[3], train_x[1:,:], train_y[1:,:] )
    print(f"Run : {run} reward is : {reward}")


env.close()