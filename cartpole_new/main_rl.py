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
from cartpole_new.cartpole_policy import policy, policy_grad, random_exploration, Sum_of_gaussians_initialize
from cartpole_new.ut_utils import *

# visualization
from robot_models.custom_cartpole_constrained import CustomCartPoleEnv
from robot_models.cartpole2D import step
from gym_wrappers.record_video import RecordVideo

key = random.PRNGKey(2)

@jit
def get_future_reward(X, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y):
    states, weights = initialize_sigma_points(X)
    reward = 0
    def body(t, inputs):
        reward, states, weights = inputs
        mean_position = get_mean( states, weights )
        solution = policy( params_policy, mean_position )
        # print(f"input: {solution}")
        next_states_expanded, next_weights_expanded = sigma_point_expand( states, weights, solution, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y)
        next_states, next_weights = sigma_point_compress( next_states_expanded, next_weights_expanded )
        states = next_states
        weights = next_weights
        reward = reward + reward_UT_Mean_Evaluator_basic( states, weights )
        return reward, states, weights
    
    return lax.fori_loop( 0, H, body, (reward, states, weights) )[0]

@jit
def get_future_reward_grad():
    return grad(get_future_reward, 3)


# Set up environment
env_to_render = CustomCartPoleEnv(render_mode="human")
env = RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/Research/FORESEE/", name_prefix="cartpole_sigma_test_ideal" )
observation, info = env.reset(seed=42)

key = random.PRNGKey(100)
key, subkey = random.split(key)
params_policy =  Sum_of_gaussians_initialize(subkey, state_dim=4, input_dim=1, type = 'with angles', lengthscale = 1)

t = 0
H = 20
dt_inner = 0.05
dt_outer = 0.05
tf = H * dt_outer

state = np.copy(env.get_state())

# run once


# optimization parameters
optimize_offline = True
use_adam = True
use_custom_gd = False
n_restarts = 20#100
maxiter = 500000


if (optimize_offline):
    #train using scipy ###########################

    if use_custom_gd:
        for i in range(100):
            reward, param_policy_grad = get_future_reward_grad( state, dt_outer, params_policy)
            param_policy_grad = np.clip( param_policy_grad, -2.0, 2.0 )
            params_policy = params_policy - lr_rate * param_policy_grad
            params_policy =  np.clip( params_policy, -10, 10 )
        print(f"reward final GD : { get_future_reward_minimize_jit( params_policy ) }")

    if use_adam:
        for j in range(n_restarts):

            key, params_policy = initialize_parameters(key)

            cost = get_future_reward_minimize_jit(params_policy)
            cost_initial = np.copy(cost)
            best_cost_local = np.copy(cost_initial)
            best_params_local = np.copy(params_policy)

            start_learning_rate = 0.5#0.001

            # optimizer = optax.adam(start_learning_rate)
            # opt_state = optimizer.init(params_policy)

            # Exponential decay of the learning rate.
            scheduler = optax.exponential_decay(
                init_value=start_learning_rate, 
                transition_steps=1000,
                decay_rate=0.999)

            # Combining gradient transforms using `optax.chain`.
            gradient_transform = optax.chain(
                optax.clip_by_global_norm(2.0),  # Clip by the gradient by the global norm.
                optax.scale_by_adam(),  # Use the updates from adam.
                optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
                # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
                optax.scale(-1.0)
            )

            opt_state = gradient_transform.init(params_policy)


            iter_adam = 1000#100000
            for i in range(iter_adam + 1):#range(50000):
                t0 = time.time()
                cost = get_future_reward_minimize_jit(params_policy)
                if (cost<best_cost):
                    # print(f"cost:{cost}, prev;{best_cost}")
                    best_cost = np.copy(cost)
                    best_params = np.copy(params_policy)
                if (cost<best_cost_local):
                    best_cost_local = np.copy(cost)
                    best_params_local = np.copy(params_policy)
                if i==iter_adam:
                    continue
                # grads = jax.grad(get_future_reward_minimize_jit)(params_policy)
                grads = get_future_reward_minimize_grad_jit(params_policy)
                # updates, opt_state = optimizer.update(grads, opt_state)
                updates, opt_state = gradient_transform.update(grads, opt_state)
                params_policy = optax.apply_updates(params_policy, updates)
                if i%100==0:
                    print(f"i:{i}, cost:{cost}, grad:{np.max(np.abs(grads))}")
                # params_policy = model.get_parameters()
                # print(f"time: {time.time()-t0}, cost:{cost}")
            print(f"run: {j}, cost initial:{cost_initial}, best cost local:{best_cost_local}, cost final:{best_cost}")
        params_policy = best_params
            
        with open('adam_1_jax_test.npy', 'wb') as f:
            np.save(f, best_params)    

            
    ##################################
# exit()
    
    
    
# exit()

if with_gp:

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