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
from cartpole_new.cartpole_policy import policy, policy_grad, random_exploration, Sum_of_gaussians_initialize
from cartpole_new.ut_utils.ut_utils import *

# visualization
from robot_models.custom_cartpole_mc_pilco import CustomCartPoleEnv
from robot_models.cartpole2D_mcpilco import step
from cartpole_new.gym_wrappers.record_video import RecordVideo

key = random.PRNGKey(2)

@jit
def get_future_reward(X, params_policy, dt):
    states, weights, weights_cov = initialize_sigma_points(X)
    reward = 0
    
    states = np.copy(X)
        
    def body(t, inputs):
        reward, states, weights, weights_cov = inputs    
        # mean_position = get_mean( states, weights )
        # solution = policy( mean_position, params_policy )
        # next_states_expanded, next_weights_expanded, next_weights_cov_expanded = sigma_point_expand( states, weights, weights_cov, solution, dt)
        # next_states, next_weights, next_weights_cov = sigma_point_compress( next_states_expanded, next_weights_expanded, next_weights_cov_expanded )
        # states = next_states
        # weights = next_weights
        # weights_cov = next_weights_cov
        # reward = reward + reward_UT_Mean_Evaluator_basic( states, weights, weights_cov )
        
        # direct deterministic
        action = policy( states, params_policy ).reshape(-1,1)
        states = step(states, action, dt)
        # reward = reward + mc_pilco_reward(states)
        reward = reward + compute_reward(states)
        # reward = reward + np.square(states[0,0])
        
        return reward, states, weights, weights_cov
    
    return lax.fori_loop( 0, H, body, (reward, states, weights, weights_cov) )[0]

@jit
def get_future_reward_grad(X, params_policy, dt):
    return grad(get_future_reward, 1)(X, params_policy, dt)


# Set up environment
env_to_render = CustomCartPoleEnv(render_mode="human")
env = RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/Research/FORESEE/", name_prefix="cartpole_sigma_test_ideal" )
observation, info = env.reset(seed=42)

key = random.PRNGKey(100)
key, subkey = random.split(key)
policy_type = 'with angles'
key, params_policy =  Sum_of_gaussians_initialize(subkey, state_dim=4, input_dim=1, type = policy_type, lengthscale = 1)

t = 0
dt_inner = 0.02
dt_outer = 0.02
tf = 3.0#H * dt_outer
H = int( tf/dt_inner )
# H = 10
grad_clip = 2.0

state = np.copy(env.get_state())

# run once

# optimization parameters
optimize_offline = True
use_adam = True
use_custom_gd = False
use_jax_scipy = False
n_restarts = 50#100
iter_adam = 4000
adam_start_learning_rate = 0.05#0.001
custom_gd_lr_rate = 0.005#0.5
#0.005 for compute_reward, grad_clip = 2.0

t0 = time.time()
reward = get_future_reward( state, params_policy, dt_outer)
grads = get_future_reward_grad( state, params_policy, dt_outer )
print(f"initial reward: {reward}, grad:{np.max(np.abs(grads))}, time to jit:{ time.time()-t0 }")
# exit()

if (optimize_offline):
    #train using scipy ###########################

    if use_custom_gd:
        for i in range(10000):
            param_policy_grad = get_future_reward_grad( state, params_policy, dt_inner)
            param_policy_grad = np.clip( param_policy_grad, -grad_clip, grad_clip )
            params_policy = params_policy - custom_gd_lr_rate * param_policy_grad
            # params_policy =  np.clip( params_policy, -10, 10 )
        print(f"reward final GD : { get_future_reward( state, params_policy, dt_outer ) }")

    if use_jax_scipy:
        minimize_function = lambda params: get_future_reward(state, params, dt_outer )
        solver = jaxopt.ScipyMinimize(fun=minimize_function, maxiter=iter_adam)
        params_policy, cost_state = solver.run(params_policy)
        print(f"Jaxopt state: {cost_state}")

    if use_adam:
        cost = get_future_reward( state, params_policy, dt_outer )
        best_params = np.copy(params_policy)
        best_cost = np.copy(cost)
        for j in range(n_restarts):

            key, params_policy = Sum_of_gaussians_initialize(key, state_dim=4, input_dim=1, type = policy_type, lengthscale = 1)

            cost = get_future_reward( state, params_policy, dt_outer )
            cost_initial = np.copy(cost)
            best_cost_local = np.copy(cost_initial)
            best_params_local = np.copy(params_policy)

            optimizer = optax.adam(adam_start_learning_rate)
            opt_state = optimizer.init(params_policy)

            # # Exponential decay of the learning rate.
            # scheduler = optax.exponential_decay(
            #     init_value=adam_start_learning_rate, 
            #     transition_steps=1000,
            #     decay_rate=0.999)

            # # Combining gradient transforms using `optax.chain`.
            # gradient_transform = optax.chain(
            #     optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
            #     optax.scale_by_adam(),  # Use the updates from adam.
            #     optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
            #     # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            #     optax.scale(-1.0)
            # )

            # opt_state = gradient_transform.init(params_policy)
            
            for i in range(iter_adam + 1):
                t0 = time.time()
                cost = get_future_reward( state, params_policy, dt_outer )
                
                if (cost<best_cost):
                    best_cost = np.copy(cost)
                    best_params = np.copy(params_policy)
                if (cost<best_cost_local):
                    best_cost_local = np.copy(cost)
                    best_params_local = np.copy(params_policy)
                if i==iter_adam:
                    continue
                
                grads = get_future_reward_grad( state, params_policy, dt_outer )
                
                updates, opt_state = optimizer.update(grads, opt_state)
                # updates, opt_state = gradient_transform.update(grads, opt_state)
                
                params_policy = optax.apply_updates(params_policy, updates)
                # if i%100==0:
                #     print(f"i:{i}, cost:{cost}, grad:{np.max(np.abs(grads))}")

                # print(f"time: {time.time()-t0}, cost:{cost}")
            print(f"run: {j}, cost initial:{cost_initial}, best cost local:{best_cost_local}, cost final:{best_cost}")
        
        params_policy = np.copy(best_params)
            
        with open('new_ideal.npy', 'wb') as f:
            np.save(f, best_params)    

action = 0
input("Press Enter to continue...")
while t < tf:

    if not optimize_offline:
        # tune parameters
        state_dim = 5
        num_basis = 200
        
        
        reward, param_policy_grad = get_future_reward( state, params_policy, dt_outer ), get_future_reward_grad( state, params_policy, dt_outer )
        centers = params_policy[state_dim:state_dim+num_basis*state_dim].reshape(state_dim, num_basis)
        weights = params_policy[-num_basis:]
        centers_grad = param_policy_grad[state_dim:state_dim+num_basis*state_dim].reshape(state_dim, num_basis)
        print(f"reward:{reward}, grad: {np.max(np.abs(param_policy_grad))}, action:{action}, params min:{np.min(params_policy)}, max:{np.max(params_policy)}, weights_max = {np.max(np.abs(weights))}, grad_max:{ np.max(np.abs( centers_grad[3:4,:] )) }")
        param_policy_grad = np.clip( param_policy_grad, -grad_clip, grad_clip )
        # if np.linalg.norm( param_policy_grad ) > grad_clip * param_policy_grad.size :
        #     param_policy_grad = param_policy_grad / np.linalg.norm(param_policy_grad) * grad_clip * param_policy_grad.size
        params_policy = params_policy - custom_gd_lr_rate * param_policy_grad
        

        # clip is needed
        # res = minimize( get_future_reward_jit, params_policy, method='BFGS', tol=1e-8, options=dict(maxiter=maxiter) ) #1e-8
        # params_policy = res.x
   

    action = policy( state, params_policy ).reshape(-1,1)
    print(f"theta:{ state[2,0]*180/np.pi }, input:{ action }")#, state:{ state.T }")
    next_state = step(state, action, dt_inner)
    env.set_state( (next_state[0,0].item(), next_state[1,0].item(), next_state[2,0].item(), next_state[3,0].item()) )
    env.render()  
    
    state = np.copy(next_state)
    t = t + dt_inner
env.close()

            
    ##################################
# exit()


# mean_position = get_mean( states, weights )
# solution = policy( mean_position, params_policy )
# next_states_expanded, next_weights_expanded, next_weights_cov_expanded = sigma_point_expand( states, weights, weights_cov, solution, dt)
# next_states, next_weights, next_weights_cov = sigma_point_compress( next_states_expanded, next_weights_expanded, next_weights_cov_expanded )
# states = next_states
# weights = next_weights
# weights_cov = next_weights_cov
# reward = reward + reward_UT_Mean_Evaluator_basic( states, weights, weights_cov )
# # return np.sum(states)
# return reward