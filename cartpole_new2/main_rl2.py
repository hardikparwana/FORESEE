import time
import jax
import jax.numpy as np
from jax import random, grad, jit, lax
import optax
import jaxopt
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})

from utils.utils import *
from cartpole_new2.gp_utils import * # initialize_gp, train_gp, predict_gp
from cartpole_new2.cartpole_policy import policy, policy9, policy_grad, random_exploration, Sum_of_gaussians_initialize, random_policy
from cartpole_new2.ut_utils.ut_utils import *

# visualization
from robot_models.custom_cartpole_mc_pilco import CustomCartPoleEnv
from robot_models.cartpole2D_mcpilco import step, get_next_states_from_dynamics
from cartpole_new2.gym_wrappers.record_video import RecordVideo



key = random.PRNGKey(2)

@jit
def get_future_reward(X, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y):
    states, weights, weights_cov = initialize_sigma_points(X)
    reward = 0
    
    gp0 = initialize_gp_prediction( gp_params1, gp_train_x, gp_train_y[:,0].reshape(-1,1) )
    gp1 = initialize_gp_prediction( gp_params2, gp_train_x, gp_train_y[:,1].reshape(-1,1) )
    gp2 = initialize_gp_prediction( gp_params3, gp_train_x, gp_train_y[:,2].reshape(-1,1) )
    gp3 = initialize_gp_prediction( gp_params4, gp_train_x, gp_train_y[:,3].reshape(-1,1) )
    
    def body(t, inputs):
        reward, states, weights, weights_cov = inputs
        # mean_position = get_mean( states, weights )
        control_inputs = policy9( states, params_policy )
        next_states_mean, next_states_cov = get_next_states_with_gp( states, control_inputs, [gp0, gp1, gp2, gp3] )
        next_states_expanded, next_weights_expanded, next_weights_cov_expanded = sigma_point_expand_with_mean_cov( next_states_mean, next_states_cov, weights, weights_cov)
        next_states, next_weights, next_weights_cov = sigma_point_compress( next_states_expanded, next_weights_expanded, next_weights_cov_expanded )
        states = next_states
        weights = next_weights
        weights_cov = next_weights_cov
        reward = reward + reward_UT_Mean_Evaluator_basic( states, weights, weights_cov )
        return reward, states, weights, weights_cov
    
    return lax.fori_loop( 0, H, body, (reward, states, weights, weights_cov) )[0]

# @jit
def get_future_reward_for_compare(X, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y):
    states, weights, weights_cov = initialize_sigma_points(X)
    reward = 0
    new_states_true = np.copy(states)
    predicted_states_mus = np.copy(states)
    predicted_states_covs = np.copy(states)*0
    
    gp0 = initialize_gp_prediction( gp_params1, gp_train_x, gp_train_y[:,0].reshape(-1,1) )
    gp1 = initialize_gp_prediction( gp_params2, gp_train_x, gp_train_y[:,1].reshape(-1,1) )
    gp2 = initialize_gp_prediction( gp_params3, gp_train_x, gp_train_y[:,2].reshape(-1,1) )
    gp3 = initialize_gp_prediction( gp_params4, gp_train_x, gp_train_y[:,3].reshape(-1,1) )
    
    for i in range(H):
        control_inputs = policy9( states, params_policy )
        print(f"  prediction: action: {control_inputs[0,0]}, state: {states[:,0]}")
        next_states_mean, next_states_cov = get_next_states_with_gp( states, control_inputs, [gp0, gp1, gp2, gp3] )
     
        # # find smallest dist:
        # test_x = np.append( states, control_inputs.reshape(1,-1), axis=0 ).T
        # for j in range(9):
        #     diff = gp_train_x - test_x[j,:].reshape(1,-1)
        #     dist = np.min( np.linalg.norm(diff, axis=1)  )
        #     print(f" dist:{dist}, cov:{ next_states_cov[3,j] }, diff:{diff[:,0]} ")        
                
        new_states_true = np.append( new_states_true, get_next_states_from_dynamics(states, control_inputs, dt_outer) , axis=1 )
        predicted_states_mus = np.append( predicted_states_mus, next_states_mean, axis=1 )
        predicted_states_covs = np.append( predicted_states_covs, next_states_cov, axis=1 )
        
        next_states_expanded, next_weights_expanded, next_weights_cov_expanded = sigma_point_expand_with_mean_cov( next_states_mean, next_states_cov*0, weights, weights_cov)
        next_states, next_weights, next_weights_cov = sigma_point_compress( next_states_expanded, next_weights_expanded, next_weights_cov_expanded )
        states = next_states
        weights = next_weights
        weights_cov = next_weights_cov
        reward = reward + reward_UT_Mean_Evaluator_basic( states, weights, weights_cov )
    
    return reward, new_states_true, predicted_states_mus, predicted_states_covs

# solution = policy( mean_position, params_policy )
# next_states_expanded, next_weights_expanded, next_weights_cov_expanded = sigma_point_expand_rk4( states, weights, weights_cov, solution, dt_outer)
# next_states_expanded, next_weights_expanded, next_weights_cov_expanded = sigma_point_expand_with_gp( states, weights, weights_cov, solution, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y)

@jit
def get_future_reward_grad(X, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y ):
    return grad(get_future_reward, 1)(X, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y)


def train_policy( key, use_custom_gd, use_jax_scipy, use_adam, adam_start_learning_rate, init_state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y ):

# if (optimize_offline):
#     #train using scipy ###########################

    if use_custom_gd:
        for i in range(100):
            param_policy_grad = get_future_reward_grad( init_state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y)
            param_policy_grad = np.clip( param_policy_grad, -grad_clip, grad_clip )
            params_policy = params_policy - custom_gd_lr_rate * param_policy_grad
            # params_policy =  np.clip( params_policy, -10, 10 )
        print(f"reward final GD : { get_future_reward( init_state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y ) }")

    if use_jax_scipy:
        minimize_function = lambda params: get_future_reward( init_state, params, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y )
        solver = jaxopt.ScipyMinimize(fun=minimize_function, maxiter=iter_adam)
        params_policy, cost_state = solver.run(params_policy)
        print(f"Jaxopt state: {cost_state}")

    if use_adam:
        # print(f"inside adam")
        # t0 = time.time()
        cost = get_future_reward( init_state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y )
        # print(f"adam first reward: {time.time()-t0}")
        best_params = np.copy(params_policy)
        best_cost = np.copy(cost)
        costs_adam = []
        for j in range(n_restarts):
            
            if (j>0):
                key, params_policy = Sum_of_gaussians_initialize(key, state_dim=4, input_dim=1, type = policy_type, lengthscale = 1)

            cost = get_future_reward( init_state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y )
            cost_run = [cost]
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
                cost = get_future_reward( init_state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y)
                cost_run.append(cost)
                if (cost<best_cost):
                    best_cost = np.copy(cost)
                    best_params = np.copy(params_policy)
                if (cost<best_cost_local):
                    best_cost_local = np.copy(cost)
                    best_params_local = np.copy(params_policy)
                if i==iter_adam:
                    continue
                # print(f"inside adam")
                # t0 = time.time()
                grads = get_future_reward_grad( init_state, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y)
                # print(f"adam first grad: {time.time()-t0}")
                
                updates, opt_state = optimizer.update(grads, opt_state)
                # updates, opt_state = gradient_transform.update(grads, opt_state)
                
                params_policy = optax.apply_updates(params_policy, updates)
                # if i%100==0:
                #     print(f"i:{i}, cost:{cost}, grad:{np.max(np.abs(grads))}")

                # print(f"time: {time.time()-t0}, cost:{cost}")
            costs_adam.append( cost_run )
            print(f"run: {j}, cost initial:{cost_initial}, best cost local:{best_cost_local}, cost final:{best_cost}")
        
        params_policy = np.copy(best_params)
            
        with open('new_rl.npy', 'wb') as f:
            np.save(f, best_params)    
        print(f" *************** NANs? :{np.any(np.isnan(params_policy)==True)} ")
    return key, params_policy, costs_adam


# Set up environment
exp_name = "cartpole_new2_rl2_test2_lr005_adamiter2000"
env_to_render = CustomCartPoleEnv(render_mode="human")
env = RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/Research/FORESEE/", name_prefix="cartpole_sigma_test_ideal" )
observation, info = env.reset(seed=42)

key = random.PRNGKey(100)
key, subkey = random.split(key)
policy_type = 'with angles'
key, params_policy =  Sum_of_gaussians_initialize(subkey, state_dim=4, input_dim=1, type = policy_type, lengthscale = 1)
action = policy( np.array([0,0,0.1,0]).reshape(-1,1), params_policy ).reshape(-1,1)
t = 0
dt_inner = 0.05#0.02
dt_outer = 0.05#0.02
tf = 3.0#H * dt_outer
H = int( tf/dt_inner )
# H = 10
grad_clip = 2.0

state = np.copy(env.get_state())

# optimization parameters
optimize_offline = True
use_adam = True
use_custom_gd = False
use_jax_scipy = False
n_restarts = 6#50#100
iter_adam = 2000#4000#1000
adam_start_learning_rate = 0.005#0.05#0.001
custom_gd_lr_rate = 0.005#0.5

# sometimes good with adam 1000, time 0.05

# RL setup
num_trials = 5
tf_trials = [3.0, 3.0, 3.0, 3.0, 3.0]
random_threshold = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# GP setup
likelihoods = [0]*4
posteriors = [0]*4
parameter_states = [0]*4
learned_params = [0]*4
Ds = [0]*4  
mus = [0]*4
stds = [0]*4
mus2 = [0]*4
stds2 = [0]*4
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
    state_init = np.copy(state)
    action = np.zeros((1,1))
    
    # Run Policy and collect data
    t = 0
    reward_trial = 0
    while t < tf_trials[run]:
        
        key, subkey  = random.split(key)
        if (random.uniform( subkey ) < random_threshold[run] ):
            key, subkey  = random.split(key)
            action = random_policy(subkey).reshape(-1,1)
            print(f"action:{action}")
        else:
            action = policy( state, params_policy ).reshape(-1,1)
            print(f"action policy:{action}, state:{state.T}")
            
        next_state = step(state, action, dt_inner)
        env.set_state( (next_state[0,0].item(), next_state[1,0].item(), next_state[2,0].item(), next_state[3,0].item()) )
        env.render()  
        
        train_x = np.append( train_x, np.append(state.reshape(1,-1), action.reshape(1,-1), axis=1), axis=0 )
        train_y = np.append(train_y, next_state.reshape(1,-1), axis=0)
        
        state = np.copy(next_state)
        reward_trial = reward_trial + mc_pilco_reward(state)
        
        t = t + dt_inner
    print(f"Trial reward: {reward_trial}")
    
    if (run>0):
        fig, ax = plt.subplots(4)
        for i in range(4):
            if 1:#( (i==1) or (i==3) ):
                gp_temp = initialize_gp_prediction( learned_params[i], train_x[1:,:], train_y[1:,i].reshape(-1,1) )
                pred_temp = gp_temp(train_x[1:,:])
                mus[i], stds[i] = pred_temp.mean(), np.sqrt(pred_temp.variance())
                mus2[i], stds2[i] = predict_gp( likelihoods[i], posteriors[i], learned_params[i], Ds[i], train_x[1:,:] )
                ax[i].plot(np.linspace(0, train_x[1:,:].shape[0], train_x[1:,:].shape[0]), mus[i], 'g', label = 'Predicted values')
                ax[i].plot(np.linspace(0, train_x[1:,:].shape[0], train_x[1:,:].shape[0]), mus2[i], 'c', label = '2nd Predicted values')
                ax[i].plot(np.linspace(0, train_x[1:,:].shape[0], train_x[1:,:].shape[0]), train_y[1:,i], 'r--', label = 'True values')
                ax[i].fill_between(np.linspace(0, train_x[1:,:].shape[0], train_x[1:,:].shape[0]), mus[i] - 2* stds[i], mus[i] + 2* stds[i], alpha = 0.2, color="tab:orange", linewidth=1)
                ax[i].fill_between(np.linspace(0, train_x[1:,:].shape[0], train_x[1:,:].shape[0]), mus2[i] - 2* stds2[i], mus2[i] + 2* stds2[i], alpha = 0.2, color="tab:green", linewidth=1)
                ax[i].legend()
        plt.savefig(exp_name + "run_plot_gp_iter_"+str(run)+".png")
              
    # Learn GP
    for i in range(4):
        if 1:#((i==1) or (i==3)):
            likelihoods[i], posteriors[i], parameter_states[i] = initialize_gp(num_datapoints = train_x.shape[0]) 
            likelihoods[i], posteriors[i], learned_params[i], Ds[i] = train_gp( likelihoods[i], posteriors[i], parameter_states[i], train_x[1:,:], train_y[1:,i].reshape(-1,1) )      
    
    # Evaluate GPS
    plt.ioff()
    fig, ax = plt.subplots(4)
    for i in range(4):
        if 1:#( (i==1) or (i==3) ):
            gp_temp = initialize_gp_prediction( learned_params[i], train_x[1:,:], train_y[1:,i].reshape(-1,1) )
            pred_temp = gp_temp(train_x[1:,:])
            mus[i], stds[i] = pred_temp.mean(), np.sqrt(pred_temp.variance())
            mus2[i], stds2[i] = predict_gp( likelihoods[i], posteriors[i], learned_params[i], Ds[i], train_x[1:,:] )
            ax[i].plot(np.linspace(0, train_x[1:,:].shape[0], train_x[1:,:].shape[0]), mus[i], 'g', label = 'Predicted values')
            ax[i].plot(np.linspace(0, train_x[1:,:].shape[0], train_x[1:,:].shape[0]), mus2[i], 'c', label = '2nd Predicted values')
            ax[i].plot(np.linspace(0, train_x[1:,:].shape[0], train_x[1:,:].shape[0]), train_y[1:,i], 'r--', label = 'True values')
            ax[i].fill_between(np.linspace(0, train_x[1:,:].shape[0], train_x[1:,:].shape[0]), mus[i] - 2* stds[i], mus[i] + 2* stds[i], alpha = 0.2, color="tab:orange", linewidth=1)
            ax[i].fill_between(np.linspace(0, train_x[1:,:].shape[0], train_x[1:,:].shape[0]), mus2[i] - 2* stds2[i], mus2[i] + 2* stds2[i], alpha = 0.2, color="tab:green", linewidth=1)
            ax[i].legend()
    plt.savefig(exp_name + "plot_gp_iter_"+str(run)+".png")
    
    t0 = time.time()
    reward, pred_states_true,pred_states_mus,pred_states_covs = get_future_reward_for_compare( state_init, params_policy, learned_params[0], learned_params[1], learned_params[2], learned_params[3], train_x[1:,:], train_y[1:,:] )
    print(f"first compare reward: {reward}, time: {time.time()-t0}")
    print(f"covariances in prediction: max:{np.max(np.abs(pred_states_covs))}")#, values: { pred_states_covs }")
    
    fig, ax = plt.subplots(4)
    for i in range(4):
        ax[i].plot(np.linspace(0, pred_states_true.shape[1], pred_states_true.shape[1]), pred_states_mus[i,:], 'g', label = 'Predicted values')
        ax[i].plot(np.linspace(0, pred_states_true.shape[1], pred_states_true.shape[1]), pred_states_true[i,:], 'r--', label = 'True values')
        ax[i].fill_between(np.linspace(0, pred_states_true.shape[1], pred_states_true.shape[1]), pred_states_mus[i,:] - 2* pred_states_covs[i,:], pred_states_mus[i,:] + 2* pred_states_covs[i,:], alpha = 0.2, color="tab:blue", linewidth=1)
        ax[i].legend()
    plt.savefig(exp_name + "predicted_plot_gp_iter_"+str(run)+".png")
    
    t0 = time.time()
    reward = get_future_reward( state_init, params_policy, learned_params[0], learned_params[1], learned_params[2], learned_params[3], train_x[1:,:], train_y[1:,:] )
    print(f"first reward: {reward}, time: {time.time()-t0}")
    t0 = time.time()
    grads = get_future_reward_grad( state_init, params_policy, learned_params[0], learned_params[1], learned_params[2], learned_params[3], train_x[1:,:], train_y[1:,:])
    print(f"first reward grad: time: {time.time()-t0}")
    
    # Train policy
    key, params_policy, costs_adam = train_policy( key, use_custom_gd = use_custom_gd, use_jax_scipy = use_jax_scipy, use_adam = use_adam, adam_start_learning_rate = adam_start_learning_rate, init_state = state_init, params_policy = params_policy, gp_params1 = learned_params[0], gp_params2 = learned_params[1], gp_params3 = learned_params[2], gp_params4 = learned_params[3], gp_train_x = train_x[1:,:], gp_train_y = train_y[1:,:] )
    fig, ax = plt.subplots(n_restarts)
    for i in range(n_restarts):
        ax[i].plot( costs_adam[i] )
    plt.savefig(exp_name + "adam_loss_iter_"+str(run)+".png")
    
   
    # Evaluate Policy
    reward = get_future_reward( state_init, params_policy, learned_params[0], learned_params[1], learned_params[2], learned_params[3], train_x[1:,:], train_y[1:,:] )
    print(f"Run : {run} reward is : {reward}")


env.close()

# TODOs
# 1. smaller time step
# 2. plot GP prediction vs true state during trial
# 3. use GP for 1st and 2nd state too. or do RK4 somehow
#4. more Adam iterations