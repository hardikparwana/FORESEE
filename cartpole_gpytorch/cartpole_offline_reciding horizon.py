import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})

import cvxpy as cp
import torch
torch.autograd.set_detect_anomaly(True)

from utils.utils import *
from cp_utils.ut_utilsJIT import *
# from cp_utils.gp_utils import *
from cp_utils.gp_uni_utils import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from robot_models.custom_cartpole_constrained import CustomCartPoleEnv
from gym_wrappers.record_video import RecordVideo
from cartpole_policy import policy, traced_policy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning) 

def initialize_tensors(robot, param_w, param_mu, param_Sigma):
    x, x_dot, theta, theta_dot = robot.state
    robot.X_torch = torch.tensor( np.array([ x, x_dot, theta, theta_dot ]).reshape(-1,1), requires_grad = True, dtype=torch.float )
    robot.w_torch = torch.tensor( param_w, requires_grad = True, dtype=torch.float )
    robot.mu_torch = torch.tensor( param_mu, requires_grad = True, dtype=torch.float )
    robot.Sigma_torch = torch.tensor( param_Sigma, requires_grad = True, dtype=torch.float )

x_lim = 1.5
def get_future_reward(robot, gp, params):
           
    prior_states, prior_weights = initialize_sigma_points_JIT(robot.X_torch)
    states = [prior_states]
    weights = [prior_weights]
    reward = torch.tensor([0],dtype=torch.float)
    
    maintain_constraints = []
    improve_constraints = []    
    
    for i in range(H):  
        # print("hello")
        # Get mean position
        mean_position = traced_get_mean_JIT( states[i], weights[i] )
        # print("h1")
        if np.abs(mean_position[0].detach().numpy()) > 1.5:
            improve_constraints.append( torch.square( mean_position[0] ) )
            print(f"Become Infeasible at :{i}. Need to improve feasibility first")
            if i==0:
                print("Initial state violates the constraint. Can't do anything!")
                return maintain_constraints, improve_constraints, True, reward, True
                # exit()
            return maintain_constraints, improve_constraints, False, reward, False
        elif torch.square( mean_position[0] ) > x_lim**2 * 5.0 / 6.0:
            maintain_constraints.append( x_lim**2 - torch.square( mean_position[0] ) )
        
        # Get control input      
        solution = traced_policy( robot.w_torch, robot.mu_torch, robot.Sigma_torch, mean_position )
        # getGrad(params[0], l_bound = -20.0, u_bound = 20.0 )
        # getGrad(params[1], l_bound = -20.0, u_bound = 20.0 )
        # getGrad(params[2], l_bound = -20.0, u_bound = 20.0 )
        # print("h2")
        # Get expanded next state
        next_states_expanded, next_weights_expanded = sigma_point_expand_JIT( states[i], weights[i], solution, torch.tensor(dt_outer, dtype=torch.float), gp)#, gps )        
        # print("h3")
        # Compress back now
        next_states, next_weights = traced_sigma_point_compress_JIT( next_states_expanded, next_weights_expanded )
        # print("h4")
        # Store states and weights
        states.append( next_states ); weights.append( next_weights )
        # print("h5")
        # Get reward 
        reward = reward + traced_reward_UT_Mean_Evaluator_basic( states[i+1], weights[i+1] )
        # print("reward", reward[-1])
        # print("h6")
        
    return maintain_constraints, improve_constraints, True, reward, False
        
    # return reward


def constrained_update( objective, maintain_constraints, improve_constraints, params ) :
    
    num_params = params[0].detach().numpy().size + params[1].detach().numpy().size + params[2].detach().numpy().size
    d = cp.Variable((num_params,1))
    
    # Get Performance optimal direction
    # try:
    objective.sum().backward(retain_graph = True) 
    w_grad = getGrad(params[0], l_bound = -20.0, u_bound = 20.0 )
    mu_grad = getGrad(params[1], l_bound = -20.0, u_bound = 20.0 )
    Sigma_grad = getGrad(params[2], l_bound = -20.0, u_bound = 20.0 )
    objective_grad = np.append( np.append( w_grad.reshape(1,-1), mu_grad.reshape(1,-1), axis = 1 ), Sigma_grad.reshape(1,-1) , axis = 1)
    # except:
    #     objective_grad = np.zeros( num_params ).reshape(1,-1)
    
    # Get constraint improve direction # assume one at a time
    improve_constraint_direction = np.zeros( num_params ).reshape(1,-1)
    for i, constraint in enumerate( improve_constraints):
        constraint.sum().backward(retain_graph=True)
        w_grad = getGrad(params[0], l_bound = -20.0, u_bound = 20.0 )
        mu_grad = getGrad(params[1], l_bound = -20.0, u_bound = 20.0 )
        Sigma_grad = getGrad(params[2], l_bound = -20.0, u_bound = 20.0 )
        improve_constraint_direction = improve_constraint_direction + np.append( np.append( w_grad.reshape(1,-1), mu_grad.reshape(1,-1), axis = 1 ), Sigma_grad.reshape(1,-1) , axis = 1)
    
    # Get allowed directions
    N = len(maintain_constraints)
    if N>0:
        d_maintain = np.zeros((N,num_params))#cp.Variable( (N, num_params) )
        constraints = []
        for i, constraint in enumerate(maintain_constraints):
            constraint.sum().backward(retain_graph=True)
            w_grad = getGrad(params[0], l_bound = -20.0, u_bound = 20.0 )
            mu_grad = getGrad(params[1], l_bound = -20.0, u_bound = 20.0 )
            Sigma_grad = getGrad(params[2], l_bound = -20.0, u_bound = 20.0 )
            d_maintain[i,:] = np.append( np.append( w_grad.reshape(1,-1), mu_grad.reshape(1,-1), axis = 1 ), Sigma_grad.reshape(1,-1) , axis = 1)[0]
            
            if constraints ==[]: 
                constraints = constraint.detach().numpy().reshape(-1,1)
            else:
                constraints = np.append( constraints, constraint.detach().numpy().reshape(-1,1), axis = 0 )       

        const = [ constraints + d_maintain @ d >= 0 ]
        const += [ cp.sum_squares( d ) <= 200 ]
        if len(improve_constraint_direction)>0:
            obj = cp.Minimize( improve_constraint_direction @ d )
        else:
            obj = cp.Minimize(  objective_grad @ d  )
        problem = cp.Problem( obj, const )    
        problem.solve( solver = cp.GUROBI )    
        if problem.status != 'optimal':
            print("Cannot Find feasible direction")
            exit()
        
        # print("update direction: ", d.value.T)
        
        return d.value
    
    else:
        if len( improve_constraints ) > 0:
            obj = cp.Maximize( improve_constraint_direction @ d )
            # print("update direction: ", -improve_constraint_direction.reshape(-1,1).T)
            return -improve_constraint_direction.reshape(-1,1)
        else:
            return -objective_grad.reshape(-1,1)

def generate_psd_params():
    n = 4
    N = 50
    
    diag = np.random.rand(n) + n
    off_diag = np.random.rand(int( (n**2-n)/2.0 ))
    params = np.append(diag, off_diag, axis = 0).reshape(1,-1)
    
    for i in range(1,50):
        # Diagonal elements
        params_temp = np.random.rand( int(n + (n**2 -n)/2.0) ).reshape(1,-1)
        
        # ## lower Off-diagonal
        # off_diag = np.random.rand(int( (n**2-n)/2.0 ))
        
        # params_temp = np.append(diag, off_diag, axis = 0).reshape(1,-1)
        params = np.append( params, params_temp, axis = 0 )
    
    return params

# Set up environment
env_to_render = CustomCartPoleEnv(render_mode="human")#rgb_array
env = RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/", name_prefix="cartpole_constrained_H20" )
observation, info = env.reset(seed=42)

polemass_length, gravity, length, masspole, total_mass, tau = torch.tensor(env.polemass_length), torch.tensor(env.gravity), torch.tensor(env.length), torch.tensor(env.masspole), torch.tensor(env.total_mass), torch.tensor(env.tau)

# Initialize sim parameters
t = 0
dt_inner = 0.02
dt_outer = 0.06 # 0.02 # first video with 0.06
gp_learn_loop = 20
outer_loop = 4#4#10 #2

# Initialize parameters
H_learning_gp = 60
H_policy_run_time = H_learning_gp * outer_loop
N = 50
H = 30#20 # prediction horizon

np.random.seed(0)
param_w = np.random.rand(N) - 0.5#+ 0.5#+ 2.0  #0.5 work with Lr: 5.0
param_mu = np.random.rand(4,N) - 0.5 * np.ones((4,N)) #- 3.5 * np.ones((4,N))
param_Sigma = generate_psd_params()
initialize_tensors( env, param_w, param_mu, param_Sigma )
# param_Sigma = np.random.rand(4,N)

lr_rate = 0.05 #0.4 #0.1 #0.5#0.001 #0.5
noise = torch.tensor(0.1, dtype=torch.float)
first_run = True
# X = torch.rand(4).reshape(-1,1)




train_x = []#np.array( [ 0, 0, 0, 0, 0 ] ).reshape(1,-1)
train_y = []#np.array( [ 0, 0, 0, 0 ] ).reshape(1, -1)
gp = []
likelihoods = []
for i in range(4):
    likelihoods.append(gpytorch.likelihoods.GaussianLikelihood())
    gp.append(ExactGPModel(torch.tensor(train_x, dtype=torch.float), torch.tensor(train_y, dtype=torch.float), likelihoods[-1]))
    gp[-1].eval()
    likelihoods[-1].eval()

# GP initialization
# likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=4)
# # self.train_x = np.array( [ 0, 0, 1.0, 0.0 ] ).reshape(1,-1)
# train_x = []#np.array( [ 0, 0, 0, 0, 0 ] ).reshape(1,-1)
# train_y = []#np.array( [ 0, 0, 0, 0 ] ).reshape(1, -1)
# gp = MultitaskGPModel(torch.tensor(train_x, dtype=torch.float), torch.tensor(train_y, dtype=torch.float), likelihood, num_tasks=4)
# gp.eval()
# likelihood.eval()

def simulate_scenario( gp, train_x, train_y, use_policy=False, randomize=False, run_name = "gp_fit.png" ):
    
    observation, info = env.reset(seed=42)

    Xs = np.copy(env.get_state())
    Us = []
    # train_x = []#np.array( [ 0, 0, 0, 0, 0 ] ).reshape(1,-1)
    # train_y = []

    cur_pose = np.copy(env.get_state())
    prev_pose = np.copy(cur_pose)
    
    for i in range(H_learning_gp): #300 time steps
                       
        # Find input
        state = env.get_state()        
        state_torch = torch.tensor( state, dtype=torch.float )
    
        if use_policy:
            if randomize:
                if (np.random.rand()>0.5):
                    action = policy( env.w_torch, env.mu_torch, env.Sigma_torch, state_torch )
                else:
                    # action = 5*torch.tensor(env.action_space.sample()-0.5)#   2*torch.rand(1)
                    action = torch.tensor(10*(np.random.rand() - 0.5), dtype=torch.float)
            else:
                action = policy( env.w_torch, env.mu_torch, env.Sigma_torch, state_torch )
        else:
            action = torch.tensor(10*(np.random.rand() - 0.5),dtype=torch.float)
            
        
        if (abs(action.item()))>20:
            print("ERROR*************************")
            exit()
        # print("action", action)
        
        observation, reward, terminated, truncated, info = env.step(action.item())        
        Xs = np.append( Xs, env.get_state(), axis = 1 )
        Us.append(action.item())
        
        env.render()

        # t = t + dt_inner
        
        if terminated or truncated:
            observation, info = env.reset()
            
        prev_pose = np.copy( cur_pose )
        cur_pose = np.copy(state)
        diff = cur_pose - prev_pose
        diff[2] = wrap_angle_numpy(diff[2])
        new_y = diff / dt_inner
        new_x = np.append( cur_pose, action.detach().numpy().reshape(-1,1), axis=0 )
        # print(f"state:{new_x.T}")
        if train_x == []:
            train_x = np.copy(new_x.reshape(1,-1))
            train_y = np.copy(new_y.reshape(1,-1))

        # Check if already have this state
        new_point = True
        for ij in range(np.shape(train_x)[0]):
            diff_new = np.linalg.norm(new_x.T-train_x[ij,:])
            # print(f"diff: {diff_new}")
            if diff_new<0.7: # if data very similar, do not add
                new_point = False
                break
        if new_point:
            train_x = np.append( train_x,  new_x.reshape(1,-1), axis = 0 )
            train_y = np.append( train_y,  new_y.reshape(1,-1), axis = 0 )
        
    # Now fit GP on Collected Data
    
    print("training")
    estimator_init = True
    
    train_x_torch = torch.tensor(train_x, dtype=torch.float, requires_grad=True)
    train_y_torch = torch.tensor(train_y, dtype=torch.float, requires_grad=True)

    for i in range(4):
        gp[i].set_train_data( train_x_torch, train_y_torch[:,i], strict = False)
    train_gp(gp,likelihoods, train_x_torch, train_y_torch, training_iterations = 30)
    for i in range(4):
        gp[i].eval()
        likelihoods[i].eval()
        
    # Visualize training results
    # print(f"train_x:{train_x_torch}")
    # print(f"train_y:{train_y_torch}")
    fig, ax = plt.subplots(2,2)
    pred1 = gp[0](train_x_torch)
    pred2 = gp[1](train_x_torch)
    pred3 = gp[2](train_x_torch)
    pred4 = gp[3](train_x_torch)
    mu_hat, cov = torch.cat((pred1.mean, pred2.mean, pred3.mean, pred4.mean)), torch.cat((torch.diagonal(pred1.covariance_matrix), torch.diagonal(pred2.covariance_matrix), torch.diagonal(pred3.covariance_matrix), torch.diagonal(pred4.covariance_matrix)))
    train_x_temp = train_x_torch.detach().numpy()
    train_y_temp = train_y_torch.detach().numpy()
    # mu = mu.detach().numpy()
    # cov = np.diag(cov.detach().numpy())
    covar = np.zeros((np.shape(train_x_temp)[0], 4))
    mu = np.zeros((np.shape(train_x_temp)[0], 4))
    for j in range(np.shape(train_x_temp)[0]):
        covar[j,0] = cov[np.shape(train_x_temp)[0]*0 + j]
        covar[j,1] = cov[np.shape(train_x_temp)[0]*1 + j]
        covar[j,2] = cov[np.shape(train_x_temp)[0]*2 + j]
        covar[j,3] = cov[np.shape(train_x_temp)[0]*3 + j]
        mu[j,0] = mu_hat[np.shape(train_x_temp)[0]*0 + j]
        mu[j,1] = mu_hat[np.shape(train_x_temp)[0]*1 + j]
        mu[j,2] = mu_hat[np.shape(train_x_temp)[0]*2 + j]
        mu[j,3] = mu_hat[np.shape(train_x_temp)[0]*3 + j]
    index_n = np.linspace(0,np.shape(train_x_temp)[0],np.shape(train_x_temp)[0])
    ax[0,0].plot( index_n, train_y_temp[:,0], 'r' )
    ax[0,0].plot( index_n, mu[:,0], 'g' )
    ax[0,0].fill_between(index_n, mu[:,0] - 2*np.sqrt(covar[:,0]), mu[:,0] + 2*np.sqrt(covar[:,0]), alpha=0.2, color = 'm')
    ax[0,1].plot( index_n, train_y_temp[:,1], 'r' )
    ax[0,1].plot( index_n, mu[:,1], 'g' )
    ax[0,1].fill_between(index_n, mu[:,1] - 2*np.sqrt(covar[:,1]), mu[:,1] + 2*np.sqrt(covar[:,1]), alpha=0.2, color = 'm')
    ax[1,0].plot( index_n, train_y_temp[:,2], 'r' )
    ax[1,0].plot( index_n, mu[:,2], 'g' )
    ax[1,0].fill_between(index_n, mu[:,2] - 2*np.sqrt(covar[:,2]), mu[:,2] + 2*np.sqrt(covar[:,2]), alpha=0.2, color = 'm')
    ax[1,1].plot( index_n, train_y_temp[:,3], 'r' )
    ax[1,1].plot( index_n, mu[:,3], 'g' )
    ax[1,1].fill_between(index_n, mu[:,3] - 2*np.sqrt(covar[:,3]), mu[:,3] + 2*np.sqrt(covar[:,3]), alpha=0.2, color = 'm')
    # plt.show()
    fig.savefig(run_name)
    
    return gp, train_x, train_y

    
    
    
def optimize_policy(env, gp, params, initialize_new_policy=False, lr_rate = 0.4):
    
    observation, info = env.reset(seed=42)
    
    param_w = params[0]
    param_mu = params[1]
    param_Sigma = params[2]
    
    if initialize_new_policy:
        np.random.seed(0)
        param_w = np.random.rand(N) - 0.5#+ 0.5#+ 2.0  #0.5 work with Lr: 5.0
        param_mu = np.random.rand(4,N) - 0.5 * np.ones((4,N)) #- 3.5 * np.ones((4,N))
        param_Sigma = generate_psd_params()
        initialize_tensors( env, param_w, param_mu, param_Sigma )     
       
    # Find input
    state = env.get_state()
    
    state_torch = torch.tensor( state, dtype=torch.float )        
    initialize_tensors( env, param_w, param_mu, param_Sigma )
    
    t = 0
    for i in range( H_policy_run_time ):
        # print("hello")
        if i==100:
            lr_rate = lr_rate / 2
        elif i == 200:
            lr_rate = lr_rate / 2
        
        if (i % outer_loop != 0) or i<1:
            # print(f"i:{i}")
            # move state forward
            state_torch = torch.tensor( state, dtype=torch.float )
            action = policy( env.w_torch, env.mu_torch, env.Sigma_torch, state_torch )
            
            if (abs(action.item()))>20:
                print("ERROR*************************")
                exit()
            observation, reward, terminated, truncated, info = env.step(action.item())
            env.render()
            t = t + dt_inner
            
            if terminated or truncated:
                observation, info = env.reset()
                
        else:
            
            initialize_tensors( env, param_w, param_mu, param_Sigma )

            success = False
            repeat_iter = 0
            while not success:
                if repeat_iter > 10:
                    break
                maintain_constraints, improve_constraints, success, reward, done = get_future_reward( env, gp, [env.w_torch, env.mu_torch, env.Sigma_torch] ) 
                if done:
                    break
                grads = constrained_update( reward, maintain_constraints, improve_constraints, [env.w_torch, env.mu_torch, env.Sigma_torch] )
                
                grads = np.clip( grads, -2.0, 2.0 )
                param_w = np.clip(param_w + lr_rate * grads[0:param_w.size][:,0], -10, 10 )
                param_mu = np.clip(param_mu + lr_rate * grads[param_w.size:param_w.size + param_mu.size].reshape( 4, 50 ), -10, 10 )
                param_Sigma = np.clip(param_Sigma + lr_rate * grads[param_w.size + param_mu.size:].reshape( 50, 10 ), -1.0, 1.0 )
                # print(f"params w:{param_w}, mu:{param_w}, Sigma:{param_Sigma}")

                initialize_tensors(env, param_w, param_mu, param_Sigma)
                repeat_iter += 1


            print(f" i:{i} Successfully made it feasible!") 
            
    return env, [param_w, param_mu, param_Sigma]

params = [param_w, param_mu, param_Sigma]
train_x = []
train_y = []
gp, train_x, train_y = simulate_scenario(gp, train_x, train_y, use_policy=False, randomize=False, run_name = "gp_fit0.png")
env, params = optimize_policy( env, gp, params, initialize_new_policy=False, lr_rate = lr_rate )

# env_to_render = CustomCartPoleEnv(render_mode="human")#rgb_array
# env = RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/", name_prefix="cartpole_constrained_H20" )
# observation, info = env.reset(seed=42)
# initialize_tensors( env, param_w, param_mu, param_Sigma )
gp, train_x, train_y = simulate_scenario(gp, train_x, train_y, use_policy=True, randomize=False, run_name = "gp_fit1.png")
env, params = optimize_policy( env, gp, params, initialize_new_policy=False, lr_rate = lr_rate )

gp, train_x, train_y = simulate_scenario(gp, train_x, train_y, use_policy=True, randomize=False, run_name = "gp_fit2.png")
env, params = optimize_policy( env, gp, params, initialize_new_policy=False, lr_rate = lr_rate )

gp, train_x, train_y = simulate_scenario(gp, train_x, train_y, use_policy=True, randomize=False, run_name = "gp_fit3.png")
env, params = optimize_policy( env, gp, params, initialize_new_policy=False, lr_rate = lr_rate )

gp, train_x, train_y = simulate_scenario(gp, train_x, train_y, use_policy=True, randomize=False, run_name = "gp_fit4.png")
env, params = optimize_policy( env, gp, params, initialize_new_policy=False, lr_rate = lr_rate )

gp, train_x, train_y = simulate_scenario(gp, train_x, train_y, use_policy=True, randomize=False, run_name = "gp_fit5.png")
# params = optimize_policy( env, gp, params, initialize_new_policy=False, lr_rate = lr_rate )
# initialize_tensors( env, param_w, param_mu, param_Sigma )

# tp1 = np.linspace( 0, dt_inner * Xs.shape[1], Xs.shape[1]  )
# figure1, axis1 = plt.subplots( 1 , 1)
# axis1.plot( tp1, Xs[0,:], 'k', label='Cart Position' )
# axis1.plot( tp1, Xs[1,:], 'r', label='Cart Velocity' )
# axis1.plot( tp1, Xs[2,:], 'g', label='Pole angle' )
# axis1.plot( tp1, Xs[3,:], 'c', label='Pole  velocity' )

# if True:
#     figure1.savefig("cartpole_states.eps")
#     figure1.savefig("cartpole_states.png")

        
# env.close_video_recorder()
env.close()