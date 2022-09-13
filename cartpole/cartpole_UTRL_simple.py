import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})

import torch
torch.autograd.set_detect_anomaly(True)

from utils.utils import *
from ut_utils.ut_utilsJIT import *
from utils.mvgp_jit import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from robot_models.custom_cartpole import CustomCartPoleEnv
from gym_wrappers.record_video import RecordVideo
from cartpole_policy import policy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning) 

def initialize_tensors(robot, param_w, param_mu, param_Sigma):
    x, x_dot, theta, theta_dot = robot.state
    robot.X_torch = torch.tensor( np.array([ x, x_dot, theta, theta_dot ]).reshape(-1,1), requires_grad = True, dtype=torch.float )
    robot.w_torch = torch.tensor( param_w, requires_grad = True, dtype=torch.float )
    robot.mu_torch = torch.tensor( param_mu, requires_grad = True, dtype=torch.float )
    robot.Sigma_torch = torch.tensor( param_Sigma, requires_grad = True, dtype=torch.float )
    
traced_get_mean_JIT = []
traced_policy = []
traced_sigma_point_expand_JIT = []
traced_sigma_point_compress_JIT = []
traced_reward_UT_Mean_Evaluator_basic = []

def get_future_reward(robot, gp_params, K_invs, X_s, Y_s, noise, gps):
           
    prior_states, prior_weights = initialize_sigma_points_JIT(robot.X_torch)
    states = [prior_states]
    weights = [prior_weights]
    reward = torch.tensor([0],dtype=torch.float)
    
    global first_run, traced_get_mean_JIT, traced_policy, traced_sigma_point_expand_JIT, traced_sigma_point_compress_JIT, traced_reward_UT_Mean_Evaluator_basic

    start_t = 1
    if (first_run):
        print("**************************** t ******************************* ", t)
        i = 0
        print("Starting tracing")
        # Get mean position
        traced_get_mean_JIT = get_mean_JIT #torch.jit.trace( get_mean_JIT, (states[i], weights[i] ) )
        mean_position = traced_get_mean_JIT( states[i], weights[i] )
        
        # Get control input
        traced_policy = policy #torch.jit.trace( policy, ( robot.w_torch, robot.mu_torch, robot.Sigma_torch, mean_position ) )        
        solution = traced_policy( robot.w_torch, robot.mu_torch, robot.Sigma_torch, mean_position )
   
        # Get expanded next state
        traced_sigma_point_expand_JIT = sigma_point_expand_JIT #torch.trace.jit( sigma_point_expand_JIT, (torch.tensor(GA, dtype=torch.float), torch.tensor(PE, dtype=torch.float), gp_params, K_invs, noise, X_s, Y_s, states[i], weights[i], solution, dt_outer, dt_inner) )
        next_states_expanded, next_weights_expanded = sigma_point_expand_JIT( torch.tensor(GA, dtype=torch.float), torch.tensor(PE, dtype=torch.float), gp_params, K_invs, noise, X_s, Y_s, states[i], weights[i], solution, dt_outer, dt_inner, polemass_length, gravity, length, masspole, total_mass, tau) #, gps )       
            
        # Compress back now
        traced_sigma_point_compress_JIT = sigma_point_compress_JIT ### not used # torch.jit.trace( sigma_point_compress_JIT, ( next_states_expanded, next_weights_expanded ) )
        next_states, next_weights = traced_sigma_point_compress_JIT( next_states_expanded, next_weights_expanded )
        
        # Store states and weights
        states.append( next_states ); weights.append( next_weights )
            
        # Get reward 
        traced_reward_UT_Mean_Evaluator_basic = torch.jit.trace( reward_UT_Mean_Evaluator_basic, ( states[i+1], weights[i+1] ) )
        reward = reward + traced_reward_UT_Mean_Evaluator_basic( states[i+1], weights[i+1] )
        
        first_run = False
        print("TRACED!!!!")
    else:
        start_t = 0
        
    for i in range(start_t,H):  
        # Get mean position
        mean_position = traced_get_mean_JIT( states[i], weights[i] )
        
        # Get control input      
        solution = traced_policy( robot.w_torch, robot.mu_torch, robot.Sigma_torch, mean_position )
   
        # Get expanded next state
        next_states_expanded, next_weights_expanded = traced_sigma_point_expand_JIT( torch.tensor(GA, dtype=torch.float), torch.tensor(PE, dtype=torch.float), gp_params, K_invs, noise, X_s, Y_s, states[i], weights[i], solution, dt_outer, dt_inner, polemass_length, gravity, length, masspole, total_mass, tau)#, gps )        
        
        # Compress back now
        next_states, next_weights = traced_sigma_point_compress_JIT( next_states_expanded, next_weights_expanded )
        
        # Store states and weights
        states.append( next_states ); weights.append( next_weights )
            
        # Get reward 
        reward = reward + traced_reward_UT_Mean_Evaluator_basic( states[i+1], weights[i+1] )
        
    return reward

def generate_psd_matrices():
    n = 4
    N = 50
    # np.random.seed(0)
    B = []
    for i in range(N):
        A = np.random.rand(n,n) 
        A = 0.5 * ( A + A.T )
        A = A + n * np.eye(n)
        if i == 0:
            B = np.copy(A).reshape( (n,n,1) )
        else:
            B = np.append( B, A.reshape( (n,n,1) ), axis = 2 )   
    return B

# Set up environment
env_to_render = CustomCartPoleEnv(render_mode="human")
env = env_to_render #RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/", name_prefix="Excartpole" )
observation, info = env.reset(seed=42)

polemass_length, gravity, length, masspole, total_mass, tau = torch.tensor(env.polemass_length), torch.tensor(env.gravity), torch.tensor(env.length), torch.tensor(env.masspole), torch.tensor(env.total_mass), torch.tensor(env.tau)

# Initialize parameters
N = 50
H = 10
param_w = np.random.rand(N) - 0.5 
param_mu = np.random.rand(4,N) - 0.5 * np.ones((4,N))
# param_Sigma = torch.rand(10)
param_Sigma = generate_psd_matrices()
lr_rate = 0.5
noise = torch.tensor(0.1, dtype=torch.float)
first_run = True
# X = torch.rand(4).reshape(-1,1)

# Initialize sim parameters
t = 0
dt_inner = 0.02
dt_outer = 0.02
outer_loop = 2
GA = 1.0
PE = 0.0
# gps = initialize_gps(noise.item())


train_X = np.array([0,0,0,0, 0]).reshape(-1,1)
train_Y = np.array([0,0,0,0]).reshape(-1,1)

initialize_tensors( env, param_w, param_mu, param_Sigma )

plt.ion()

for i in range(300):
     
    if (i % outer_loop != 0) or i<1:
    
        # Find input
        state = env.get_state()
        state_torch = torch.tensor( state, dtype=torch.float )
        if i<1:
            action = 20*torch.tensor(env.action_space.sample()-0.5)#   2*torch.rand(1)
        else:
            action = policy( env.w_torch, env.mu_torch, env.Sigma_torch, state_torch )
            if (abs(action.item()))>20:
                print("ERROR*************************")
                # exit()
        print("action", action)
        observation, reward, terminated, truncated, info = env.step(action.item())
        env.render()
        
        # Get training data fro GP
        # state_next = env.get_state()
        # state_dot = (state_next - state) / dt_inner
        # action_np = np.array([[action.item()]])
        # sys_state = np.append( state, action_np, axis =0 )
        # if i==0:
        #     train_X = np.copy( sys_state )
        #     train_Y = np.copy( state_dot )
        # else:
        #     train_X = np.append( train_X, sys_state, axis = 1 )
        #     train_Y = np.append( train_Y, state_dot, axis = 1 )
        
        t = t + dt_inner
        
        if terminated or truncated:
            observation, info = env.reset()
        
    else:
        
        initialize_tensors( env, param_w, param_mu, param_Sigma )
        
        psds = generate_psd_matrices()
        
        covs = torch.tensor(1)
        params = torch.tensor(1)
        gps = torch.tensor(1)
        X_s = torch.tensor(1)
        Y_s = torch.tensor(1)
        
        reward = get_future_reward( env, params, covs, torch.tensor(X_s, dtype=torch.float), torch.tensor(Y_s, dtype=torch.float), noise, gps )        
        reward.backward(retain_graph = True)
        
        w_grad = getGrad( env.w_torch )
        mu_grad = getGrad( env.mu_torch )
        Sigma_grad = getGrad( env.Sigma_torch )
        
        param_w = np.clip( param_w - lr_rate * w_grad, -1000.0, 1000.0 )
        param_mu = np.clip( param_mu - lr_rate * mu_grad, -1000.0, 1000.0 )        
        
        # print(f"w:{param_w}")#, mu:{param_mu}")
        # param_Sigma = np.clip( param_Sigma - lr_rate * Sigma_grad, -30.0, 30.0 )
        # Sigma = np.diag( param_Sigma[0:4] )
        # Sigma[0,1] = param_Sigma[4]; Sigma[1,0] = param_Sigma[4]; Sigma[0,2] = param_Sigma[5]; Sigma[2,0] = param_Sigma[5]
        # Sigma[0,3] = param_Sigma[6]; Sigma[3,0] = param_Sigma[6]; Sigma[1,2] = param_Sigma[7]; Sigma[2,1] = param_Sigma[7]
        # Sigma[1,3] = param_Sigma[8]; Sigma[3,1] = param_Sigma[8]; Sigma[2,3] = param_Sigma[9]; Sigma[3,2] = param_Sigma[9]
        # if np.abs(np.linalg.det(Sigma))<0.01:
        #     print("*** ERROR******: wrong covariance matrix")
    
        
# env.close_video_recorder()
env.close()