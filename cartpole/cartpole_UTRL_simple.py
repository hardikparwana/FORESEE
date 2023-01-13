import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})

import torch
torch.autograd.set_detect_anomaly(True)

from utils.utils import *
from ut_utils.ut_utilsJIT import *

from robot_models.custom_cartpole import CustomCartPoleEnv
from gym_wrappers.record_video import RecordVideo
from cartpole_policy import policy, traced_policy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning) 

def initialize_tensors(robot, param_w, param_mu, param_Sigma):
    # x, x_dot, theta, theta_dot = robot.state
    # robot.X_torch = torch.tensor( np.array([ x, x_dot, theta, theta_dot ]).reshape(-1,1), requires_grad = True, dtype=torch.float )
    robot.X_torch = torch.tensor( robot.get_state(), requires_grad = True, dtype=torch.float )
    robot.w_torch = torch.tensor( param_w, requires_grad = True, dtype=torch.float )
    robot.mu_torch = torch.tensor( param_mu, requires_grad = True, dtype=torch.float )
    robot.Sigma_torch = torch.tensor( param_Sigma, requires_grad = True, dtype=torch.float )

def get_future_reward(robot, gp_params, K_invs, X_s, Y_s, noise, gps):
           
    prior_states, prior_weights = initialize_sigma_points_JIT(robot.X_torch)
    states = [prior_states]
    weights = [prior_weights]
    reward = torch.tensor([0],dtype=torch.float)
        
    for i in range(H):  
        # print("hello")
        # Get mean position
        mean_position = traced_get_mean_JIT( states[i], weights[i] )
        
        # Get control input      
        solution = traced_policy( robot.w_torch, robot.mu_torch, robot.Sigma_torch, mean_position )
   
        # Get expanded next state
        next_states_expanded, next_weights_expanded = sigma_point_expand_JIT( states[i], weights[i], solution, dt_outer, dt_inner, polemass_length, gravity, length, masspole, total_mass, tau)#, gps )        
        
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
env_to_render = CustomCartPoleEnv(render_mode="rgb_array")
# env = env_to_render #RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/", name_prefix="Excartpole" )
env = RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/", name_prefix="cartpole_unconstrained_h20" )
observation, info = env.reset(seed=42)

polemass_length, gravity, length, masspole, total_mass, tau = torch.tensor(env.polemass_length), torch.tensor(env.gravity), torch.tensor(env.length), torch.tensor(env.masspole), torch.tensor(env.total_mass), torch.tensor(env.tau)

# Initialize parameters
N = 50
H = 20
np.random.seed(0)
param_w = np.random.rand(N) - 0.5 #+ 0.5#+ 2.0  #0.5 work with Lr: 5.0
param_mu = np.random.rand(4,N) - 0.5 * np.ones((4,N)) #- 3.5 * np.ones((4,N))
param_Sigma = generate_psd_params()

# param_Sigma = np.random.rand(4,N)

lr_rate = 0.4 #0.5 #0.5#0.001 #0.5
noise = torch.tensor(0.1, dtype=torch.float)
first_run = True
# X = torch.rand(4).reshape(-1,1)

# Initialize sim parameters
t = 0
dt_inner = 0.02
dt_outer = 0.06 # 0.02
outer_loop = 2#10 #2
GA = 1.0
PE = 0.0
# gps = initialize_gps(noise.item())


train_X = np.array([0,0,0,0, 0]).reshape(-1,1)
train_Y = np.array([0,0,0,0]).reshape(-1,1)

initialize_tensors( env, param_w, param_mu, param_Sigma )

plt.ion()

for i in range(400): #300
     
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
        print(f"action:{action}, state:{state[0]}")
        observation, reward, terminated, truncated, info = env.step(action.item())
        env.render()
        
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
        
        w_grad = getGrad( env.w_torch, l_bound = -2, u_bound = 2 )
        mu_grad = getGrad( env.mu_torch, l_bound = -2, u_bound = 2 )
        Sigma_grad = getGrad( env.Sigma_torch, l_bound = -2, u_bound = 2 )
        
        param_w = np.clip( param_w - lr_rate * w_grad, -1000.0, 1000.0 )
        param_mu = np.clip( param_mu - lr_rate * mu_grad, -1000.0, 1000.0 )       
        param_Sigma[:,0:4] = np.clip( param_Sigma[:,0:4] - lr_rate * Sigma_grad[:,0:4], 1, 3 ) 
        param_Sigma[:,4:] = np.clip( param_Sigma[:,4:] - lr_rate * Sigma_grad[:,4:], -1, 1 )
        
        
# env.close_video_recorder()
env.close()