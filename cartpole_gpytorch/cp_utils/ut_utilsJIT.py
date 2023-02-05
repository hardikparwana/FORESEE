import numpy as np
import torch
from utils.sqrtm import sqrtm
from robot_models.custom_cartpole import get_state_dot_noisy_torch

def get_mean_JIT(sigma_points, weights):
    weighted_points = sigma_points * weights[0]
    mu = torch.sum( weighted_points, 1 ).reshape(-1,1)
    return mu
traced_get_mean_JIT = get_mean_JIT #torch.jit.trace( get_mean_JIT, (torch.ones(4,9), torch.ones(1,9)/9.0 ) )

def get_mean_cov_JIT81(sigma_points, weights):
    
    # mean
    weighted_points = sigma_points * weights[0]
    mu = torch.sum( weighted_points, 1 ).reshape(-1,1)
    
    # covariance
    centered_points = sigma_points - mu
    weighted_centered_points = centered_points * weights[0] 
    cov = torch.matmul( weighted_centered_points, centered_points.T )
    
    # print(f"Checking for Nans: {torch.isnan(cov).any()}")
    return mu, cov
# traced_get_mean_cov_JIT81 = torch.jit.trace( get_mean_cov_JIT81, (torch.ones( (4 , 81) ), 0.5 * torch.ones( (1,81) ) ) )
traced_get_mean_cov_JIT81 = get_mean_cov_JIT81
                                            
# @torch.jit.script
def get_ut_cov_root(cov):
    k = -1
    n = cov.shape[0]
        
    if torch.linalg.det( cov )< 0.01:
        root_term = cov
    else:
        root_term = sqrtm((n+k)*cov)
    return root_term

# @torch.jit.script
def get_ut_cov_root_diagonal(cov):
    k = -1
    n = cov.shape[0]
    
    cov_abs = torch.abs(cov)
    if cov_abs[0,0]>0.01:
        root0 = torch.sqrt(cov[0,0])
    else:
        root0 = torch.tensor(0, dtype=torch.float)
    if cov_abs[1,1]>0.01:
        root1 = torch.sqrt(cov[1,1])
    else:
        root1 = torch.tensor(0, dtype=torch.float)
    if cov_abs[2,2]>0.01:
        root2 = torch.sqrt(cov[2,2])
    else:
        root2 = torch.tensor(0, dtype=torch.float)
    if cov_abs[3,3]>0.01:
        root3 = torch.sqrt(cov[2,2])
    else:
        root3 = torch.tensor(0, dtype=torch.float)
    
    root_term = torch.diag( (n+k) * torch.cat( ( root0.reshape(-1,1), root1.reshape(-1,1), root2.reshape(-1,1), root3.reshape(-1,1) ), dim = 1 )[0] )

    return root_term

# @torch.jit.script
def initialize_sigma_points_JIT(X):
    # return 2N + 1 points
    n = X.shape[0]
    num_points = 2*n + 1
    sigma_points = torch.clone( X )
    for _ in range(n):
        sigma_points = torch.cat( (sigma_points, torch.clone( X )) , dim = 1)
        sigma_points = torch.cat( (sigma_points, torch.clone( X )) , dim = 1)
    weights = torch.ones((1,num_points)) * 1.0/( num_points )
    return sigma_points, weights

def generate_sigma_points9_JIT( mu, cov_root, base_term, factor ):
    
    n = mu.shape[0]     
    N = 2*n + 1 # total points

    # TODO
    # k = n - 3
    k = 0.5 #2

    new_points = base_term + factor * torch.clone(mu)
    new_weights = torch.tensor([1.0*k/(n+k)]).reshape(-1,1)#torch.tensor([k/(n+k)]).reshape(-1,1)
    for i in range(n):
        new_points = torch.cat( (new_points, base_term + factor * (mu - cov_root[:,i].reshape(-1,1)) ), dim = 1 )
        new_points = torch.cat( (new_points, base_term + factor * (mu + cov_root[:,i].reshape(-1,1)) ), dim = 1 )

        new_weights = torch.cat( (new_weights, torch.tensor(1.0/2/(n+k)).reshape(1,-1)), dim = 1 )
        new_weights = torch.cat( (new_weights, torch.tensor(1.0/2/(n+k)).reshape(-1,1)), dim = 1 )

    return new_points, new_weights

mu_t = torch.ones((4,1)).reshape(-1,1)
cov_t = torch.tensor([ [ 1.0, 0.0, 0.0, 0.0 ], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 1.5] ])
# traced_generate_sigma_points9_JIT = torch.jit.trace( generate_sigma_points9_JIT, ( mu_t, cov_t, torch.ones((4,1)), torch.tensor(1.0) ) )
traced_generate_sigma_points9_JIT = generate_sigma_points9_JIT
# def sigma_point_expand_JIT(GA, PE, gp_params, K_invs, noise, X_s, Y_s, sigma_points, weights, control, dt_outer, dt_inner, polemass_length, gravity, length, masspole, total_mass, tau):#, gps):
def sigma_point_expand_JIT(sigma_points, weights, control, dt_outer, gp):#, gps):
   
    n, N = sigma_points.shape       
    
    input_x = torch.cat( (sigma_points[:,0].reshape(-1,1),control.reshape(-1,1)), axis=0 ).reshape(1,-1)
    # input_x.sum().backward(retain_graph=True)
    # min_v = torch.min(torch.norm(gp.train_inputs[0] - input_x, dim=1))
    # print(f"Input OK, min:{min_v}")
    prediction = gp( input_x )
    mu, cov = prediction.mean.reshape(-1,1), prediction.covariance_matrix      
    # mu.sum().backward(retain_graph=True)
    # print("done")
    root_term = get_ut_cov_root_diagonal(cov) 
    # print(f"input: {input_x}, Cov: {cov}")
    temp_points, temp_weights = traced_generate_sigma_points9_JIT( mu, root_term, sigma_points[:,0].reshape(-1,1), dt_outer )
    new_points = torch.clone( temp_points )
    new_weights = (torch.clone( temp_weights ) * weights[0,0]).reshape(1,-1)
        
    for i in range(1,N):
        
        input_x = torch.cat( (sigma_points[:,i].reshape(-1,1),control.reshape(-1,1)), axis=0 ).reshape(1,-1)
        # input_x.sum().backward(retain_graph=True)
        # print("Input OK")
        prediction = gp( input_x )
        mu, cov = prediction.mean.reshape(-1,1), prediction.covariance_matrix  
        # if 1:#torch.norm(torch.diagonal(cov))>40:
        #   print(f"cov: {cov[0,0]}. {cov[1,1]}, {cov[2,2]}, {cov[3,3]}") 
        # min_v = torch.min(torch.norm(gp.train_inputs[0] - input_x, dim=1))
        # print(f"Input OK, min:{min_v}")
        # mu.sum().backward(retain_graph=True)
        # print("done2")
        root_term = get_ut_cov_root_diagonal(cov)       
        temp_points, temp_weights = traced_generate_sigma_points9_JIT( mu, root_term, sigma_points[:,i].reshape(-1,1), dt_outer )
        new_points = torch.cat((new_points, temp_points), dim=1 )
        new_weights = torch.cat( (new_weights, (temp_weights * weights[0,i]).reshape(1,-1) ) , dim=1 )

    return new_points, new_weights

def sigma_point_compress_JIT( sigma_points, weights ):
    mu, cov = traced_get_mean_cov_JIT81( sigma_points, weights )
    cov_root_term = get_ut_cov_root( cov )  
    base_term = torch.zeros((mu.shape))
    return traced_generate_sigma_points9_JIT( mu, cov_root_term, base_term, torch.tensor(1.0) )
traced_sigma_point_compress_JIT = sigma_point_compress_JIT

def reward_UT_Mean_Evaluator_basic(sigma_points, weights):
    mu = compute_reward_jit( sigma_points[:,0].reshape(-1,1)  ) *  weights[0,0]
    for i in range(1, sigma_points.shape[1]):
        mu = mu + compute_reward_jit( sigma_points[:,i].reshape(-1,1)  ) *  weights[0,i]
    return mu

def compute_reward_jit( state ):
    theta = state[2,0] # want theta and theta_dot to be 0
    speed = state[1,0]
    pos = state[0,0]
    # return - 100 * torch.cos(theta) #+ 0.1 * torch.square(speed)
    return - 100 * torch.cos(theta) + 0.1 * torch.square(speed)
    # return - 100 * torch.cos(theta) + 0.1 * torch.square(pos) + 0.1 * torch.square(speed)
    # return - 100 * torch.cos(theta) + 0.1 * torch.square(pos)
    
# traced_reward_UT_Mean_Evaluator_basic = torch.jit.trace( reward_UT_Mean_Evaluator_basic, ( torch.ones(4,9), torch.ones(1,9) ) )
traced_reward_UT_Mean_Evaluator_basic = reward_UT_Mean_Evaluator_basic