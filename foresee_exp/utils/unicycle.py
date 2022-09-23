import numpy as np
import torch

def unicycle_f_torch_jit(x):
    return torch.tensor([0.0,0.0,0.0],dtype=torch.float).reshape(-1,1)

def unicycle_g_torch_jit(x):
    # return torch.tensor([ [torch.cos(x[2,0]),0.0],[torch.sin(x[2,0]),0.0],[0,1] ])
    g1 = torch.cat( (torch.cos(x[2,0]).reshape(-1,1),torch.tensor([[0]]) ), dim=1 )
    # g2 = torch.cat( ( torch.tensor([[0]]), torch.sin(x[2,0]).reshape(-1,1) ), dim=1 )
    g2 = torch.cat( (torch.sin(x[2,0]).reshape(-1,1), torch.tensor([[0]]) ), dim=1 )
    g3 = torch.tensor([[0,1]],dtype=torch.float)
    gx = torch.cat((g1,g2,g3))
    return gx

def unicycle_step_torch( state, control, dt ):
    f = unicycle_f_torch_jit( state )
    g = unicycle_g_torch_jit( state )
    next_state = state + ( f + g @ control ) * dt
    next_state[2] = torch.atan2( torch.sin( next_state[2] ), torch.cos( next_state[2] ) )
    return next_state
traced_unicycle_step_torch = torch.jit.trace( unicycle_step_torch, ( torch.tensor([0,0,0], dtype=torch.float).reshape(-1,1), torch.tensor([0,0], dtype=torch.float).reshape(-1,1), torch.tensor(0.1, dtype=torch.float) ) )

def unicycle_SI2D_clf_condition_evaluator( robotJ_state, robotK_state, robotK_state_dot, k_torch ):
    V, dV_dxj, dV_dxk = unicycle_SI2D_lyapunov_tensor_jit( robotJ_state, robotK_state )
    
    B = - dV_dxj @ unicycle_f_torch_jit( robotJ_state ) - dV_dxk @ robotK_state_dot - k_torch * V
    A = - dV_dxj @ unicycle_g_torch_jit( robotJ_state )
    
    return A, B 


def unicycle_SI2D_cbf_fov_condition_evaluator( robotJ_state, robotK_state, robotK_state_dot, alpha_torch):
    h1, dh1_dxj, dh1_dxk, h2, dh2_dxj, dh2_dxk, h3, dh3_dxj, dh3_dxk = unicycle_SI2D_fov_barrier_jit(robotJ_state, robotK_state)   
    
    B1 = dh1_dxj @ unicycle_f_torch_jit( robotJ_state ) + dh1_dxk @ robotK_state_dot + alpha_torch[0] * h1
    A1 = dh1_dxj @ unicycle_g_torch_jit( robotJ_state ) 
    
    B2 = dh2_dxj @ unicycle_f_torch_jit( robotJ_state ) + dh2_dxk @ robotK_state_dot + alpha_torch[1] * h2
    A2 = dh2_dxj @ unicycle_g_torch_jit( robotJ_state ) 
    
    B3 = dh3_dxj @ unicycle_f_torch_jit( robotJ_state ) + dh3_dxk @ robotK_state_dot + alpha_torch[2] * h3
    A3 = dh3_dxj @ unicycle_g_torch_jit( robotJ_state ) 
    
    B = torch.cat( (B1, B2, B3), dim = 0 )
    A = torch.cat( (A1, A2, A3), dim = 0 )
    
    # print(f"h1:{h1}, , h2:{h2}, h3:{h3}, dh_dx1:{dh1_dxj}, dh_dxk:{ dh1_dxk }, g:{unicycle_g_torch_jit( robotJ_state ) } ")
    
    return A, B

def unicycle_SI2D_fov_barrier_jit(X, targetX):
    
    # print(f"X:{X}, targetX:{targetX}")
    
    max_D = 2.0
    min_D = 0.3
    FoV_angle = torch.tensor(np.pi/3, dtype=torch.float)
    
    # Max distance
    h1 = max_D**2 - torch.square( torch.norm( X[0:2] - targetX[0:2] ) )
    dh1_dxi = torch.cat( ( -2*( X[0:2] - targetX[0:2] ), torch.tensor([[0.0]]) ), 0).T
    dh1_dxj =  2*( X[0:2] - targetX[0:2] ).T
    
    # Min distance
    h2 = torch.square(torch.norm( X[0:2] - targetX[0:2] )) - min_D**2
    dh2_dxi = torch.cat( ( 2*( X[0:2] - targetX[0:2] ), torch.tensor([[0.0]]) ), 0).T
    dh2_dxj = - 2*( X[0:2] - targetX[0:2]).T

    # Max angle
    p = targetX[0:2] - X[0:2]

    # dir_vector = torch.tensor([[torch.cos(x[2,0])],[torch.sin(x[2,0])]]) # column vector
    dir_vector = torch.cat( ( torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1) ) )
    
    bearing_angle  = torch.matmul(dir_vector.T , p )/ torch.norm(p)
    h3 = (bearing_angle - torch.cos(FoV_angle/2))/(1.0-torch.cos(FoV_angle/2))
    # print(f"dir_vector: {dir_vector.T}, bearing:angle:{ bearing_angle }, h3:{h3}")

    norm_p = torch.norm(p)
    dh3_dx = dir_vector.T / norm_p - ( dir_vector.T @ p)  * p.T / torch.pow(norm_p,3)    
    dh3_dTheta = ( -torch.sin(X[2]) * p[0] + torch.cos(X[2]) * p[1] ).reshape(1,-1)  /torch.norm(p)
    dh3_dxi = torch.cat(  ( -dh3_dx , dh3_dTheta), 1  ) /(1.0-torch.cos(FoV_angle/2))
    dh3_dxj = dh3_dx /(1.0-torch.cos(FoV_angle/2))
    
    # print(f"dist_sq:{torch.square(torch.norm( X[0:2] - targetX[0:2] ))}, h1:{h1}, h2:{h2}, h3:{h3}")
    # print(f" dh3_dxi:{dh3_dxi}, dh3:dxj:{dh3_dxj} ")
    
    return h1, dh1_dxi, dh1_dxj, h2, dh2_dxi, dh2_dxj, h3, dh3_dxi, dh3_dxj

def unicycle_SI2D_clf_cbf_fov_evaluator( robotJ_state, robotK_state, robotK_state_dot, k_torch, alpha_torch ):
    
    A1, B1 = unicycle_SI2D_clf_condition_evaluator( robotJ_state, robotK_state, robotK_state_dot, k_torch )
    A2, B2 = unicycle_SI2D_cbf_fov_condition_evaluator( robotJ_state, robotK_state, robotK_state_dot, alpha_torch )   
    
    B = torch.cat( (B1, B2), dim=0 )
    A = torch.cat( (A1, A2), dim=0 )
    
    return A, B
    
def wrap_angle_tensor_JIT(angle):
    # factor = torch.tensor(2*3.14157,dtype=torch.float)
    # if angle>torch.tensor(3.14157):
    #     angle = angle - factor
    # if angle<torch.tensor(-3.14157):
    #     angle = angle + factor
    # return angle
    return torch.atan2( torch.sin( angle ), torch.cos( angle ) )

def unicycle_nominal_input_tensor_jit(X, targetX):
    k_omega = 2.0 #0.5#2.5
    k_v = 2.0 #0.5
    diff = targetX[0:2,0] - X[0:2,0]

    theta_d = torch.atan2(targetX[1,0]-X[1,0],targetX[0,0]-X[0,0])
    error_theta = wrap_angle_tensor_JIT( theta_d - X[2,0] )

    omega = k_omega*error_theta 
    v = k_v*( torch.norm(diff) ) * torch.cos( error_theta )
    v = v.reshape(-1,1)
    omega = omega.reshape(-1,1)
    U = torch.cat((v,omega))
    return U.reshape(-1,1)
traced_unicycle_nominal_input_tensor_jit = torch.jit.trace( unicycle_nominal_input_tensor_jit, ( torch.tensor([0,0,0],dtype=torch.float).reshape(-1,1), torch.tensor([0,0],dtype=torch.float).reshape(-1,1) ) )

def unicycle_SI2D_lyapunov_tensor_jit(X, G):
    min_D = 0.3
    max_D = 2.0
    avg_D = (min_D + max_D)/2.0
    V = torch.square ( torch.norm( X[0:2] - G[0:2] ) - avg_D )
    
    factor = 2*(torch.norm( X[0:2]- G[0:2] ) - avg_D)/torch.norm( X[0:2] - G[0:2] ) * (  X[0:2] - G[0:2] ).reshape(1,-1) 
    dV_dxi = torch.cat( (factor, torch.tensor([[0]])), dim  = 1 )
    dV_dxj = -factor
    # print(f" dist:{ torch.norm( X[0:2]- G[0:2] ) }"  )
    return V, dV_dxi, dV_dxj

def unicycle_compute_reward_jit(X,targetX):
    
    max_D = torch.tensor(2.0)
    min_D = torch.tensor(0.3)
    FoV_angle = torch.tensor(3.13/3)    

    p = targetX[0:2] - X[0:2]
    dir_vector = torch.cat( ( torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1) ) )
    bearing_angle  = torch.matmul(dir_vector.T , p )/ torch.norm(p)
    h3 = (bearing_angle - torch.cos(FoV_angle/2))/(1.0-torch.cos(FoV_angle/2))
    
    return torch.square( torch.norm( X[0:2,0] - targetX[0:2,0]  ) - torch.tensor((min_D+max_D)/2) ) - 2 * h3