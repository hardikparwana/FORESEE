import jax.numpy as np
from jax import random, jit, grad
from jax import lax
# import numpy
import time
# Nonlinear RBF network

def squash_inputs( u, u_max = 10 ):
    return u_max * np.tanh( u / u_max )

def random_exploration( key, input_dim=1, u_max = 10 ):
    key, subkey = random.split(key)
    rand_u = u_max*(2*random.uniform( subkey, shape=( input_dim, 1 )) -1 )
    return rand_u[0,0]

def Sum_of_gaussians_initialize(key, state_dim, input_dim, type = 'gaussian', u_max = 10, num_basis = 50, lengthscale = 1, centers_init_min = -1, centers_init_max = 1):
        
    if type=='gaussian':
        # without extra angle        
        lengthscales_init = lengthscale* np.ones(state_dim)
        # lengthscales_init = 0.1 + random.uniform(subkey, shape=(state_dim,))
        log_lengthscales = np.log( lengthscales_init )
    
        key, subkey = random.split(key)
        angle_centers = 2*np.pi*(random.uniform(subkey, shape=(num_basis,1))-0.5)
        key, subkey = random.split(key)
        not_angle_centers = 2*np.pi*(random.uniform(subkey, shape=(num_basis,3))-0.5)    
        centers_init = np.append( not_angle_centers, angle_centers, axis=1 )
    elif type=='with angles':
        #with extra angles        
        lengthscales_init = lengthscale* np.ones(state_dim+1)
        log_lengthscales = np.log( lengthscales_init )
    
        key, subkey = random.split(key)
        angle_centers = 2*np.pi*(random.uniform(subkey, shape=(num_basis,1))-0.5)
        cos_center = np.cos(angle_centers)
        sin_center = np.sin(angle_centers)
        key, subkey = random.split(key)
        not_angle_centers = 2*np.pi*(random.uniform(subkey, shape=(num_basis,3))-0.5)    
        centers_init = np.append( not_angle_centers, np.append(sin_center, cos_center, axis=1), axis=1 )
        
    elif type=='random':
        # all random
        key, subkey = random.split(key)
        centers_init = centers_init_min * ( np.ones(state_dim, num_basis) ) + (centers_init_max-centers_init_min)*random.uniform( subkey, shape=(state_dim, num_basis) )
    else:
        print(f"Incorrect type passed")
        exit(  )
    
    # weights_init = np.ones((input_dim, num_basis))
    key, subkey = random.split(key)
    weights_init = u_max* 2 * (random.uniform(subkey, shape=(input_dim, num_basis))-0.5)
    
    return key, np.append( log_lengthscales.reshape(-1,1), np.append( centers_init.reshape(-1,1), weights_init.reshape(-1,1) , axis=0), axis=0 )
    
def Sum_of_gaussians9( state, policy_params, u_max = 10, state_dim = 4, input_dim = 1, num_basis = 2 ):
    log_lengthscales = policy_params[0:state_dim]
    centers = policy_params[state_dim:state_dim+num_basis*state_dim].reshape(state_dim, num_basis)
    weights = policy_params[-input_dim*num_basis:]
    
    lengthscales = np.exp( log_lengthscales )    
    scale_factor = np.ones(state_dim)
    state = (state.T / scale_factor).T
    exponent = np.sum( np.square((state[:,0].reshape(-1,1) - centers)/lengthscales)  , axis = 0 )
    control_input = squash_inputs( np.sum( weights[:,0] * np.exp( -exponent ) ).reshape(-1,1), u_max)
    for i in range(1,9):
        exponent = np.sum( np.square((state[:,i].reshape(-1,1) - centers)/lengthscales)  , axis = 0 )
        control_input = np.append( control_input, squash_inputs( np.sum( weights[:,0] * np.exp( -exponent ) ).reshape(-1,1), u_max), axis=1)
    return control_input

def Sum_of_gaussians_with_angle9( state, policy_params, u_max = 10, state_dim = 4, input_dim = 1, num_basis = 2 ):
    new_state = np.array([ state[0,0], state[1,0], state[3,0], np.sin(state[2,0]), np.cos(state[2,0]) ]).reshape(-1,1)
    for i in range(1,9):
        new_state = np.append( new_state, np.array([ state[0,i], state[1,i], state[3,i], np.sin(state[2,i]), np.cos(state[2,i]) ]).reshape(-1,1), axis=1)
    return Sum_of_gaussians9( new_state, policy_params, u_max = u_max, state_dim = state_dim+1, input_dim = input_dim, num_basis = num_basis )
    
# @jit
def policy9(state, params_policy):
    # return Sum_of_gaussians9( state, params_policy, u_max = 10, state_dim = 4, input_dim = 1, num_basis = 50 )
    return Sum_of_gaussians_with_angle9( state, params_policy, u_max = 10, state_dim = 4, input_dim = 1, num_basis = 50 )
    
def Sum_of_gaussians( state, policy_params, u_max = 10, state_dim = 4, input_dim = 1, num_basis = 2 ):
    log_lengthscales = policy_params[0:state_dim]
    centers = policy_params[state_dim:state_dim+num_basis*state_dim].reshape(state_dim, num_basis)
    weights = policy_params[-input_dim*num_basis:]
    
    lengthscales = np.exp( log_lengthscales )    
    scale_factor = np.ones(state_dim)
    state = (state.T / scale_factor).T
    exponent = np.sum( np.square((state - centers)/lengthscales)  , axis = 0 )
    control_input = squash_inputs( np.sum( weights[:,0] * np.exp( -exponent ) ).reshape(-1,1), u_max)
    return control_input

def Sum_of_gaussians_with_angle( state, policy_params, u_max = 10, state_dim = 4, input_dim = 1, num_basis = 2 ):
    new_state = np.array([ state[0,0], state[1,0], state[3,0], np.sin(state[2,0]), np.cos(state[2,0]) ]).reshape(-1,1)
    return Sum_of_gaussians( new_state, policy_params, u_max = u_max, state_dim = state_dim+1, input_dim = input_dim, num_basis = num_basis )
    
# @jit
def policy(state, params_policy):
    # return Sum_of_gaussians( state, params_policy, u_max = 10, state_dim = 4, input_dim = 1, num_basis = 50 )
    return Sum_of_gaussians_with_angle( state, params_policy, u_max = 10, state_dim = 4, input_dim = 1, num_basis = 50 )
 
# @jit 
def random_policy( key, u_max = 10 ):
    return random_exploration( key, input_dim=1, u_max = u_max )

# @jit
def policy_grad( state, params_policy):
    return grad( policy, 1 )(state, params_policy)

# policy_grad = jit(grad(policy, 1))
# policy_grad_jit = jit(grad(policy))

if 0:
    print("Testing Cart Pole Policy")
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    key, params_policy = Sum_of_gaussians_initialize(subkey, state_dim=4, input_dim=1, type = 'gaussian', num_basis = 50, lengthscale = 1, centers_init_min = -1, centers_init_max = 1)

    # key, params_policy = Sum_of_gaussians_initialize(subkey, state_dim=4, input_dim=1, type = 'with angles', num_basis = 50, lengthscale = 1, centers_init_min = -1, centers_init_max = 1)
   
    init_state = np.array([0.0,0,0,0]).reshape(-1,1)   
    # init_state2 = np.array([1.0,0,0,0]).reshape(-1,1)   
    # init_state = np.append( init_state, init_state2, axis=1 )
    
    # policy9( init_state, params_policy )
    # policy_grad( init_state, params_policy )   
    
    t0 = time.time()
    action = policy9( init_state, params_policy)
    print(f"policy time jit:{time.time()-t0}, action:{action}")
    
    random_policy( key )
    print(f"random: {random_policy(key)}")
    
    # t0 = time.time()
    # grads = policy_grad( init_state, params_policy)
    # print(f"grad time jit:{time.time()-t0}, grad = {grads}")
    

    
