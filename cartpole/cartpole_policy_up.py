import jax.numpy as np
from jax import random, jit
# import numpy
import time
# Nonlinear RBF network
def policy(params_policy, Sigma_invs, X):
# def policy(param_w, param_mu, param_Sigma, X):
    n = 4 # dim of state
    m = 1 # dim of input
    N = 30 # number of basis functions  #50:593 , 30: 355. matmul instead of @ 444... gets slower
    
    param_w = params_policy[0:N]
    param_mu = params_policy[N:4*N+N].reshape(4,N)
    # param_Sigma = params_policy[4*N+N:4*N+N+10*N].reshape(10,N)
  
    # First basis function
    diff = X - param_mu[:,0].reshape(-1,1)
  
    # Given lower triangular
    i = 0
    lower = n*n*i
    upper = n*n*(i+1)
    Sigma_inv = Sigma_invs[lower:upper].reshape(n,n)
    # phi = np.exp( -0.5 * diff.T @ Sigma_inv @ diff )        
    phi = np.exp( -0.5 * np.matmul(np.matmul(diff.T, Sigma_inv),diff ) ) 
    pi = param_w[0] * phi
    
    # Remaining basis functions
    for i in range(1,N):
        diff = X - param_mu[:,i].reshape(-1,1)
    
        lower = n*n*i
        upper = n*n*(i+1)
        Sigma_inv = Sigma_invs[lower:upper].reshape(n,n)
        # phi = np.exp( -0.5 * diff.T @ Sigma_inv @ diff )
        phi = np.exp( -0.5 * np.matmul(np.matmul(diff.T, Sigma_inv),diff ) ) 
        pi = pi + param_w[i] * phi        
    # if np.abs( pi ) > 10:
    #     pi = pi / np.abs(pi) * 10
    return pi

policy_jit = jit(policy)

# print("Testing Cart Pole Policy")

# test_key = random.PRNGKey(0)
# test_N = 50
# test_param_w = random.uniform(test_key, shape=(test_N,1))[:,0] #numpy.random.rand(N)
# test_param_mu = random.uniform(test_key, shape=(4,test_N))#numpy.random.rand(4,N)
# test_param_Sigma = random.uniform(test_key, shape=(test_N,10)) #numpy.random.rand(N,10)
# test_X = random.uniform(test_key, shape=(4,1)) #numpy.random.rand(4).reshape(-1,1)

# # policy_jit = jit(policy)

# # run once to force JIT compilation
# t0 = time.time()
# test_u = policy_jit( test_param_w, test_param_mu, test_param_Sigma, test_X )
# print(f"time:{time.time()-t0}")

# t0 = time.time()
# test_u = policy_jit( test_param_w, test_param_mu, test_param_Sigma, test_X )
# print(f"time:{time.time()-t0}")
# print("u", test_u)