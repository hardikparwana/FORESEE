import jax.numpy as np
from jax import random, jit, grad
from jax import lax
# import numpy
import time
from random_utils import generate_psd_params
# Nonlinear RBF network

def policy(params_policy, X):
    # n = 4 # dim of state
    # N = 50 # number of basis functions  #50:593 , 30: 355. matmul instead of @ 444... gets slower
    state = np.array([
        X[0,0],
        X[1,0],
        np.cos(X[2,0]),
        np.sin(X[2,0]),
        X[3,0]
    ]).reshape(-1,1)
    N = 50
    n = 5
    param_w = params_policy[0:N]
    param_mu = params_policy[N:n*N+N].reshape(n,N)
    param_Sigma = params_policy[N*n+N:].reshape(N,15)

    pi = np.zeros((1,1))

    # for t in range(N):
    def body(t, pi):
        diff = state - param_mu[:,t].reshape(-1,1)    
        Sigma = np.array([  
            [ param_Sigma[t,0], 0.0, 0.0, 0.0, 0.0 ],
            [ param_Sigma[t,5], param_Sigma[t,1], 0.0, 0.0, 0.0 ],
            [ param_Sigma[t,6], param_Sigma[t,7], param_Sigma[t,2], 0.0, 0.0 ],
            [ param_Sigma[t,8], param_Sigma[t,9], param_Sigma[t,10], param_Sigma[t,3], 0.0 ],
            [ param_Sigma[t,11], param_Sigma[t,12], param_Sigma[t,13], param_Sigma[t,14], param_Sigma[t,4] ]
        ])
        Sigma = n * np.eye(n) + Sigma.T @ Sigma
        Sigma_inv = np.linalg.inv( Sigma )
        phi = np.exp( -0.5 * diff.T @ Sigma_inv @ diff )
        phi = np.exp( -0.5 * np.matmul(np.matmul(diff.T, Sigma_inv),diff ) ) 
        pi = pi + param_w[t] * phi
        return pi
    # return pi
    return np.clip(lax.fori_loop( 0, N, body, pi ), -5, 5)[0,0]

policy_jit = jit(policy)
policy_grad = grad(policy, 0)
policy_grad_jit = jit(grad(policy))

if 0:
    print("Testing Cart Pole Policy")

    test_key = random.PRNGKey(0)
    test_N = 50

    # Testing Parameters
    key = random.PRNGKey(100)
    n = 5
    N = 50
    key, subkey = random.split(key)
    param_w = random.uniform(subkey, shape=(N,1))[:,0] - 0.5#+ 0.5#+ 2.0  #0.5 work with Lr: 5.0

    key, subkey = random.split(key)
    param_mu = random.uniform(subkey, shape=(n,N))- 0.5 * np.ones((n,N)) #- 3.5 * np.ones((4,N))
    param_Sigma = generate_psd_params(n,N) # 10,N
    params_policy = np.append( np.append( param_w, param_mu.reshape(-1,1)[:,0] ), param_Sigma.reshape(-1,1)[:,0]  )

    # Testing input state
    test_X = random.uniform(key, shape=(4,1))

    # run once to force JIT compilation
    t0 = time.time()
    test_u = policy( params_policy, test_X )
    print(f"time scan:{time.time()-t0}")

    # print("u grad", test_u)
