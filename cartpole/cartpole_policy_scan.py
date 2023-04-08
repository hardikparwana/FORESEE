import jax.numpy as np
from jax import random, jit, grad
from jax import lax
# import numpy
import time
# Nonlinear RBF network

def policy(params_policy, Sigma_invs, X):
    # n = 4 # dim of state
    # N = 50 # number of basis functions  #50:593 , 30: 355. matmul instead of @ 444... gets slower
    N = 30
    n = 4
    param_w = params_policy[0:N]
    param_mu = params_policy[N:n*N+N].reshape(n,N)

    pi = np.zeros((1,1))

    def body(t, pi):
        diff = X - param_mu[:,t].reshape(-1,1)    
        Sigma_inv = Sigma_invs[t].reshape(n,n)
        phi = np.exp( -0.5 * diff.T @ Sigma_inv @ diff )
        # phi = np.exp( -0.5 * np.matmul(np.matmul(diff.T, Sigma_inv),diff ) ) 
        pi = pi + param_w[t] * phi
        return pi
    
    return np.clip(lax.fori_loop( 0, N, body, pi ), -10, 10)

policy_jit = jit(policy)
policy_grad_jit = jit(grad(policy))

if 0:
    print("Testing Cart Pole Policy")

    test_key = random.PRNGKey(0)
    test_N = 50

    def generate_psd_matrix_inverse():
        n = 4
        N = test_N
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        params_temp = random.uniform( subkey, shape=( int(n + (n**2 -n)/2.0),1) )
        Sigma = np.array([  
                [ params_temp[0,0], 0.0, 0.0, 0.0 ],
                [ params_temp[4,0], params_temp[1,0], 0.0, 0.0 ],
                [ params_temp[5,0], params_temp[6,0], params_temp[2,0], 0.0 ],
                [ params_temp[7,0], params_temp[8,0], params_temp[9,0], params_temp[3,0] ]
            ])
        Sigma = 4 * np.eye(4) + Sigma.T @ Sigma
        Sigma_inverse = np.linalg.inv( Sigma )
        Sigma_inverse = (Sigma_inverse + Sigma_inverse.T) /2.0
        Sigmas = np.copy( Sigma_inverse.reshape(1,-1) )

        for i in range(1,N):
            # Diagonal elements
            key, subkey = random.split(key)
            params_temp = random.uniform( subkey, shape=( 1,int(n + (n**2 -n)/2.0)) )
            Sigma = np.array([  
                [ params_temp[0,0], 0.0, 0.0, 0.0 ],
                [ params_temp[4,0], params_temp[1,0], 0.0, 0.0 ],
                [ params_temp[5,0], params_temp[6,0], params_temp[2,0], 0.0 ],
                [ params_temp[7,0], params_temp[8,0], params_temp[9,0], params_temp[3,0] ]
            ])
            Sigma = 4 * np.eye(4) + Sigma.T @ Sigma
            Sigma_inverse = np.linalg.inv( Sigma )
            Sigma_inverse = (Sigma_inverse + Sigma_inverse.T) /2.0
            Sigmas = np.append( Sigmas, np.copy( Sigma_inverse.reshape(1,-1) ), axis=0 )
        return Sigmas

    # Testing Parameters
    key = random.PRNGKey(100)
    n = 4
    N = 50
    key, subkey = random.split(key)
    param_w = random.uniform(subkey, shape=(N,1))[:,0] - 0.5#+ 0.5#+ 2.0  #0.5 work with Lr: 5.0

    key, subkey = random.split(key)
    param_mu = random.uniform(subkey, shape=(4,N))- 0.5 * np.ones((4,N)) #- 3.5 * np.ones((4,N))

    Sigma_invs = generate_psd_matrix_inverse()
    params_policy = np.append( param_w, param_mu.reshape(-1,1)[:,0] )

    # Testing input state
    test_X = random.uniform(key, shape=(4,1))

    # run once to force JIT compilation
    t0 = time.time()
    test_u = policy( params_policy, Sigma_invs, test_X, n, N )
    print(f"time scan:{time.time()-t0}")

    t0 = time.time()
    test_u = policy( params_policy, Sigma_invs, test_X, n, N )
    print(f"time scan again:{time.time()-t0}")

    t0 = time.time()
    test_u = policy_jit( params_policy, Sigma_invs, test_X, n, N )
    print(f"time:{time.time()-t0}")

    t0 = time.time()
    test_u = policy_jit( params_policy, Sigma_invs, test_X, n, N )
    print(f"time jit again:{time.time()-t0}")

    print("u", test_u)

    t0 = time.time()
    test_u = policy_grad_jit( params_policy, Sigma_invs, test_X, n, N )
    print(f"time:{time.time()-t0}")

    t0 = time.time()
    test_u = policy_grad_jit( params_policy, Sigma_invs, test_X, n, N )
    print(f"time jit again:{time.time()-t0}")

    # print("u grad", test_u)
