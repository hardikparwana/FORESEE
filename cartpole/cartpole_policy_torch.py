import torch
import time
# Nonlinear RBF network
def policy(param_w, param_mu, param_Sigma, X):
    n = 4 # dim of state
    m = 1 # dim of input
    N = 50 # number of basis functions

    # First basis function
    diff = X - param_mu[:,0].reshape(-1,1)
 
    # Given lower triangular
    Sigma = torch.diag( param_Sigma[0,0:4] )
    Sigma[1,0] = param_Sigma[ 0, 4 ]; Sigma[2,0] = param_Sigma[ 0, 5 ]; Sigma[2,1] = param_Sigma[ 0, 6 ]; Sigma[3,0] = param_Sigma[ 0, 7 ]; Sigma[3,1] = param_Sigma[ 0, 8 ]; Sigma[3,2] = param_Sigma[ 0, 9 ];
    Sigma_final = 4 * torch.eye(4) + torch.transpose(Sigma, 0, 1) @ Sigma
    phi = torch.exp( -0.5 * diff.T @ torch.inverse(Sigma_final) @ diff )

    pi = param_w[0] * phi
    
    # Remaining basis functions
    for i in range(1,N):
        diff = X - param_mu[:,i].reshape(-1,1)
     
        # Given lower triangular params
        Sigma = torch.diag( param_Sigma[i,0:4] )
        Sigma[1,0] = param_Sigma[ i, 4 ]; Sigma[2,0] = param_Sigma[ i, 5 ]; Sigma[2,1] = param_Sigma[ i, 6 ]; Sigma[3,0] = param_Sigma[ i, 7 ]; Sigma[3,1] = param_Sigma[ i, 8 ]; Sigma[3,2] = param_Sigma[ i, 9 ];
        Sigma_final = 4 * torch.eye(4) + torch.transpose(Sigma, 0, 1) @ Sigma
        phi = torch.exp( -0.5 * diff.T @ torch.inverse(Sigma_final) @ diff )
        
        pi = pi + param_w[i] * phi
        
    # if torch.abs( pi ) > 10:
    #     pi = pi / torch.abs(pi) * 10
    return pi


        
print("Testing Cart Pole Policy")

N = 50
param_w = torch.rand(N)
param_mu = torch.rand((4,N))
param_Sigma = torch.rand((N,10))
X = torch.rand(4).reshape(-1,1)

traced_policy = torch.jit.trace( policy, ( param_w, param_mu, param_Sigma, X ) )  
u =  traced_policy( param_w, param_mu, param_Sigma, X )
u =  traced_policy( param_w, param_mu, param_Sigma, X )
t0 = time.time()
u =  traced_policy( param_w, param_mu, param_Sigma, X )
print(f"time:{time.time()-t0}")
print("u", u)