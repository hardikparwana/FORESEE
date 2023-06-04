import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
#from jax import jit

def dynamics_step( base_term, state_dot, dt ):
    next_state = base_term + state_dot * dt
#     print(f"next_state:{next_state}")
    return next_state

def dynamics_xdot(state, action = np.array([0])):
    return 100*np.array([np.cos(state[0,0]), np.sin(state[1,0])]).reshape(-1,1)

# assume this is true dynamics
def dynamics_xdot_noisy(state, action = np.array([0])):
    xdot = dynamics_xdot(state, action)
    error_square = 0.01 + np.square(xdot) # /2  #never let it be 0!!!!
    cov = np.diag( error_square[:,0] )
    xdot = xdot + xdot/2 #X_dot = X_dot + X_dot/6
    return xdot, cov

# @jit
def get_mean( sigma_points, weights ):
    weighted_points = sigma_points * weights[0]
    mu = np.sum( weighted_points, 1 ).reshape(-1,1)
    return mu

# @jit
def get_mean_cov(sigma_points, weights):
    
    # mean
    weighted_points = sigma_points * weights[0]
    mu = np.sum( weighted_points, 1 ).reshape(-1,1)
    
    # covariance
    centered_points = sigma_points - mu
    weighted_centered_points = centered_points * weights[0] 
    cov = weighted_centered_points @ centered_points.T
    return mu, cov

#@jit
def get_ut_cov_root_diagonal(cov):
    offset = 0.000  # TODO: make sure not zero here
    root0 = np.sqrt((offset+cov[0,0]))
    root1 = np.sqrt((offset+cov[1,1]))
    # return cov
    root_term = np.diag( np.array([root0, root1]) )
    return root_term

#@jit
def initialize_sigma_points(X):
    # return 2N + 1 points
    n = X.shape[0]
    num_points = 2*n + 1
    sigma_points = np.repeat( X, num_points, axis=1 )
    weights = np.ones((1,num_points)) * 1.0/( num_points )
    return sigma_points, weights

# @jit
def generate_sigma_points_gaussian( mu, cov_root, base_term, factor ):
    n = mu.shape[0]     
    N = 2*n + 1 # total points

    k = 0.5 # n-3 # 0.5**
    new_points = dynamics_step(base_term, mu, factor) # new_points = base_term + factor * mu
    new_weights = np.array([[1.0*k/(n+k)]])
    for i in range(n):
        new_points = np.append( new_points, dynamics_step(base_term, (mu - np.sqrt(n+k)*cov_root[:,i].reshape(-1,1)), factor) , axis = 1 )
        new_points = np.append( new_points, dynamics_step(base_term, (mu + np.sqrt(n+k)*cov_root[:,i].reshape(-1,1)), factor) , axis = 1 )
        new_weights = np.append( new_weights, np.array([[1.0/(n+k)/2.0]]), axis = 1 )
        new_weights = np.append( new_weights, np.array([[1.0/(n+k)/2.0]]), axis = 1 )
    return new_points, new_weights

# # Sanity check of above function
mu = np.array([1.5, 2.5]).reshape(-1,1)
# cov = np.array([
#     [0.6, 0],
#     [0.0, 3.4]
# ])
mu = np.array([0.075, 0.075]).reshape(-1,1)
cov = np.array([
    [0.00252, 0],
    [0.0, 0.00252]
])
# mu = np.array([1.5]).reshape(-1,1)
# cov = np.array([[1.2]])
cov_root = np.sqrt( cov )

# # p, w = initialize_sigma_points(mu)#, cov)
# # mu1, cov1 = get_mean_cov( p, w, w )
# # print(f"org mu: {mu1}, cov: {cov1}")

points, weights = generate_sigma_points_gaussian( mu, cov_root, np.zeros((2,1)), 1.0 )
mu2, cov2 = get_mean_cov( points, weights )
print( f"mean:{mu2}, cov:{cov2}" )
# exit()
