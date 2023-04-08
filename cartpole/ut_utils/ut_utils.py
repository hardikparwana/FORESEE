import jax.numpy as np
from jax import jit, lax
from robot_models.cartpole2D import get_state_dot_noisy

def get_mean( sigma_points, weights ):
    weighted_points = sigma_points * weights[0]
    mu = np.sum( weighted_points, 1 ).reshape(-1,1)
    return mu
get_mean_jit = jit(get_mean)

def get_mean_cov(sigma_points, weights):
    
    # mean
    weighted_points = sigma_points * weights[0]
    mu = np.sum( weighted_points, 1 ).reshape(-1,1)
    
    # covariance
    centered_points = sigma_points - mu
    weighted_centered_points = centered_points * weights[0] 
    cov = weighted_centered_points @ centered_points.T
    return mu, cov
get_mean_cov_jit = jit(get_mean_cov)

# def get_ut_cov_root(cov):
#     k = -1
#     n = cov.shape[0]
#     root_term = jax.scipy.linalg.sqrtm((n+k)*cov)
#     return root_term
# get_ut_cov_root_jit = jit(get_ut_cov_root)

def get_ut_cov_root_diagonal(cov):
    k = 0.5#-1
    n = cov.shape[0]
    offset = 0.001
    root0 = np.sqrt((n+k)*(offset+cov[0,0]))
    root1 = np.sqrt((n+k)*(offset+cov[1,1]))
    root2 = np.sqrt((n+k)*(offset+cov[2,2]))
    root3 = np.sqrt((n+k)*(offset+cov[3,3]))
    # return cov
    root_term = np.diag( np.array([root0, root1, root2, root3]) )
    return root_term
get_ut_cov_root_diagonal_jit = jit(get_ut_cov_root_diagonal)

def initialize_sigma_points(X):
    # return 2N + 1 points
    n = X.shape[0]
    num_points = 2*n + 1
    sigma_points = np.repeat( X, num_points, axis=1 )
    weights = np.ones((1,num_points)) * 1.0/( num_points )
    return sigma_points, weights
initialize_sigma_points_jit = jit(initialize_sigma_points)

def generate_sigma_points( mu, cov_root, base_term, factor ):
    
    n = mu.shape[0]     
    N = 2*n + 1 # total points

    # TODO
    k = 0.5 # n-3 # 0.5**

    new_points = base_term + factor * mu
    new_weights = np.array([[1.0*k/(n+k)]])
    for i in range(n):
        new_points = np.append( new_points, base_term + factor * (mu - cov_root[:,i].reshape(-1,1)) , axis = 1 )
        new_points = np.append( new_points, base_term + factor * (mu + cov_root[:,i].reshape(-1,1)) , axis = 1 )

        new_weights = np.append( new_weights, np.array([[1.0*k/(n+k)/2.0]]), axis = 1 )
        new_weights = np.append( new_weights, np.array([[1.0*k/(n+k)/2.0]]), axis = 1 )
    return new_points, new_weights
    # def body(t,inputs):
    #     new_points, new_weights = inputs
    #     new_points = np.append( new_points, base_term + factor * (mu - cov_root[:,t].reshape(-1,1)) , axis = 1 )
    #     new_points = np.append( new_points, base_term + factor * (mu + cov_root[:,t].reshape(-1,1)) , axis = 1 )
    #     new_weights = np.append( new_weights, np.array([[1.0*k/(n+k)/2.0]]), axis = 1 )
    #     new_weights = np.append( new_weights, np.array([[1.0*k/(n+k)/2.0]]), axis = 1 )
    #     return new_points, new_weights
    # return lax.fori_loop( 0, N, body, (new_points, new_weights) )
    
generate_sigma_points_jit = jit(generate_sigma_points)

# def sigma_point_expand_JIT(GA, PE, gp_params, K_invs, noise, X_s, Y_s, sigma_points, weights, control, dt_outer, dt_inner, polemass_length, gravity, length, masspole, total_mass, tau):#, gps):
def sigma_point_expand(sigma_points, weights, control, dt_outer, dynamics_params):#, gps):
   
    n, N = sigma_points.shape   
      
    #TODO  
    mu, cov = get_state_dot_noisy(sigma_points[:,0].reshape(-1,1), control.reshape(-1,1), dynamics_params)
    root_term = get_ut_cov_root_diagonal(cov) 
    temp_points, temp_weights = generate_sigma_points( mu, root_term, sigma_points[:,0].reshape(-1,1), dt_outer )
    new_points = np.copy( temp_points )
    new_weights = ( np.copy( temp_weights ) * weights[0,0]).reshape(1,-1)
        
    for i in range(1,N):
        mu, cov = get_state_dot_noisy(sigma_points[:,i].reshape(-1,1), control.reshape(-1,1), dynamics_params)
        root_term = get_ut_cov_root_diagonal(cov)           
        temp_points, temp_weights = generate_sigma_points( mu, root_term, sigma_points[:,i].reshape(-1,1), dt_outer )
        new_points = np.append(new_points, temp_points, axis=1 )
        new_weights = np.append( new_weights, (temp_weights * weights[0,i]).reshape(1,-1) , axis=1 )

    return new_points, new_weights
sigma_point_expand_jit = jit(sigma_point_expand)

def sigma_point_compress( sigma_points, weights ):
    mu, cov = get_mean_cov( sigma_points, weights )
    cov_root_term = get_ut_cov_root_diagonal( cov )  
    base_term = np.zeros((mu.shape))
    return generate_sigma_points( mu, cov_root_term, base_term, np.array([1.0]) )
sigma_point_compress_jit = jit(sigma_point_compress)


def reward_UT_Mean_Evaluator_basic(sigma_points, weights):
    mu = compute_reward( sigma_points[:,0].reshape(-1,1)  ) *  weights[0,0]
    for i in range(1, sigma_points.shape[1]):
        mu = mu + compute_reward( sigma_points[:,i].reshape(-1,1)  ) *  weights[0,i]
    return mu
reward_UT_Mean_Evaluator_basic_jit = jit(reward_UT_Mean_Evaluator_basic)

# minimize reward
def compute_reward( state ):
    theta = state[2,0] # want theta and theta_dot to be 0
    speed = state[1,0]
    pos = state[0,0]
    return - 100 * np.cos(theta)# + 0.1 * np.square(speed)
compute_reward_jit = jit(compute_reward)