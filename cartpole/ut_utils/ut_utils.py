import jax.numpy as np
from robot_models.cartpole2D import get_state_dot_noisy

def get_mean( sigma_points, weights ):
    weighted_points = sigma_points * weights[0]
    mu = np.sum( weighted_points, 1 ).reshape(-1,1)
    return mu

def get_mean_cov(sigma_points, weights):
    
    # mean
    weighted_points = sigma_points * weights[0]
    mu = np.sum( weighted_points, 1 ).reshape(-1,1)
    
    # covariance
    centered_points = sigma_points - mu
    weighted_centered_points = centered_points * weights[0] 
    cov = weighted_centered_points @ centered_points.T
    return mu, cov

def get_ut_cov_root(cov):
    k = -1
    n = cov.shape[0]
    root_term = jax.scipy.linalg.sqrtm((n+k)*cov)
    return root_term

def get_ut_cov_root_diagonal(cov):
    k = -1
    n = cov.shape[0]
        
    root0 = np.sqrt(cov[0,0])
    root1 = np.sqrt(cov[1,1])
    root2 = np.sqrt(cov[2,2])
    root3 = np.sqrt(cov[3,3])
    
    #TODO: check this formula
    root_term = (n+k) * np.diag( np.array([root0, root1, root2, root3]) )
    return root_term

def initialize_sigma_points(X):
    # return 2N + 1 points
    n = X.shape[0]
    num_points = 2*n + 1
    sigma_points = np.repeat( X, num_points, axis=1 )
    weights = np.ones((1,num_points)) * 1.0/( num_points )
    return sigma_points, weights

def generate_sigma_points( mu, cov_root, base_term, factor ):
    
    n = mu.shape[0]     
    N = 2*n + 1 # total points

    # TODO
    # k = n - 3
    k = 0.5 #2

    new_points = base_term + factor * mu
    new_weights = np.array([[1.0*k/(n+k)]])
    for i in range(n):
        new_points = np.append( new_points, base_term + factor * (mu - cov_root[:,i].reshape(-1,1)) , axis = 1 )
        new_points = np.append( new_points, base_term + factor * (mu + cov_root[:,i].reshape(-1,1)) , axis = 1 )

        new_weights = np.append( new_weights, np.array([[1.0*k/(n+k)]]), axis = 1 )
        new_weights = np.append( new_weights, np.array([[1.0*k/(n+k)]]), axis = 1 )

    return new_points, new_weights


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

def sigma_point_compress( sigma_points, weights ):
    mu, cov = get_mean_cov( sigma_points, weights )
    cov_root_term = get_ut_cov_root_diagonal( cov )  
    base_term = np.zeros((mu.shape))
    return generate_sigma_points( mu, cov_root_term, base_term, np.array([1.0]) )

def reward_UT_Mean_Evaluator_basic(sigma_points, weights):
    mu = compute_reward( sigma_points[:,0].reshape(-1,1)  ) *  weights[0,0]
    for i in range(1, sigma_points.shape[1]):
        mu = mu + compute_reward( sigma_points[:,i].reshape(-1,1)  ) *  weights[0,i]
    return mu

def compute_reward( state ):
    theta = state[2,0] # want theta and theta_dot to be 0
    speed = state[1,0]
    pos = state[0,0]
    return - 100 * np.cos(theta) + 0.1 * np.square(speed)
