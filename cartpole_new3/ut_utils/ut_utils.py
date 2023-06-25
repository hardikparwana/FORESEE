import jax.numpy as np
from jax import jit, lax
from robot_models.cartpole2D_mcpilco import get_state_dot_noisy, step_using_xdot, get_state_dot_noisy_rk4
from cartpole_new3.gp_utils import predict_with_gp_params
from cartpole_new3.cartpole_policy import policy
from utils.utils import wrap_angle

#check get_ut_cov_root_diagonal

@jit
def get_mean( sigma_points, weights ):
    weighted_points = sigma_points * weights[0]
    mu = np.sum( weighted_points, 1 ).reshape(-1,1)
    return mu

@jit
def get_mean_cov(sigma_points, weights, weights_cov):
    
    # mean
    weighted_points = sigma_points * weights[0]
    mu = np.sum( weighted_points, 1 ).reshape(-1,1)
    
    # covariance
    centered_points = sigma_points - mu
    weighted_centered_points = centered_points * weights_cov[0] 
    cov = weighted_centered_points @ centered_points.T
    return mu, cov

@jit
def get_ut_cov_root_diagonal(cov):
    offset = 0.0000000001 # TODOs set offset so that it is never zero
    root0 = np.sqrt((offset+cov[0,0]))
    root1 = np.sqrt((offset+cov[1,1]))
    root2 = np.sqrt((offset+cov[2,2]))
    root3 = np.sqrt((offset+cov[3,3]))
    root_term = np.diag( np.array([root0, root1, root2, root3]) )
    # root_term = np.zeros((4,4))
    return root_term

@jit
def initialize_sigma_points(X):
    # return 2N + 1 points
    n = X.shape[0]
    num_points = 2*n + 1
    sigma_points = np.repeat( X, num_points, axis=1 )
    weights = np.ones((1,num_points)) * 1.0/( num_points )
    return sigma_points, weights, np.copy(weights)

# @jit
def generate_sigma_points_gaussian( mu, cov_root, base_term, factor ):
    n = mu.shape[0]     
    N = 2*n + 1 # total points

    # k = 1.0#0.5 # n-3 # 0.5**
    
    alpha = 1.0
    beta = 0.0#2.0#2.0 # optimal for gaussian
    k = 1.0
    Lambda = alpha**2 * ( n+k ) - n
    new_points = step_using_xdot(base_term, mu, factor) # new_points = base_term + factor * mu
    new_weights = np.array([[1.0*Lambda/(n+Lambda)]])    
    new_weights_cov = np.array([[ 1.0*Lambda/(n+Lambda) + 1 - alpha**2 + beta]])
    for i in range(n):
        new_points = np.append( new_points, step_using_xdot(base_term, (mu - np.sqrt(n+Lambda) * cov_root[:,i].reshape(-1,1)), factor) , axis = 1 )
        new_points = np.append( new_points, step_using_xdot(base_term, (mu + np.sqrt(n+Lambda) * cov_root[:,i].reshape(-1,1)), factor) , axis = 1 )
        new_weights = np.append( new_weights, np.array([[1.0/(n+Lambda)/2.0]]), axis = 1 )
        new_weights = np.append( new_weights, np.array([[1.0/(n+Lambda)/2.0]]), axis = 1 )
        new_weights_cov = np.append( new_weights_cov, np.array([[1.0/(n+Lambda)/2.0]]), axis = 1 )
        new_weights_cov = np.append( new_weights_cov, np.array([[1.0/(n+Lambda)/2.0]]), axis = 1 )
    # print(f"weights1: {new_weights}")
    # print(f"weights2: {new_weights_cov}")
    return new_points, new_weights, new_weights_cov

# def get_state_dot_with_gp(state, control, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y):
@jit
def get_next_state_with_gp(state, control, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y):
    test_x = np.append(state.reshape(1,-1), control.reshape(1,-1), axis=1)
    # mu1, var1 = predict_with_gp_params(gp_params1, gp_train_x, gp_train_y[:,0].reshape(-1,1), test_x)
    mu2, var2 = predict_with_gp_params(gp_params2, gp_train_x, gp_train_y[:,1].reshape(-1,1), test_x)
    # mu3, var3 = predict_with_gp_params(gp_params3, gp_train_x, gp_train_y[:,2].reshape(-1,1), test_x)
    mu4, var4 = predict_with_gp_params(gp_params4, gp_train_x, gp_train_y[:,3].reshape(-1,1), test_x)
    dt = 0.02
    mu1, var1 = np.array([ state[0,0] + dt * state[1,0]  ]), np.array([0.0])
    mu3, var3 = np.array([ wrap_angle(state[2,0] + dt * state[3,0])  ]), np.array([0,0])
    return np.concatenate((mu1, mu2, mu3, mu4)).reshape(-1,1), np.diag( np.concatenate( (var1, var2, var3, var4) ) )

def get_next_states_with_gp( states, control_inputs, gps ):
    states_gp = np.concatenate( (states[0,:].reshape(1,-1), states[1,:].reshape(1,-1), states[3,:].reshape(1,-1), np.sin(states[2,:]).reshape(1,-1), np.cos(states[2,:]).reshape(1,-1)), axis=0 )
    test_x = np.append( states_gp, control_inputs.reshape(1,-1), axis=0 ).T
    dt = 0.05
    
    pred2 = gps[1](test_x)
    mu2, var2 = pred2.mean(), pred2.variance()
    
    pred4 = gps[3](test_x)
    mu4, var4 = pred4.mean(), pred4.variance()
    
     # pred1 = gps[0](test_x)
    # mu1, var1 = pred1.mean(), pred1.variance()
    # mu1, var1 = np.array([ states[0,:] + dt * states[1,:]  ]), np.zeros((1,9))
    mu1, var1 = np.array([ states[0,:] + dt * states[1,:] + dt / 2 * mu2  ]), np.zeros((1,9))
    
    # pred3 = gps[2](test_x)
    # mu3, var3 = pred3.mean(), pred3.variance()
    # mu3, var3 = np.array([ wrap_angle(states[2,:] + dt * states[3,:])  ]), np.zeros((1,9))
    mu3, var3 = np.array([ states[2,:] + dt * states[3,:] + dt / 2 * mu4   ]), np.zeros((1,9))
    
    return np.concatenate((mu1.reshape(1,-1), mu2.reshape(1,-1), mu3.reshape(1,-1), mu4.reshape(1,-1)), axis=0), np.concatenate( (var1.reshape(1,-1), var2.reshape(1,-1), var3.reshape(1,-1), var4.reshape(1,-1)), axis=0 )
    
    
@jit
def sigma_point_expand_with_gp(sigma_points, weights, weights_cov, control, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y):
   
    n, N = sigma_points.shape   
    # dt_outer = 0  
    #TODO  
    mu, cov = get_next_state_with_gp(sigma_points[:,0].reshape(-1,1), control.reshape(-1,1), gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y )
    root_term = get_ut_cov_root_diagonal(cov) 
    # temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,0].reshape(-1,1), 1.0 )
    temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, np.zeros((sigma_points.shape[0],1)), 1.0 )
    new_points = np.copy( temp_points )
    new_weights = ( np.copy( temp_weights1 ) * weights[0,0]).reshape(1,-1)
    new_weights_cov = ( np.copy( temp_weights2 ) * weights_cov[0,0]).reshape(1,-1)
        
    for i in range(1,N):
        mu, cov = get_next_state_with_gp(sigma_points[:,i].reshape(-1,1), control.reshape(-1,1), gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y )
        root_term = get_ut_cov_root_diagonal(cov)           
        # temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,i].reshape(-1,1), 1.0 )
        temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, np.zeros((sigma_points.shape[0],1)), 1.0 )
        new_points = np.append(new_points, temp_points, axis=1 )
        new_weights = np.append( new_weights, (temp_weights1 * weights[0,i]).reshape(1,-1) , axis=1 )
        new_weights_cov = np.append( new_weights_cov, (temp_weights2 * weights_cov[0,i]).reshape(1,-1) , axis=1 )

    return new_points, new_weights, new_weights_cov

@jit
def sigma_point_expand_with_gp_input(sigma_points, weights, weights_cov, params_policy, gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y):
   
    n, N = sigma_points.shape   
    # dt_outer = 0  
    #TODO  
    control = policy( sigma_points[:,0].reshape(-1,1), params_policy )
    mu, cov = get_next_state_with_gp(sigma_points[:,0].reshape(-1,1), control.reshape(-1,1), gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y )
    root_term = get_ut_cov_root_diagonal(cov) 
    # temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,0].reshape(-1,1), 1.0 )
    temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, np.zeros((sigma_points.shape[0],1)), 1.0 )
    new_points = np.copy( temp_points )
    new_weights = ( np.copy( temp_weights1 ) * weights[0,0]).reshape(1,-1)
    new_weights_cov = ( np.copy( temp_weights2 ) * weights_cov[0,0]).reshape(1,-1)
        
    for i in range(1,N):
        control = policy( sigma_points[:,i].reshape(-1,1), params_policy )
        mu, cov = get_next_state_with_gp(sigma_points[:,i].reshape(-1,1), control.reshape(-1,1), gp_params1, gp_params2, gp_params3, gp_params4, gp_train_x, gp_train_y )
        root_term = get_ut_cov_root_diagonal(cov)           
        # temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,i].reshape(-1,1), 1.0 )
        temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, np.zeros((sigma_points.shape[0],1)), 1.0 )
        new_points = np.append(new_points, temp_points, axis=1 )
        new_weights = np.append( new_weights, (temp_weights1 * weights[0,i]).reshape(1,-1) , axis=1 )
        new_weights_cov = np.append( new_weights_cov, (temp_weights2 * weights_cov[0,i]).reshape(1,-1) , axis=1 )

    return new_points, new_weights, new_weights_cov

@jit
def sigma_point_expand_with_mean_cov(points_means, points_covs, weights, weights_cov):
   
    n, N = points_means.shape   
    mu, cov = points_means[:,0].reshape(-1,1), np.diag( points_covs[:,0] )
    root_term = get_ut_cov_root_diagonal(cov) 
    temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, np.zeros((points_means.shape[0],1)), 1.0 )
    new_points = np.copy( temp_points )
    new_weights = ( np.copy( temp_weights1 ) * weights[0,0]).reshape(1,-1)
    new_weights_cov = ( np.copy( temp_weights2 ) * weights_cov[0,0]).reshape(1,-1)
        
    for i in range(1,N):
        mu, cov = points_means[:,i].reshape(-1,1), np.diag( points_covs[:,i] )
        root_term = get_ut_cov_root_diagonal(cov)           
        temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, np.zeros((points_means.shape[0],1)), 1.0 )
        new_points = np.append(new_points, temp_points, axis=1 )
        new_weights = np.append( new_weights, (temp_weights1 * weights[0,i]).reshape(1,-1) , axis=1 )
        new_weights_cov = np.append( new_weights_cov, (temp_weights2 * weights_cov[0,i]).reshape(1,-1) , axis=1 )

    return new_points, new_weights, new_weights_cov

@jit
def sigma_point_expand(sigma_points, weights, weights_cov, control, dt):
   
    n, N = sigma_points.shape   
    # dt_outer = 0  
    #TODO  
    mu, cov = get_state_dot_noisy(sigma_points[:,0].reshape(-1,1), control.reshape(-1,1) )
    root_term = get_ut_cov_root_diagonal(cov) 
    temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,0].reshape(-1,1), dt )
    new_points = np.copy( temp_points )
    new_weights = ( np.copy( temp_weights1 ) * weights[0,0]).reshape(1,-1)
    new_weights_cov = ( np.copy( temp_weights2 ) * weights_cov[0,0]).reshape(1,-1)
        
    for i in range(1,N):
        mu, cov = get_state_dot_noisy(sigma_points[:,i].reshape(-1,1), control.reshape(-1,1) )
        root_term = get_ut_cov_root_diagonal(cov)           
        temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,i].reshape(-1,1), dt )
        new_points = np.append(new_points, temp_points, axis=1 )
        new_weights = np.append( new_weights, (temp_weights1 * weights[0,i]).reshape(1,-1) , axis=1 )
        new_weights_cov = np.append( new_weights_cov, (temp_weights2 * weights_cov[0,i]).reshape(1,-1) , axis=1 )

    return new_points, new_weights, new_weights_cov

@jit
def sigma_point_expand_rk4(sigma_points, weights, weights_cov, control, dt):
   
    n, N = sigma_points.shape   
    # dt_outer = 0  
    #TODO  
    mu, cov = get_state_dot_noisy_rk4(sigma_points[:,0].reshape(-1,1), control.reshape(-1,1), dt )
    root_term = get_ut_cov_root_diagonal(cov) 
    temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,0].reshape(-1,1), dt )
    new_points = np.copy( temp_points )
    new_weights = ( np.copy( temp_weights1 ) * weights[0,0]).reshape(1,-1)
    new_weights_cov = ( np.copy( temp_weights2 ) * weights_cov[0,0]).reshape(1,-1)
        
    for i in range(1,N):
        mu, cov = get_state_dot_noisy_rk4(sigma_points[:,i].reshape(-1,1), control.reshape(-1,1), dt )
        root_term = get_ut_cov_root_diagonal(cov)           
        temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,i].reshape(-1,1), dt )
        new_points = np.append(new_points, temp_points, axis=1 )
        new_weights = np.append( new_weights, (temp_weights1 * weights[0,i]).reshape(1,-1) , axis=1 )
        new_weights_cov = np.append( new_weights_cov, (temp_weights2 * weights_cov[0,i]).reshape(1,-1) , axis=1 )

    return new_points, new_weights, new_weights_cov

@jit
def sigma_point_expand_rk4_input(sigma_points, weights, weights_cov, params_policy, dt):
   
    n, N = sigma_points.shape   
    # dt_outer = 0  
    #TODO  
    control = policy( sigma_points[:,0].reshape(-1,1), params_policy )
    mu, cov = get_state_dot_noisy_rk4(sigma_points[:,0].reshape(-1,1), control.reshape(-1,1), dt )
    root_term = get_ut_cov_root_diagonal(cov) 
    temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,0].reshape(-1,1), dt )
    new_points = np.copy( temp_points )
    new_weights = ( np.copy( temp_weights1 ) * weights[0,0]).reshape(1,-1)
    new_weights_cov = ( np.copy( temp_weights2 ) * weights_cov[0,0]).reshape(1,-1)
        
    for i in range(1,N):
        control = policy( sigma_points[:,i].reshape(-1,1), params_policy )
        mu, cov = get_state_dot_noisy_rk4(sigma_points[:,i].reshape(-1,1), control.reshape(-1,1), dt )
        root_term = get_ut_cov_root_diagonal(cov)           
        temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,i].reshape(-1,1), dt )
        new_points = np.append(new_points, temp_points, axis=1 )
        new_weights = np.append( new_weights, (temp_weights1 * weights[0,i]).reshape(1,-1) , axis=1 )
        new_weights_cov = np.append( new_weights_cov, (temp_weights2 * weights_cov[0,i]).reshape(1,-1) , axis=1 )

    return new_points, new_weights, new_weights_cov

@jit
def sigma_point_expand_rk4_zero_input(sigma_points, weights, weights_cov, params_policy, dt):
   
    n, N = sigma_points.shape   
    # dt_outer = 0  
    #TODO  
    control = np.array([[0.0]])
    mu, cov = get_state_dot_noisy_rk4(sigma_points[:,0].reshape(-1,1), control.reshape(-1,1), dt )
    root_term = get_ut_cov_root_diagonal(cov) 
    temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,0].reshape(-1,1), dt )
    new_points = np.copy( temp_points )
    new_weights = ( np.copy( temp_weights1 ) * weights[0,0]).reshape(1,-1)
    new_weights_cov = ( np.copy( temp_weights2 ) * weights_cov[0,0]).reshape(1,-1)
        
    for i in range(1,N):
        control = np.array([[0.0]])
        mu, cov = get_state_dot_noisy_rk4(sigma_points[:,i].reshape(-1,1), control.reshape(-1,1), dt )
        root_term = get_ut_cov_root_diagonal(cov)           
        temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,i].reshape(-1,1), dt )
        new_points = np.append(new_points, temp_points, axis=1 )
        new_weights = np.append( new_weights, (temp_weights1 * weights[0,i]).reshape(1,-1) , axis=1 )
        new_weights_cov = np.append( new_weights_cov, (temp_weights2 * weights_cov[0,i]).reshape(1,-1) , axis=1 )

    return new_points, new_weights, new_weights_cov

@jit
def sigma_point_expand_rk4_zero_input_noisy(sigma_points, weights, weights_cov, params_policy, dt):
   
    n, N = sigma_points.shape   
    # dt_outer = 0  
    #TODO  
    control = np.array([[0.0]])
    mu, cov = get_state_dot_noisy(sigma_points[:,0].reshape(-1,1), control.reshape(-1,1) )
    root_term = get_ut_cov_root_diagonal(cov) 
    temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,0].reshape(-1,1), dt )
    new_points = np.copy( temp_points )
    new_weights = ( np.copy( temp_weights1 ) * weights[0,0]).reshape(1,-1)
    new_weights_cov = ( np.copy( temp_weights2 ) * weights_cov[0,0]).reshape(1,-1)
        
    for i in range(1,N):
        control = np.array([[0.0]])
        mu, cov = get_state_dot_noisy(sigma_points[:,i].reshape(-1,1), control.reshape(-1,1) )
        root_term = get_ut_cov_root_diagonal(cov)           
        temp_points, temp_weights1, temp_weights2 = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,i].reshape(-1,1), dt )
        new_points = np.append(new_points, temp_points, axis=1 )
        new_weights = np.append( new_weights, (temp_weights1 * weights[0,i]).reshape(1,-1) , axis=1 )
        new_weights_cov = np.append( new_weights_cov, (temp_weights2 * weights_cov[0,i]).reshape(1,-1) , axis=1 )

    return new_points, new_weights, new_weights_cov

# @jit
def sigma_point_compress( sigma_points, weights, weights_cov ):
    mu, cov = get_mean_cov( sigma_points, weights, weights_cov )
    cov_root_term = get_ut_cov_root_diagonal( cov )  
    base_term = np.zeros((mu.shape))
    return generate_sigma_points_gaussian( mu, cov_root_term, base_term, np.array([1.0]) )

def reward_UT_Mean_Evaluator_basic(sigma_points, weights, weights_cov):
    # return np.sum(sigma_points)
    mu = 0
    mu = mu + mc_pilco_reward( sigma_points[:,0].reshape(-1,1)  ) *  weights[0,0]
    for i in range(1, sigma_points.shape[1]):
        mu = mu + mc_pilco_reward( sigma_points[:,i].reshape(-1,1)  ) *  weights[0,i]
    return mu
reward_UT_Mean_Evaluator_basic_jit = jit(reward_UT_Mean_Evaluator_basic)
reward_UT_Mean_Evaluator_basic_sum = lambda a,b: np.sum(reward_UT_Mean_Evaluator_basic(a,b)[0])

# minimize reward
def compute_reward( state ):
    # return np.sum(state[2,0]+state[1,0])
    theta = state[2,0] # want theta and theta_dot to be 0
    speed = state[1,0]
    pos = state[0,0]
    # return np.square(theta-0.0)
    # print(f"theta:{theta}")
    # return state[1,0]
    # return -10*np.cos(theta)#+0.08*np.square(pos/2)#+0.001*np.square(speed)
    # return -100*np.cos(theta)+0.1*np.square(speed)+10*np.square(pos)
    # return - 100 * np.cos(theta) + 0.1 * np.square(pos)
    return 100 * np.cos(theta) + 1.0 * np.square(pos)


def mc_pilco_reward(state):
    """ 
    Cost function given by the combination of the saturated distance between |theta| and 'target angle', and between x and 'target position'.
    """   
    x = state[0,0]#states_sequence[:,:,pos_index]
    theta = state[2,0]#states_sequence[:,:,angle_index]
    lengthscales = [3.0, 1.0] # theta, p

    target_x = 0#target_state[1]
    target_theta = np.pi#  target_state[0]

    # return ( (np.abs(theta)-target_theta) / lengthscales[0] )**2 + ( (x-target_x)/lengthscales[1] )**2
    return (1-np.exp( -( (np.abs(theta)-target_theta) / lengthscales[0] )**2 - ( (x-target_x)/lengthscales[1] )**2 ) )
    # return (1-np.exp( -( (np.cos(theta)-np.cos(target_theta)) / lengthscales[0] )**2 - ( (x-target_x)/lengthscales[1] )**2 ) )