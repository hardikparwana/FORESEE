import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import jax.numpy as jnp
from jax import jit
import numpy as np

@jit
def step(x,u,dt):
    return x+u*dt


def dynamics_step( base_term, state_dot, dt ):
    next_state = base_term + state_dot * dt
#     print(f"next_state:{next_state}")
    return next_state

# extended unicycle
def dynamics_xdot(state, action):
    return jnp.array([  state[3,0]*jnp.cos(state[2,0]), state[3,0]*jnp.sin(state[2,0]), action[1,0], action[0,0]  ]).reshape(-1,1)

def dynamics_xdot_np(state, action):
    return np.array([  state[3,0]*np.cos(state[2,0]), state[3,0]*np.sin(state[2,0]), action[1,0], action[0,0]  ]).reshape(-1,1)

# assume a single control input
# assume this is true dynamics
def dynamics_xdot_noisy(state, action):
    xdot = dynamics_xdot(state, action)
    # return xdot, jnp.zeros((4,4))
    error_square = 0.01 + 0.1 * jnp.square(xdot) # /2  #never let it be 0!!!!
    cov = jnp.diag( error_square[:,0] )
    # cov = jnp.zeros((4,4))
    xdot = xdot + xdot/2 #X_dot = X_dot + X_dot/6
    return xdot, cov

def dynamics_xdot_noisy_np(state, action):
    xdot = dynamics_xdot_np(state, action)
    # return xdot, jnp.zeros((4,4))
    error_square = 0.01 + 0.1 * np.square(xdot) # /2  #never let it be 0!!!!
    cov = np.diag( error_square[:,0] )
    xdot = xdot + xdot/2 #X_dot = X_dot + X_dot/6
    return xdot, cov

@jit
def get_mean( sigma_points, weights ):
    weighted_points = sigma_points * weights[0]
    mu = jnp.sum( weighted_points, 1 ).reshape(-1,1)
    return mu

@jit
def get_mean_cov(sigma_points, weights):
    
    # mean
    weighted_points = sigma_points * weights[0]
    mu = jnp.sum( weighted_points, 1 ).reshape(-1,1)
    
    # covariance
    centered_points = sigma_points - mu
    # weighted_centered_points = centered_points * weights[0] 
    # cov = weighted_centered_points @ centered_points.T
    cov = jnp.diag(jnp.sum(centered_points**2 * weights[0], axis=1))
    return mu, cov

def get_mean_cov_np(sigma_points, weights):
    
    # mean
    weighted_points = sigma_points * weights[0]
    mu = np.sum( weighted_points, 1 ).reshape(-1,1)
    
    # covariance
    centered_points = sigma_points - mu
    # weighted_centered_points = centered_points * weights[0] 
    # cov = weighted_centered_points @ centered_points.T
    cov = np.diag(np.sum(centered_points**2 * weights[0], axis=1))
    return mu, cov

def get_ut_cov_root_diagonal(cov):
    # return jnp.zeros((4,4))
    offset = 0.000  # TODO: make sure not zero here
    root0 = jnp.sqrt((offset+cov[0,0]))
    root1 = jnp.sqrt((offset+cov[1,1]))
    root2 = jnp.sqrt((offset+cov[2,2]))
    root3 = jnp.sqrt((offset+cov[3,3]))
    # return cov
    root_term = jnp.diag( jnp.array([root0, root1, root2, root3]) )
    return root_term

@jit
def get_mean_cov_skew_kurt_for_generation( sigma_points, weights ):
    # mean
    weighted_points = sigma_points * weights[0]
    mu = jnp.sum( weighted_points, 1 ).reshape(-1,1)    
    centered_points = sigma_points - mu    
    cov = jnp.diag(jnp.sum(centered_points**2 * weights[0], axis=1))
    # return mu, cov, jnp.zeros((2,1)), jnp.zeros((2,1))
    
    skewness_temp = jnp.sum(centered_points**3 * weights[0], axis=1) #/ cov[0,0]**(3/2) # for scipy    
    skewness = skewness_temp[0] / cov[0,0]**(3/2)
    skewness = jnp.append(skewness, skewness_temp[1] / cov[1,1]**(3/2))
    skewness = jnp.append(skewness, skewness_temp[2] / cov[2,2]**(3/2))
    skewness = jnp.append(skewness, skewness_temp[3] / cov[3,3]**(3/2))
    kurt_temp = jnp.sum(centered_points**4 * weights[0], axis=1)# / cov[0,0]**(4/2)  # -3 # -3 for scipy
    kurt = kurt_temp[0]/cov[0,0]**(4/2)
    kurt = jnp.append(kurt, kurt_temp[1]/cov[1,1]**(4/2))
    kurt = jnp.append(kurt, kurt_temp[2]/cov[2,2]**(4/2))
    kurt = jnp.append(kurt, kurt_temp[3]/cov[3,3]**(4/2))
    
    # skewness = jnp.zeros((4,1))
    # kurt = jnp.sqrt(3) * jnp.ones((4,1))
    return mu, cov, skewness.reshape(-1,1), kurt.reshape(-1,1)

@jit
def generate_sigma_points_gaussian( mu, cov_root, base_term, factor ):
    n = mu.shape[0]     
    N = 2*n + 1 # total points

    alpha = 1.0
    beta = 0.0#2.0#2.0 # optimal for gaussian
    k = 1.0
    Lambda = alpha**2 * ( n+k ) - n
    new_points = dynamics_step(base_term, mu, factor) # new_points = base_term + factor * mu
    new_weights = jnp.array([[1.0*Lambda/(n+Lambda)]])    
    new_weights_cov = jnp.array([[ 1.0*Lambda/(n+Lambda) + 1 - alpha**2 + beta]])
    for i in range(n):
        new_points = jnp.append( new_points, dynamics_step(base_term, (mu - jnp.sqrt(n+Lambda) * cov_root[:,i].reshape(-1,1)), factor) , axis = 1 )
        new_points = jnp.append( new_points, dynamics_step(base_term, (mu + jnp.sqrt(n+Lambda) * cov_root[:,i].reshape(-1,1)), factor) , axis = 1 )
        new_weights = jnp.append( new_weights, jnp.array([[1.0/(n+Lambda)/2.0]]), axis = 1 )
        new_weights = jnp.append( new_weights, jnp.array([[1.0/(n+Lambda)/2.0]]), axis = 1 )
        new_weights_cov = jnp.append( new_weights_cov, jnp.array([[1.0/(n+Lambda)/2.0]]), axis = 1 )
        new_weights_cov = jnp.append( new_weights_cov, jnp.array([[1.0/(n+Lambda)/2.0]]), axis = 1 )

    return new_points, new_weights

@jit
def generate_sigma_points_gaussian_GenUT( mu, cov_root, skewness, kurt, base_term, factor ):
    n = mu.shape[0]     
    N = 2*n + 1 # total points
    # kurt = jnp.sqrt(3)
    u = 0.5 * ( - skewness + jnp.sqrt( 4 * kurt - 3 * ( skewness )**2 ) )
    # u = jnp.sqrt(3)*jnp.ones((4,1))
    v = u + skewness

    w2 = (1.0 / v) / (u+v)
    w1 = (w2 * v) / u
    w0 = jnp.array([1 - jnp.sum(w1) - jnp.sum(w2)])
    
    U = jnp.diag(u[:,0])
    V = jnp.diag(v[:,0])
    points0 = mu    
    points1 = mu - cov_root @ U
    points2 = mu + cov_root @ V
    new_points = jnp.concatenate( (points0, points1, points2), axis=1 )
    new_weights = jnp.concatenate( (w0.reshape(-1,1), w1.reshape(1,-1), w2.reshape(1,-1)), axis=1 )

    # new_points = dynamics_step(base_term, mu, factor)
    # new_weights = w0.reshape(1,-1)#jnp.array([[w0]])
    # for i in range(n):
    #     new_points = jnp.append( new_points, dynamics_step(base_term, (mu - u[i,0]*cov_root[:,i].reshape(-1,1)), factor) , axis = 1 )
    #     new_points = jnp.append( new_points, dynamics_step(base_term, (mu + v[i,0]*cov_root[:,i].reshape(-1,1)), factor) , axis = 1 )
    #     new_weights = jnp.append( new_weights, jnp.array([[w1[i,0]]]), axis = 1 )
    #     new_weights = jnp.append( new_weights, jnp.array([[w2[i,0]]]), axis = 1 )
    return new_points, new_weights

@jit
def sigma_point_expand(sigma_points, weights, control, dt):
   
    n, N = sigma_points.shape   
    #TODO  
    mu, cov = dynamics_xdot_noisy(sigma_points[:,0].reshape(-1,1), control.reshape(-1,1))
    root_term = get_ut_cov_root_diagonal(cov) 
    temp_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,0].reshape(-1,1), dt )
    new_points = jnp.copy( temp_points )
    new_weights = ( jnp.copy( temp_weights ) * weights[0,0]).reshape(1,-1)
        
    for i in range(1,N):
        mu, cov = dynamics_xdot_noisy(sigma_points[:,i].reshape(-1,1), control.reshape(-1,1))
        root_term = get_ut_cov_root_diagonal(cov)           
        temp_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,i].reshape(-1,1), dt )
        new_points = jnp.append(new_points, temp_points, axis=1 )
        new_weights = jnp.append( new_weights, (temp_weights * weights[0,i]).reshape(1,-1) , axis=1 )

    return new_points, new_weights

@jit
def sigma_point_compress( sigma_points, weights ):
    mu, cov = get_mean_cov( sigma_points, weights )
    cov_root_term = get_ut_cov_root_diagonal( cov )  
    base_term = jnp.zeros((mu.shape))
    return generate_sigma_points_gaussian( mu, cov_root_term, base_term, jnp.array([1.0]) )

@jit
def sigma_point_compress_GenUT( sigma_points, weights ):
    mu, cov, skewness, kurt = get_mean_cov_skew_kurt_for_generation( sigma_points, weights )
    # print(f"mu:{mu}, cov:{cov}, skewness:{skewness}, kurtosis:{kurt}")
    cov_root_term = get_ut_cov_root_diagonal( cov )  
    base_term = jnp.zeros((mu.shape))
    return generate_sigma_points_gaussian_GenUT( mu, cov_root_term, skewness, kurt, base_term, jnp.array([1.0]) )

@jit
def foresee_propagate_GenUT( sigma_points, weights, action, dt ):
    
    expanded_sigma_points, expanded_weights = sigma_point_expand( sigma_points, weights, action, dt )
    compressed_sigma_points, compressed_weights = sigma_point_compress_GenUT(expanded_sigma_points, expanded_weights)
    return compressed_sigma_points, compressed_weights

@jit
def foresee_propagate( sigma_points, weights, action, dt ):

    #Expansion Layer
    expanded_sigma_points, expanded_weights = sigma_point_expand( sigma_points, weights, action, dt )
    compressed_sigma_points, compressed_weights = sigma_point_compress(expanded_sigma_points, expanded_weights)
    return compressed_sigma_points, compressed_weights

def confidence_ellipse(mu, cov, ax, n_std=3.0, facecolor='none', alpha=0.5, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    pearson = cov[0, 1]/jnp.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = jnp.sqrt(1 + pearson)
    ell_radius_y = jnp.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        alpha = alpha,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = jnp.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0,0]

    # calculating the stdandard deviation of y ...
    scale_y = jnp.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1,0]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)