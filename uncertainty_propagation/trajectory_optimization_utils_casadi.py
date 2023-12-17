import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import casadi as cd
import numpy as np

def step(x,u,dt):
    return x+u*dt


def dynamics_step( base_term, state_dot, dt ):
    next_state = base_term + state_dot * dt
#     print(f"next_state:{next_state}")
    return next_state

# extended unicycle
def dynamics_xdot(state, action):
    return cd.vcat(
        [
            state[3,0]*cd.cos(state[2,0]), state[3,0]*cd.sin(state[2,0]), action[1,0], action[0,0]
        ]
    )

# assume a single control input
# assume this is true dynamics
def dynamics_xdot_noisy(state, action):
    xdot = dynamics_xdot(state, action)
    error_square = 0.01 + 0.1 * xdot*xdot # /2  #never let it be 0!!!!
    cov = cd.diag( error_square[:,0] )
    xdot = xdot + xdot/2 #X_dot = X_dot + X_dot/6
    return xdot, cov

def get_mean( sigma_points, weights ):
    weighted_points = sigma_points * cd.repmat(weights,sigma_points.shape[0], 1 )
    mu = cd.sum2(weighted_points)
    return mu

def get_mean_cov(sigma_points, weights):
    
    # mean
    weighted_points = sigma_points * cd.repmat(weights,sigma_points.shape[0], 1 )
    mu = cd.sum2(weighted_points)
    
    # covariance
    centered_points = sigma_points - mu
    cov = cd.diag( cd.sum2( centered_points*centered_points * cd.repmat(weights,sigma_points.shape[0], 1 ) ) )
    return mu, cov

def get_ut_cov_root_diagonal(cov):
    offset = 0.000  # TODO: make sure not zero here
    root0 = cd.sqrt((offset+cov[0,0]))
    root1 = cd.sqrt((offset+cov[1,1]))
    root2 = cd.sqrt((offset+cov[2,2]))
    root3 = cd.sqrt((offset+cov[3,3]))
    # return cov
    root_term = cd.diag( [root0, root1, root2, root3] )
    return root_term

def get_mean_cov_skew_kurt_for_generation( sigma_points, weights ):
    # mean
    weighted_points = sigma_points * cd.repmat(weights,sigma_points.shape[0], 1 )
    mu = cd.sum2(weighted_points)
    centered_points = sigma_points - mu
    cov = cd.diag( cd.sum2( centered_points*centered_points * cd.repmat(weights,sigma_points.shape[0], 1 ) ) )
    
    skewness_temp = cd.sum2( centered_points**3 * cd.repmat(weights,sigma_points.shape[0], 1 ) ) 
    skewness = cd.vcat(
        [
            skewness_temp[0] / cov[0,0]**(3/2),
            skewness_temp[1] / cov[1,1]**(3/2),
            skewness_temp[2] / cov[2,2]**(3/2),
            skewness_temp[3] / cov[3,3]**(3/2)
        ]
    )
   
    kurt_temp = cd.sum2( centered_points**4 * cd.repmat(weights,sigma_points.shape[0], 1 ) )
    kurt = cd.vcat(
        [
            kurt_temp[0]/cov[0,0]**(4/2),
            kurt_temp[1]/cov[1,1]**(4/2),
            kurt_temp[2]/cov[2,2]**(4/2),
            kurt_temp[3]/cov[3,3]**(4/2)
        ]
    )
    
    return mu, cov, skewness, kurt

def generate_sigma_points_gaussian( mu, cov_root, base_term, factor ):
    n = mu.shape[0]     
    N = 2*n + 1 # total points

    alpha = 1.0
    beta = 0.0#2.0#2.0 # optimal for gaussian
    k = 1.0
    Lambda = alpha**2 * ( n+k ) - n
    
    points0 = base_term + mu
    weights0 = [1.0*Lambda/(n+Lambda)] 
    
    points0 = base_term + mu * factor
    points1 = base_term + (mu + cd.sqrt(n+Lambda) * cov_root) * factor
    points2 = base_term + (mu - cd.sqrt(n+Lambda) * cov_root) * factor
    new_points = cd.hcat([
        points0, points1, pionts2
    ])
    
    weights0 = [1.0*Lambda/(n+Lambda)]
    weights1 = 1.0/(n+Lambda)/2.0 * cd.SX(np.ones(1,n))
    weights2 = 1.0/(n+Lambda)/2.0 * cd.SX(np.ones(1,n))
    new_weights = cd.hcat([
        weights0, weights1, weights2
    ])
    new_weights_cov = weights
    

    return new_points, new_weights

def generate_sigma_points_gaussian_GenUT( mu, cov_root, skewness, kurt, base_term, factor ):
    n = mu.shape[0]     
    N = 2*n + 1 # total points
    
    u = 0.5 * ( - skewness + cd.sqrt( 4 * kurt - 3 * ( skewness )**2 ) )
    v = u + skewness

    # Weights
    w2 = (1.0 / v) / (u+v)
    w1 = (w2 * v) / u
    w0 = 1 - cd.sum()
    w0 = 1 - cd.sum1(w1) - cd.sum1(w2)
    
    # Points
    U = cd.diag(u)
    V = cd.diag(v)
    points0 = base_term + mu * factor
    points1 = base_term + (mu - cd.mtimes(cov_root, U)) * factor
    points2 = base_term + (mu + cd.mtimes(cov_root, V)) * factor
      
    # New sigma points
    new_points = cd.hcat( [points0, points1, points2] )
    new_weights = cd.hcat( [w0, w1, w2] )

    return new_points, new_weights

def sigma_point_expand(sigma_points, weights, control, dt):
   
    n, N = sigma_points.shape   

    mu, cov = dynamics_xdot_noisy(sigma_points[:,0], control)
    root_term = get_ut_cov_root_diagonal(cov) 
    new_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,0], dt )
    new_weights = temp_weights * weights[0,0]
        
    for i in range(1,N):
        mu, cov = dynamics_xdot_noisy(sigma_points[:,i].reshape(-1,1), control.reshape(-1,1))
        root_term = get_ut_cov_root_diagonal(cov)           
        temp_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,i].reshape(-1,1), dt )
        
        new_points = cd.hcat([ new_points, temp_points ])
        new_weights = cd.hcat([ new_weights, temp_weights * weights[0,i]  ])

    return new_points, new_weights

def sigma_point_compress( sigma_points, weights ):
    mu, cov = get_mean_cov( sigma_points, weights )
    cov_root_term = get_ut_cov_root_diagonal( cov )  
    base_term =  cd.MX.zeros(mu.shape)# 0*mu  
    return generate_sigma_points_gaussian( mu, cov_root_term, base_term, 1.0 )

def sigma_point_compress_GenUT( sigma_points, weights ):
    mu, cov, skewness, kurt = get_mean_cov_skew_kurt_for_generation( sigma_points, weights )
    # print(f"mu:{mu}, cov:{cov}, skewness:{skewness}, kurtosis:{kurt}")
    cov_root_term = get_ut_cov_root_diagonal( cov )  
    base_term =  cd.MX.zeros(mu.shape)
    return generate_sigma_points_gaussian_GenUT( mu, cov_root_term, skewness, kurt, base_term, 1.0 )

def foresee_propagate_GenUT( sigma_points, weights, action, dt ):
    
    expanded_sigma_points, expanded_weights = sigma_point_expand( sigma_points, weights, action, dt )
    compressed_sigma_points, compressed_weights = sigma_point_compress_GenUT(expanded_sigma_points, expanded_weights)
    return compressed_sigma_points, compressed_weights

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