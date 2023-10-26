import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import time
import math
#from jax import jit

# plot ellipse

def confidence_ellipse(mu, cov, ax, n_std=3.0, facecolor='none', **kwargs):
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

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0,0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1,0]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def dynamics_step( base_term, state_dot, dt ):
    next_state = base_term + state_dot * dt
#     print(f"next_state:{next_state}")
    return next_state

def dynamics_xdot(state, action = np.array([0])):
    return 20 * np.array([np.cos(state[0,0])+0.01, np.sin(state[1,0])+0.01]).reshape(-1,1)

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

# moments for data generation
# assume rows independent. therefore disginal eleemnts are 0. skewness and kurtosis only for 1-D data
def get_mean_cov_skew_kurt_for_generation( sigma_points, weights ):
    # mean
    weighted_points = sigma_points * weights[0]
    mu = np.sum( weighted_points, 1 ).reshape(-1,1)    
    centered_points = sigma_points - mu    
    # covariance
#     weighted_centered_points = centered_points * weights[0] 
#     cov = weighted_centered_points @ centered_points.T   
    cov = np.diag(np.sum(centered_points**2 * weights[0], axis=1))
    # print(f"COV: {cov}")
    # Skewness times cov_root^-3
    skewness = np.sum(centered_points**3 * weights[0], axis=1) #/ cov[0,0]**(3/2) # for scipy    
    skewness[0] = skewness[0] / cov[0,0]**(3/2)
    skewness[1] = skewness[1] / cov[1,1]**(3/2)
    # kurtosis times cov_root^-4
    kurt = np.sum(centered_points**4 * weights[0], axis=1)# / cov[0,0]**(4/2)  # -3 # -3 for scipy
    kurt[0] = kurt[0]/cov[0,0]**(4/2)
    kurt[1] = kurt[1]/cov[1,1]**(4/2)
    return mu, cov, skewness.reshape(-1,1), kurt.reshape(-1,1)

# actual moments
def get_mean_cov_skew_kurt( sigma_points, weights ):
    # mean
    weighted_points = sigma_points * weights[0]
    mu = np.sum( weighted_points, 1 ).reshape(-1,1)    
    centered_points = sigma_points - mu    
    # covariance
#     weighted_centered_points = centered_points * weights[0] 
#     cov = weighted_centered_points @ centered_points.T    
    cov = np.diag(np.sum(centered_points**2 * weights[0], axis=1))
    # Skewness times cov_root^-3
    skewness = np.sum(centered_points**3 * weights[0], axis=1) #/ cov[0,0]**(3/2) # for scipy    
    # kurtosis times cov_root^-4
    kurt = np.sum(centered_points**4 * weights[0], axis=1) #/ cov[0,0]**(4/2)-3 # for scipy
    return mu, cov, skewness.reshape(-1,1), kurt.reshape(-1,1)

#@jit
def get_ut_cov_root_diagonal(cov):
    offset = 0.000  # TODO: make sure not zero here
    root0 = np.sqrt((offset+cov[0,0]))
    root1 = np.sqrt((offset+cov[1,1]))
    # return cov
    root_term = np.diag( np.array([root0, root1]) )
    return root_term

def pilco_propagate(mean, cov):
    mu, cov = dynamics_xdot_noisy(mean)
    return dynamics_step( mean, mu, dt ), cov * dt**2

def mc_propagate(points):
    new_points = np.copy(points)
    for i in range(points.shape[1]):
        mu, cov = dynamics_xdot_noisy(points[:,i].reshape(-1,1))
        sample = np.array([  np.random.normal(mu[0,0], np.sqrt(cov[0,0])), np.random.normal(mu[1,0], np.sqrt(cov[1,1]))  ]).reshape(-1,1) 
        points[:,i] = dynamics_step(points[:,i].reshape(-1,1), sample, dt)[:,0]
    return points

#@jit
def initialize_sigma_points(X):
    # return 2N + 1 points
    n = X.shape[0]
    num_points = 2*n + 1
    sigma_points = np.repeat( X, num_points, axis=1 )
    weights = np.ones((1,num_points)) * 1.0/( num_points )
    return sigma_points, weights

def generate_sigma_points_gaussian( mu, cov_root, base_term, factor ):
    n = mu.shape[0]     
    N = 2*n + 1 # total points

    # k = 1.0#0.5 # n-3 # 0.5**
    
    alpha = 1.0
    beta = 0.0#2.0#2.0 # optimal for gaussian
    k = 1.0
    Lambda = alpha**2 * ( n+k ) - n
    new_points = dynamics_step(base_term, mu, factor) # new_points = base_term + factor * mu
    new_weights = np.array([[1.0*Lambda/(n+Lambda)]])    
    new_weights_cov = np.array([[ 1.0*Lambda/(n+Lambda) + 1 - alpha**2 + beta]])
    for i in range(n):
#         print(f"new_points:{new_points}")
        new_points = np.append( new_points, dynamics_step(base_term, (mu - np.sqrt(n+Lambda) * cov_root[:,i].reshape(-1,1)), factor) , axis = 1 )
        new_points = np.append( new_points, dynamics_step(base_term, (mu + np.sqrt(n+Lambda) * cov_root[:,i].reshape(-1,1)), factor) , axis = 1 )
        new_weights = np.append( new_weights, np.array([[1.0/(n+Lambda)/2.0]]), axis = 1 )
        new_weights = np.append( new_weights, np.array([[1.0/(n+Lambda)/2.0]]), axis = 1 )
        new_weights_cov = np.append( new_weights_cov, np.array([[1.0/(n+Lambda)/2.0]]), axis = 1 )
        new_weights_cov = np.append( new_weights_cov, np.array([[1.0/(n+Lambda)/2.0]]), axis = 1 )
    # print(f"weights1: {new_weights}")
    # print(f"weights2: {new_weights_cov}")
    return new_points, new_weights

# @jit
def generate_sigma_points_gaussian_GenUT( mu, cov_root, skewness, kurt, base_term, factor ):
    n = mu.shape[0]     
    N = 2*n + 1 # total points
    
#     u = 0.5#k
    # print(f"GENERATE mu: {mu}, cov_root:{cov_root}, skew:{skewness}, kurt:{kurt}")
#     cov_3 = np.diag(np.reciprocal(np.diagonal(cov_root**3)))
#     cov_4 = np.diag(np.reciprocal(np.diagonal(cov_root**4)))
    u = 0.5 * ( - skewness + np.sqrt( 4 * kurt - 3 * ( skewness )**2 ) )
    v = u + skewness
#     v = u + cov_3 @ skewness
#     print(f"cov_root:{cov_root}, \n 3: {cov_3}, \n 4: {cov_4}")
    # print(f"u:{u.T},\n v:{v.T},\n term:{4 * kurt - 3 * ( skewness )**2} ")
    
    w2 = (1.0 / v) / (u+v)
    w1 = (w2 * v) / u
    w0 = 1 - np.sum(w1) - np.sum(w2)
    # print(f"w0:{w0},\n w1:{w1},\n w2:{w2}")
    
#     k = 0.5 # n-3 # 0.5**
    new_points = dynamics_step(base_term, mu, factor) # new_points = base_term + factor * mu
    new_weights = np.array([[w0]])
    for i in range(n):
        new_points = np.append( new_points, dynamics_step(base_term, (mu - u[i,0]*cov_root[:,i].reshape(-1,1)), factor) , axis = 1 )
        new_points = np.append( new_points, dynamics_step(base_term, (mu + v[i,0]*cov_root[:,i].reshape(-1,1)), factor) , axis = 1 )
        new_weights = np.append( new_weights, np.array([[w1[i,0]]]), axis = 1 )
        new_weights = np.append( new_weights, np.array([[w2[i,0]]]), axis = 1 )
    return new_points, new_weights

# @jit
def sigma_point_expand(sigma_points, weights, control):
   
    n, N = sigma_points.shape   
    # dt_outer = 0  
    #TODO  
    mu, cov = dynamics_xdot_noisy(sigma_points[:,0].reshape(-1,1), control.reshape(-1,1))
    root_term = get_ut_cov_root_diagonal(cov) 
    temp_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,0].reshape(-1,1), dt )
    new_points = np.copy( temp_points )
    new_weights = ( np.copy( temp_weights ) * weights[0,0]).reshape(1,-1)
        
    for i in range(1,N):
        mu, cov = dynamics_xdot_noisy(sigma_points[:,i].reshape(-1,1), control.reshape(-1,1))
        root_term = get_ut_cov_root_diagonal(cov)           
        temp_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,i].reshape(-1,1), dt )
        new_points = np.append(new_points, temp_points, axis=1 )
        new_weights = np.append( new_weights, (temp_weights * weights[0,i]).reshape(1,-1) , axis=1 )

    return new_points, new_weights

#@jit
def sigma_point_compress( sigma_points, weights ):
    mu, cov = get_mean_cov( sigma_points, weights )
    cov_root_term = get_ut_cov_root_diagonal( cov )  
    base_term = np.zeros((mu.shape))
    return generate_sigma_points_gaussian( mu, cov_root_term, base_term, np.array([1.0]) )

#@jit
def sigma_point_compress_GenUT( sigma_points, weights ):
    mu, cov, skewness, kurt = get_mean_cov_skew_kurt_for_generation( sigma_points, weights )
    # print(f"mu:{mu}, cov:{cov}, skewness:{skewness}, kurtosis:{kurt}")
    cov_root_term = get_ut_cov_root_diagonal( cov )  
    base_term = np.zeros((mu.shape))
    return generate_sigma_points_gaussian_GenUT( mu, cov_root_term, skewness, kurt, base_term, np.array([1.0]) )

# @jit
def foresee_propagate( sigma_points, weights, action = np.array([0]), expanded_only=False ):
    
    #Expansion Layer
    expanded_sigma_points, expanded_weights = sigma_point_expand( sigma_points, weights, action )
    if expanded_only == False:
    # Compression layer
#     compressed_sigma_points, compressed_weights = sigma_point_compress_GenUT(expanded_sigma_points, expanded_weights)
        compressed_sigma_points, compressed_weights = sigma_point_compress(expanded_sigma_points, expanded_weights)
        return compressed_sigma_points, compressed_weights, expanded_sigma_points, expanded_weights
    else:
        return expanded_sigma_points, expanded_weights, expanded_sigma_points, expanded_weights

# @jit
def foresee_propagate_GenUT( sigma_points, weights, action = np.array([0]), expanded_only=False ):
    
    #Expansion Layer
    expanded_sigma_points, expanded_weights = sigma_point_expand( sigma_points, weights, action )
    
    if expanded_only == False:
        
        # Compression layer
        compressed_sigma_points, compressed_weights = sigma_point_compress_GenUT(expanded_sigma_points, expanded_weights)
        return compressed_sigma_points, compressed_weights, expanded_sigma_points, expanded_weights
    else:
        return expanded_sigma_points, expanded_weights, expanded_sigma_points, expanded_weights

# horizon = 3
# dt = 0.05
# initial_state_mean = np.array([0.0,0.0]).reshape(-1,1)
# initial_state_cov = np.zeros((2,2))
# sigma_points_init, weights_init = initialize_sigma_points(initial_state_mean)

# # PILCO init
# pilco_mu, pilco_cov = np.copy(initial_state_mean), np.copy(initial_state_cov)

# # Monte Carlo init
# num_particles = 5000
# mc_particles = np.repeat(initial_state_mean, num_particles, axis=1)

# # FORESEE init
# sigma_points = np.copy(sigma_points_init)
# weights1 = np.copy(weights_init)
# weights2 = np.copy(weights_init)

# # Initialize data

# for t in range(horizon):
    
#     # PILCO Update
#     pilco_mu, pilco_cov = pilco_propagate(pilco_mu, pilco_cov)

#     # Monte Carlo Update
#     mc_particles = mc_propagate(mc_particles)    
    
#     # FORESEE update
#     sigma_points, weights1, weights2, full_sigma_points, full_weights1, full_weights2 = foresee_propagate(sigma_points, weights1, weights2)
    
# Visualize
# plt.ion()
# fig = plt.figure()
# ax = plt.axes()
# ax.set_xlabel("x(1)")
# ax.set_ylabel("x(2)")


# plot_mc = plt.scatter(mc_particles[0,:], mc_particles[1,:])
# plot_foresee = plt.scatter( sigma_points[0,:], sigma_points[1,:], c = 'g', alpha=1.0 )
# plot_foresee2 = plt.scatter( full_sigma_points[0,:], full_sigma_points[1,:], c = 'k', alpha=1.0 )
# plot_pilco = confidence_ellipse( pilco_mu, pilco_cov, ax, n_std=3.0, edgecolor = 'red' )

# # compute means
# print(f" MC mean:{ np.mean(mc_particles, axis=1) } ")
# print(f" UT mean:{ get_mean(sigma_points, weights1  )[:,0] } ")
# print(f" UT mean full:{ get_mean(full_sigma_points, full_weights1 )[:,0] } ")
# print(f" PILCO mean: {pilco_mu[:,0]} ")

# # Compute Cov
# _, mc_cov = get_mean_cov(mc_particles, np.ones((1,mc_particles.shape[1]))/mc_particles.shape[1], np.ones((1,mc_particles.shape[1]))/mc_particles.shape[1])
# _, foresee_cov = get_mean_cov( sigma_points, weights1, weights2 )
# _, foresee_complete_cov = get_mean_cov( full_sigma_points, full_weights1, full_weights2 )

# print(f" MC cov : \n {mc_cov}")
# print(f" foresee cov : \n {foresee_cov}")
# print(f" foresee_complete cov : \n {foresee_complete_cov}")
# print(f" pilco cov : \n {pilco_cov}")

# plt.show()   

# horizon = 10
# dt = 0.05
# initial_state_mean = np.array([0.0,0.0]).reshape(-1,1)
# initial_state_cov = np.zeros((2,2))
# sigma_points_init, weights_init = initialize_sigma_points(initial_state_mean)

# # PILCO init
# pilco_mu, pilco_cov = np.copy(initial_state_mean), np.copy(initial_state_cov)

# # Monte Carlo init
# num_particles = 20000
# mc_particles = np.repeat(initial_state_mean, num_particles, axis=1)

# # FORESEE init
# sigma_points = np.copy(sigma_points_init)
# weights = np.copy(weights_init)
# sigma_points_GenUT = np.copy(sigma_points_init)
# weights_GenUT = np.copy(weights_init)

# # Initialize data

# for t in range(horizon):
    
# #     # PILCO Update
# #     pilco_mu, pilco_cov = pilco_propagate(pilco_mu, pilco_cov)

#     # Monte Carlo Update
#     mc_particles = mc_propagate(mc_particles)    
    
#     # FORESEE update
#     sigma_points, weights, full_sigma_points, full_weights = foresee_propagate(sigma_points, weights)
    
#     # FORESEE GenUT update
#     sigma_points_GenUT, weights_GenUT, full_sigma_points_GenUT, full_weights_GenUT = foresee_propagate_GenUT(sigma_points_GenUT, weights_GenUT)
    
# # Visualize
# # plt.ion()
# fig = plt.figure()
# ax = plt.axes()
# ax.set_xlabel("X")
# ax.set_ylabel("Y")

# ax.set_facecolor("grey")
# plot_mc = plt.scatter(mc_particles[0,:], mc_particles[1,:])
# plot_foresee = plt.scatter( sigma_points[0,:], sigma_points[1,:], c = 'y' )
# # plot_foresee2 = plt.scatter( full_sigma_points[0,:], full_sigma_points[1,:], c = 'k' )
# plot_foresee_GenUT = plt.scatter( sigma_points_GenUT[0,:], sigma_points_GenUT[1,:], edgecolors = 'r', facecolor='none' )
# # plot_foresee2_GenUT = plt.scatter( full_sigma_points_GenUT[0,:], full_sigma_points_GenUT[1,:], edgecolors = 'y', facecolor='none' )
# # plot_pilco = confidence_ellipse( pilco_mu, pilco_cov, ax, n_std=3.0, edgecolor = 'red' )

# # compute means
# print(f" MC mean:{ np.mean(mc_particles, axis=1) } ")
# print(f" UT mean:{ get_mean(sigma_points, weights  )[:,0] } ")
# print(f" UT mean GenUT:{ get_mean(sigma_points_GenUT, weights_GenUT  )[:,0] } ")
# print(f" UT mean full:{ get_mean(full_sigma_points, full_weights  )[:,0] } ")
# print(f" UT mean full GenUT:{ get_mean(full_sigma_points_GenUT, full_weights_GenUT  )[:,0] } ")
# # print(f" PILCO mean: {pilco_mu[:,0]} ")

# # Compute Cov
# _, mc_cov = get_mean_cov(mc_particles, np.ones((1,mc_particles.shape[1]))/mc_particles.shape[1])
# _, foresee_cov = get_mean_cov( sigma_points, weights )
# _, foresee_complete_cov = get_mean_cov( full_sigma_points, full_weights )
# _, foresee_cov_GenUT = get_mean_cov( sigma_points_GenUT, weights_GenUT )

# print(f" MC cov : \n {mc_cov}")
# print(f" foresee cov : \n {foresee_cov}")
# print(f" foresee_complete cov : \n {foresee_complete_cov}")
# print(f" foresee cov GenUT : \n {foresee_cov_GenUT}")
# # print(f" pilco cov : \n {pilco_cov}")

# plt.show()

dt = 0.05
def compare_predictions(ax, horizons = 3, num_particles = 5000, expanded_only = False):
    horizon = 3
    
    initial_state_mean = np.array([0.0,0.0]).reshape(-1,1)
    initial_state_cov = np.zeros((2,2))
    sigma_points_init, weights_init = initialize_sigma_points(initial_state_mean)

    # PILCO init
    pilco_mu, pilco_cov = np.copy(initial_state_mean), np.copy(initial_state_cov)

    # Monte Carlo init
    mc_particles = np.repeat(initial_state_mean, num_particles, axis=1)

    # FORESEE init
    sigma_points = np.copy(sigma_points_init)
    weights = np.copy(weights_init)
    
    sigma_points_GenUT = np.copy(sigma_points_init)
    weights_GenUT = np.copy(weights_init)

    # Initialize data

    for t in range(horizons):
        # PILCO Update
        pilco_mu, pilco_cov = pilco_propagate(pilco_mu, pilco_cov)

    t0 = time.time()
    for t in range(horizons):
        # Monte Carlo Update
        mc_particles = mc_propagate(mc_particles)    
    mc_time = time.time() - t0
    print(f"mc time :{mc_time} \n")
    
    t0 = time.time()
    for t in range(horizons):
        # FORESEE update
        sigma_points, weights, full_sigma_points, full_weights = foresee_propagate(sigma_points, weights, expanded_only = expanded_only)
    foresee_time = time.time() - t0
    print(f"horizon: {t}, foresee_time:{foresee_time} \n")
    
    if not expanded_only:
        t0 = time.time()
        for t in range(horizons):
            # FORESEE update
            sigma_points_GenUT, weights_GenUT, full_sigma_points_GenUT, full_weights_GenUT = foresee_propagate_GenUT(sigma_points_GenUT, weights_GenUT, expanded_only = expanded_only)
        foresee_time_GenUT = time.time() - t0
        print(f"genUT foresee_time:{foresee_time_GenUT} \n")
    
    ax.set_facecolor("grey")
    
    # ax.set_ylim([-6.5, 11.5]) #17.5
    # ax.set_ylim([-7.0, 10.0]) #16.5
    
    plot_mc = ax.scatter(mc_particles[0,:], mc_particles[1,:], c = 'b')
    plot_foresee = ax.scatter( sigma_points[0,:], sigma_points[1,:], c = 'yellow', alpha=1.0 )
    if not expanded_only:
        plot_foresee_GenUT = ax.scatter( sigma_points_GenUT[0,:], sigma_points_GenUT[1,:], edgecolors = 'salmon', facecolor='none', alpha=1.0, s = 50, linewidth=3 )
    plot_foresee2 = ax.scatter( full_sigma_points[0,:], full_sigma_points[1,:], c = 'k', alpha=1.0 )
    plot_pilco = confidence_ellipse( pilco_mu, pilco_cov, ax, n_std=2.0, edgecolor = 'red' )

    # compute means
    # print(f" MC mean:{ np.mean(mc_particles, axis=1) } ")
    # print(f" UT mean:{ get_mean(sigma_points, weights1  )[:,0] } ")
    # print(f" UT mean full:{ get_mean(full_sigma_points, full_weights1 )[:,0] } ")
    # print(f" PILCO mean: {pilco_mu[:,0]} ")

    # Compute Cov
    mc_mu, mc_cov = get_mean_cov(mc_particles, np.ones((1,mc_particles.shape[1]))/mc_particles.shape[1])
    foresee_mu, foresee_cov = get_mean_cov( sigma_points, weights )
    foresee_genut_mu, foresee_genut_cov = get_mean_cov( sigma_points_GenUT, weights_GenUT )
    # _, foresee_complete_cov = get_mean_cov( full_sigma_points, full_weights1, full_weights2 )

    # print(f" MC cov : \n {mc_cov}")
    # print(f" foresee cov : \n {foresee_cov}")
    # # print(f" foresee_complete cov : \n {foresee_complete_cov}")
    # print(f" pilco cov : \n {pilco_cov}")
    
    return mc_mu, foresee_mu, pilco_mu, mc_cov, foresee_cov, pilco_cov, mc_time, foresee_time#, foresee_genut_mu, foresee_genut_cov

# fig = plt.figure()
# ax = plt.axes()
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# compare_predictions( ax, horizons=10, num_particles=500, expanded_only=False )
# plt.show()
if 0:
    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_xlabel("x(1)")
    ax1.set_ylabel("x(2)")
    mc_mu1, foresee_mu1, pilco_mu1, mc_cov1, foresee_cov1, pilco_cov1, time1_1, time1_2 = compare_predictions( ax1, horizons = 1, num_particles = 500, expanded_only = False )
    fig1.savefig("fig1.png")

    fig2, ax2 = plt.subplots(1, 1)
    ax2.set_xlabel("x(1)")
    ax2.set_ylabel("x(2)")
    mc_mu2, foresee_mu2, pilco_mu2, mc_cov2, foresee_cov2, pilco_cov2, time2_1, time2_2 = compare_predictions( ax2, horizons = 1, num_particles = 500, expanded_only = True )
    fig2.savefig("fig2.png")

    fig3, ax3 = plt.subplots(1, 1)
    ax3.set_xlabel("x(1)")
    ax3.set_ylabel("x(2)")
    mc_mu3, foresee_mu3, pilco_mu3, mc_cov3, foresee_cov3, pilco_cov3, time3_1, time3_2 = compare_predictions( ax3, horizons = 1, num_particles = 5000, expanded_only = False )
    fig3.savefig("fig3.png")

    fig4, ax4 = plt.subplots(1, 1)
    ax4.set_xlabel("x(1)")
    ax4.set_ylabel("x(2)")
    mc_mu4, foresee_mu4, pilco_mu4, mc_cov4, foresee_cov4, pilco_cov4, time4_1, time4_2 = compare_predictions( ax4, horizons = 1, num_particles = 5000, expanded_only = True )
    fig4.savefig("fig4.png")

    fig5, ax5 = plt.subplots(1, 1)
    ax5.set_xlabel("x(1)")
    ax5.set_ylabel("x(2)")
    mc_mu5, foresee_mu5, pilco_mu5, mc_cov5, foresee_cov5, pilco_cov5, time5_1, time5_2 = compare_predictions( ax5, horizons = 1, num_particles = 50000, expanded_only = False )
    fig5.savefig("fig5.png")

    fig6, ax6 = plt.subplots(1, 1)
    ax6.set_xlabel("x(1)")
    ax6.set_ylabel("x(2)")
    mc_mu6, foresee_mu6, pilco_mu6, mc_cov6, foresee_cov6, pilco_cov6, time6_1, time6_2 = compare_predictions( ax6, horizons = 1, num_particles = 50000, expanded_only = True )
    fig6.savefig("fig6.png")
plt.show()
# exit()
###########################################
if 0:
    fig7, ax7 = plt.subplots(1, 1)
    ax7.set_xlabel("x(1)")
    ax7.set_ylabel("x(2)")
    mc_mu7, foresee_mu7, pilco_mu7, mc_cov7, foresee_cov7, pilco_cov7, time7_1, time7_2 = compare_predictions( ax7, horizons = 2, num_particles = 500, expanded_only = False )
    fig7.savefig("fig7.png")
    # plt.show()

    fig8, ax8 = plt.subplots(1, 1)
    ax8.set_xlabel("x(1)")
    ax8.set_ylabel("x(2)")
    mc_mu8, foresee_mu8, pilco_mu8, mc_cov8, foresee_cov8, pilco_cov8, time8_1, time8_2 = compare_predictions( ax8, horizons = 2, num_particles = 500, expanded_only = True )
    fig8.savefig("fig8.png")

    fig9, ax9 = plt.subplots(1, 1)
    ax9.set_xlabel("x(1)")
    ax9.set_ylabel("x(2)")
    mc_mu9, foresee_mu9, pilco_mu9, mc_cov9, foresee_cov9, pilco_cov9, time9_1, time9_2 = compare_predictions( ax9, horizons = 2, num_particles = 5000, expanded_only = False )
    fig9.savefig("fig9.png")

    fig10, ax10 = plt.subplots(1, 1)
    ax10.set_xlabel("x(1)")
    ax10.set_ylabel("x(2)")
    mc_mu10, foresee_mu10, pilco_mu10, mc_cov10, foresee_cov10, pilco_cov10, time10_1, time10_2 = compare_predictions( ax10, horizons = 2, num_particles = 5000, expanded_only = True )
    fig10.savefig("fig10.png")

    fig11, ax11 = plt.subplots(1, 1)
    ax11.set_xlabel("x(1)")
    ax11.set_ylabel("x(2)")
    mc_mu11, foresee_mu11, pilco_mu11, mc_cov11, foresee_cov11, pilco_cov11, time11_1, time11_2 = compare_predictions( ax11, horizons = 2, num_particles = 50000, expanded_only = False )
    fig11.savefig("fig11.png")

    fig12, ax12 = plt.subplots(1, 1)
    ax12.set_xlabel("x(1)")
    ax12.set_ylabel("x(2)")
    mc_mu12, foresee_mu12, pilco_mu12, mc_cov12, foresee_cov12, pilco_cov12, time12_1, time12_2 = compare_predictions( ax12, horizons = 2, num_particles = 50000, expanded_only = True )
    fig12.savefig("fig12.png")

# plt.show()
#########################

if 0:
    fig13, ax13 = plt.subplots(1, 1)
    ax13.set_xlabel("x(1)")
    ax13.set_ylabel("x(2)")
    mc_mu13, foresee_mu13, pilco_mu13, mc_cov13, foresee_cov13, pilco_cov13, time13_1, time13_2 = compare_predictions( ax13, horizons = 3, num_particles = 500, expanded_only = False )
    fig13.savefig("fig13.png")

    fig23, ax23 = plt.subplots(1, 1)
    ax23.set_xlabel("x(1)")
    ax23.set_ylabel("x(2)")
    mc_mu23, foresee_mu23, pilco_mu23, mc_cov23, foresee_cov23, pilco_cov23, time23_1, time23_2 = compare_predictions( ax23, horizons = 3, num_particles = 500, expanded_only = True )
    fig23.savefig("fig23.png")

    fig33, ax33 = plt.subplots(1, 1)
    ax33.set_xlabel("x(1)")
    ax33.set_ylabel("x(2)")
    mc_mu33, foresee_mu33, pilco_mu33, mc_cov33, foresee_cov33, pilco_cov33, time33_1, time33_2 = compare_predictions( ax33, horizons = 3, num_particles = 5000, expanded_only = False )
    fig33.savefig("fig33.png")

    fig43, ax43 = plt.subplots(1, 1)
    ax43.set_xlabel("x(1)")
    ax43.set_ylabel("x(2)")
    mc_mu43, foresee_mu43, pilco_mu43, mc_cov43, foresee_cov43, pilco_cov43, time43_1, time43_2 = compare_predictions( ax43, horizons = 3, num_particles = 5000, expanded_only = True )
    fig43.savefig("fig43.png")

    fig53, ax53 = plt.subplots(1, 1)
    ax53.set_xlabel("x(1)")
    ax53.set_ylabel("x(2)")
    mc_mu53, foresee_mu53, pilco_mu53, mc_cov53, foresee_cov53, pilco_cov53, time53_1, time53_2 = compare_predictions( ax53, horizons = 3, num_particles = 50000, expanded_only = False )
    fig53.savefig("fig53.png")

    fig63, ax63 = plt.subplots(1, 1)
    ax63.set_xlabel("x(1)")
    ax63.set_ylabel("x(2)")
    mc_mu63, foresee_mu63, pilco_mu63, mc_cov63, foresee_cov63, pilco_cov63, time63_1, time63_2 = compare_predictions( ax63, horizons = 3, num_particles = 50000, expanded_only = True )
    fig63.savefig("fig63.png")

    # plt.show()
# exit()

######################

if 1:
    # fig14, ax14 = plt.subplots(1, 1)
    # ax14.set_xlabel("x(1)")
    # ax14.set_ylabel("x(2)")
    # mc_mu14, foresee_mu14, pilco_mu14, mc_cov14, foresee_cov14, pilco_cov14, time14_1, time14_2 = compare_predictions( ax14, horizons = 10, num_particles = 500, expanded_only = False )
    # fig14.savefig("fig14.png")

    # fig24, ax24 = plt.subplots(1, 1)
    # ax24.set_xlabel("x(1)")
    # ax24.set_ylabel("x(2)")
    # mc_mu24, foresee_mu24, pilco_mu24, mc_cov24, foresee_cov24, pilco_cov24, _ , _  = compare_predictions( ax24, horizons = 10, num_particles = 500, expanded_only = True )
    # fig24.savefig("fig24.png")

    # fig34, ax34 = plt.subplots(1, 1)
    # ax34.set_xlabel("x(1)")
    # ax34.set_ylabel("x(2)")
    # mc_mu34, foresee_mu34, pilco_mu34, mc_cov34, foresee_cov34, pilco_cov34 , _ , _ = compare_predictions( ax34, horizons = 10, num_particles = 5000, expanded_only = False )
    # fig34.savefig("fig34.png")

    # fig44, ax44 = plt.subplots(1, 1)
    # ax44.set_xlabel("x(1)")
    # ax44.set_ylabel("x(2)")
    # mc_mu44, foresee_mu44, pilco_mu44, mc_cov44, foresee_cov44, pilco_cov44, _ , _ = compare_predictions( ax44, horizons = 10, num_particles = 5000, expanded_only = True )
    # fig44.savefig("fig44.png")

    # fig54, ax54 = plt.subplots(1, 1)
    # ax54.set_xlabel("x(1)")
    # ax54.set_ylabel("x(2)")
    # mc_mu54, foresee_mu54, pilco_mu54, mc_cov54, foresee_cov54, pilco_cov54, _ , _ = compare_predictions( ax54, horizons = 6, num_particles = 50000, expanded_only = False )
    # fig54.savefig("fig54.png")

    fig64, ax64 = plt.subplots(1, 1)
    ax64.set_xlabel("x(1)")
    ax64.set_ylabel("x(2)")
    mc_mu64, foresee_mu64, pilco_mu64, mc_cov64, foresee_cov64, pilco_cov64, _, _ = compare_predictions( ax64, horizons = 6, num_particles = 50000, expanded_only = True )
    # fig64.savefig("fig64.png")
    
if 0:

    fig55, ax55 = plt.subplots(1, 1)
    ax55.set_xlabel("x(1)")
    ax55.set_ylabel("x(2)")
    mc_mu55, foresee_mu55, pilco_mu55, mc_cov55, foresee_cov55, pilco_cov55, _ , _ = compare_predictions( ax55, horizons = 10, num_particles = 50000, expanded_only = False )
    fig55.savefig("fig55.png")
    
if 0:

    fig56, ax56 = plt.subplots(1, 1)
    ax56.set_xlabel("x(1)")
    ax56.set_ylabel("x(2)")
    mc_mu56, foresee_mu56, pilco_mu56, mc_cov56, foresee_cov56, pilco_cov56, _ , _ = compare_predictions( ax56, horizons = 20, num_particles = 50000, expanded_only = False )
    fig56.savefig("fig56.png")

# plt.show()

# plot means:
# horizon on x axis


fig_1, ax_1 = plt.subplots(1,1)
# MC 500
ax_1.plot( [1, 2, 3],  [ mc_mu1[0,0], mc_mu7[0,0], mc_mu13[0,0] ], 'b-', label='MC 500' )

# MC 5000
ax_1.plot( [1, 2, 3], [ mc_mu3[0,0], mc_mu9[0,0], mc_mu33[0,0] ], 'b--', label='MC 5000' )

# MC 50000
ax_1.plot( [1, 2, 3], [ mc_mu6[0,0], mc_mu11[0,0], mc_mu63[0,0] ], 'b-*', label='MC 50000' )

# foresee_compressed
ax_1.plot( [1, 2, 3], [ foresee_mu1[0,0], foresee_mu7[0,0], foresee_mu13[0,0] ], 'k--', label='FORESEE Expansion Only' )

# foresee_expanded
ax_1.plot( [1, 2, 3], [ foresee_mu2[0,0], foresee_mu8[0,0], foresee_mu23[0,0] ], 'k-*', label='FORESEE Expansion - Compression' )

# PILCO
ax_1.plot( [1, 2, 3], [ pilco_mu1[0,0], pilco_mu7[0,0], pilco_mu13[0,0] ], 'r--', label='PILCO' )
ax_1.legend()
ax_1.set_title('Predicted Mean')
ax_1.set_xlabel('Prediction Horizon')
# ax_1.set_ylabel('Predicted Mean')
new_list = range(math.floor(1), math.ceil(3)+1)
plt.xticks(new_list)
# ax_1.xaxis.set_major_locator(mticker.MultipleLocator(1))
# plt.show()
fig_1.savefig("predicted_mean_case1.png")

fig_2, ax_2 = plt.subplots(1,1)
# MC 500
ax_2.plot( [1, 2, 3],  [ time1_1, time7_1, time13_1], 'b-', label='MC 500' )

# MC 5000
ax_2.plot( [1, 2, 3], [ time3_1, time9_1, time33_1 ], 'b--', label='MC 5000' )

# MC 50000
ax_2.plot( [1, 2, 3], [ time6_1, time11_1, time63_1 ], 'b-*', label='MC 50000' )

# foresee_compressed
ax_2.plot( [1, 2, 3], [ time1_2, time7_2, time13_2 ], 'k--', label='FORESEE Expansion Only' )

# foresee_expanded
ax_2.plot( [1, 2, 3], [ time2_2, time8_2, time23_2 ], 'k-*', label='FORESEE Expansion - Compression' )
ax_2.set_title('Implementation Time (s)')
ax_2.set_xlabel('Prediction Horizon')
ax_2.legend()
# PILCO
# ax_2.plot( [1, 2, 3], [ pilco_mu1, pilco_mu7, pilco_mu13 ], 'r--' )

new_list = range(math.floor(1), math.ceil(3)+1)
plt.xticks(new_list)
fig_2.savefig("prediction_time_case1.png")
# ax_1.xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.show()

# Visualize
# plt.ion()
# fig = plt.figure()
# ax = plt.axes()
# ax.set_xlabel("x(1)")
# ax.set_ylabel("x(2)")


# plot_mc = plt.scatter(mc_particles[0,:], mc_particles[1,:])
# plot_foresee = plt.scatter( sigma_points[0,:], sigma_points[1,:], c = 'g', alpha=1.0 )
# plot_foresee2 = plt.scatter( full_sigma_points[0,:], full_sigma_points[1,:], c = 'k', alpha=1.0 )
# plot_pilco = confidence_ellipse( pilco_mu, pilco_cov, ax, n_std=3.0, edgecolor = 'red' )

# # compute means
# print(f" MC mean:{ np.mean(mc_particles, axis=1) } ")
# print(f" UT mean:{ get_mean(sigma_points, weights1  )[:,0] } ")
# print(f" UT mean full:{ get_mean(full_sigma_points, full_weights1 )[:,0] } ")
# print(f" PILCO mean: {pilco_mu[:,0]} ")

# # Compute Cov
# _, mc_cov = get_mean_cov(mc_particles, np.ones((1,mc_particles.shape[1]))/mc_particles.shape[1], np.ones((1,mc_particles.shape[1]))/mc_particles.shape[1])
# _, foresee_cov = get_mean_cov( sigma_points, weights1, weights2 )
# _, foresee_complete_cov = get_mean_cov( full_sigma_points, full_weights1, full_weights2 )

# print(f" MC cov : \n {mc_cov}")
# print(f" foresee cov : \n {foresee_cov}")
# print(f" foresee_complete cov : \n {foresee_complete_cov}")
# print(f" pilco cov : \n {pilco_cov}")

# plt.show()

# Recorded data histogram

#500

plt.var()

# import matplotlib.pyplot as plt

# f, (ax2, ax) = plt.subplots(2, 1, sharex=True)

# bar_labels1 = ['MC 500', 'MC 5000', 'MC 50000', 'EC Layer - UT', 'EC Layer - GenUT', 'E layer - UT H1', 'E layer - UT H1', 'E layer - UT H2', 'E layer - UT H3', 'E layer - UT H4', 'E layer - UT H5' ]
# bar_counts1 = [0.007509589, 0.073462725, 0.732954264, 0.000451763,0.000496546, 0.00041794, 0.001773357, 0.00844645, 0.04363536, 0.2262163162, 1.7395920   ]

# bar_labels2 = ['MC 500', 'MC 5000', 'MC 50000', 'EC Layer - UT', 'EC Layer - GenUT', 'E layer - UT H1', 'E layer - UT H1', 'E layer - UT H2', 'E layer - UT H3', 'E layer - UT H4', 'E layer - UT H5' ]
# bar_counts2 = [0.007509589, 0.073462725, 0.732954264, 0.000451763,0.000496546, 0.00041794, 0.001773357, 0.00844645, 0.04363536, 0.2262163162, 1.7395920   ]

# ax.bar( bar_labels1, bar_counts1 )
# ax2.bar( bar_labels2, bar_counts2 )

# ax.spines['top'].set_visible(False)
# ax2.spines['bottom'].set_visible(False)

# ax.set_ylim(0.0,0.25 )
# ax2.set_ylim(0.6, 1.9)

# ax2.xaxis.tick_top()
# ax2.tick_params(labeltop=False) 
# ax.xaxis.tick_bottom()

# d = .015  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
# ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

# kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
# ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


# mc time :0.007348537445068359 
# mc time :0.00767064094543457 
# #5,000
# mc time :0.07430148124694824 
# mc time :0.07262396812438965
# #50,000 
# mc time :0.7377846240997314 
# mc time :0.7281239032745361 

# horizon=1
# foresee_time:0.000518798828125 
# foresee_time:0.00040602684020996094 
# foresee_time:0.0004935264587402344 
# foresee_time:0.00038743019104003906 
# foresee_time:0.0004055500030517578 
# foresee_time:0.0004992485046386719 
# average: 

# genUT foresee_time:0.00048804283142089844 
# genUT foresee_time:0.000492095947265625 
# genUT foresee_time:0.0005095005035400391 


# Expansion only
# horizon: 0, foresee_time:0.00041794776916503906 
# horizon: 1, foresee_time:0.0017733573913574219 
# horizon: 2, foresee_time:0.008446455001831055 
# horizon: 3, foresee_time:0.04363536834716797 
# horizon: 4, foresee_time:0.22621631622314453 
# horizon: 5, foresee_time:1.7395920753479004 
# horizon: 6, foresee_time:39.9112982749939










