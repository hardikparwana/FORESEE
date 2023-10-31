import jax
import time
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev, lax
import cyipopt
import matplotlib.pyplot as plt
from obstacles import circle
from trajectory_opt_utils import *
jax.config.update("jax_enable_x64", True)
import numpy as np

# Control Hyperparameters
n = 4  # Dimension of state
m = 2  # Dimension of control input
tf = 2.0
dt = 0.05
N = int(tf/dt) # MPC horizon
d_min = 0.3
control_bound = [3,3]

# Declare Variables
X = 0.7*jnp.zeros((n*(2*n+1),N+1))
X_weights = jnp.ones((2*n+1,N+1))/9.0
U = jnp.zeros((m,N))
robot_init_state = jnp.array([-0.5,-0.5, 0, 0]).reshape(-1,1)

# append all to one array
mpc_X = jnp.concatenate( (X.T.reshape(-1,1), X_weights.T.reshape(-1,1), U.T.reshape(-1,1)), axis=0 )[:,0] # has to be a 1D array for ipopt

# Declare Parameters
obstacle1X = jnp.array([0.7,1.0]).reshape(-1,1)
# obstacle2X = jnp.array([1.5,1.9]).reshape(-1,1)
goal = jnp.array([2.0,2.0]).reshape(-1,1)

@jit
def objective(mpc_X):
    X = mpc_X[0:n*(2*n+1)*(N+1)].reshape(n*(2*n+1),N+1, order='F')
    weights = mpc_X[n*(2*n+1)*(N+1):n*(2*n+1)*(N+1)+(2*n+1)*(N+1)].reshape(2*n+1,N+1, order='F')
    U = mpc_X[-m*N:].reshape(m,N, order='F')
    cost = 0
    def body(i, inputs):
        cost = inputs
        for j in range(9):
            cost += weights[j,i]*jnp.sum( 100*jnp.square(X[n*j+0:n*j+2,i] - goal[:,0]) )
        # cost += jnp.sum( jnp.square( U[:,i] ) )      #
        return cost  
    return lax.fori_loop( 0, N, body, cost )

# print(f"obj test: {objective(mpc_X)}")
objective_grad = jit(grad(objective, 0))


# exit()

@jit
def equality_constraint(mpc_X):
    '''
        Assume of form g(x) = 0
        Returns g(x) as 1D array
    '''
    # return jnp.array([0.0])
    X = mpc_X[0:n*(2*n+1)*(N+1)].reshape(n*(2*n+1),N+1, order='F')
    weights = mpc_X[n*(2*n+1)*(N+1):n*(2*n+1)*(N+1)+(2*n+1)*(N+1)].reshape(2*n+1,N+1, order='F')
    U = mpc_X[-m*N:].reshape(m,N, order='F')
    const = jnp.zeros(1)
    
    # Initial state constraint
    init_state_error = X[:,0] - jnp.repeat( robot_init_state, 9, axis=1 ).T.reshape(-1,1)[:,0]  #jnp.repeat(robot_init_state[:,0],9 )
    const = jnp.append(const, init_state_error)
    init_weight_error = weights[:,0] - jnp.ones(9)/9.0
    const = jnp.append(const, init_weight_error)
    
    # Dynamic Constraint
    def run(X, weights,U, const_in, const_weights_in):
        
        def body(i, inputs):
            X, weights, U, const_in, const_weights_in = inputs
            # new_states, new_weights = foresee_propagate( X[:,i].reshape(n,2*n+1, order='F'), weights[:,i].reshape(1,2*n+1), U[:,i].reshape(-1,1), dt )
            new_states, new_weights = foresee_propagate_GenUT( X[:,i].reshape(n,2*n+1, order='F'), weights[:,i].reshape(1,2*n+1), U[:,i].reshape(-1,1), dt )
            dynamics_const = X[:,i+1] - new_states.T.reshape(-1,1)[:,0]
            
            const_next = const_in.at[:,i].set(dynamics_const)
            
            weight_error = weights[:,i+1] - new_weights[0,:] 
            const_weights_next = const_weights_in.at[:,i].set(weight_error)          
            return X, weights, U, const_next, const_weights_next
        return lax.fori_loop( 0, N, body, (X, weights,U, const_in, const_weights_in) )[3:5]
    
    const_in = jnp.zeros((n*(2*n+1),N))
    const_weights_in = jnp.zeros((2*n+1,N))
    const_temp, const_temp_weights = run(X, weights,U, const_in, const_weights_in)
    const = jnp.append(const, const_temp.T.reshape(-1,1)[:,0])
    const = jnp.append(const, const_temp_weights.T.reshape(-1,1)[:,0])

    return const[1:]
equality_constraint(mpc_X)
equality_constraint_grad = jit(jacrev(equality_constraint, 0))

@jit
def barrier_average(weights, X, obstacleX):
    barrier = 0
    for j in range(9):
        barrier += weights[j]*( jnp.sum(jnp.square(X[n*j+0:n*j+2] - obstacleX)) - (2*d_min)**2  ) 
    # barrier = barrier - d_min**2 - d_min**2
    return 10*barrier

@jit
def barrier_ci(weights, X, obstacleX):
    j = 0
    dist = jnp.sum(jnp.square(X[n*j+0:n*j+2] - obstacleX)) - (2*d_min)**2
    for j in range(1,9):
        dist = jnp.append(dist, jnp.sum(jnp.square(X[n*j+0:n*j+2] - obstacleX)) - (2*d_min)**2)
    mean_dist, cov_dist = get_mean_cov(dist.reshape(1,-1), weights.reshape(1,-1))
    barrier = mean_dist[0,0] - 1.96 * jnp.sqrt(0.01+cov_dist[0,0])#1.96*jnp.sqrt(cov_dist[0,0])
    return 10*barrier

@jit
def inequality_constraint(mpc_X):
    '''
        Assume of form g(x) >= 0
        Retruns g(x) as 1D array
    '''
    # return jnp.array([0.0])

    X = mpc_X[0:n*(2*n+1)*(N+1)].reshape(n*(2*n+1),N+1, order='F')
    weights = mpc_X[n*(2*n+1)*(N+1):n*(2*n+1)*(N+1)+(2*n+1)*(N+1)].reshape(2*n+1,N+1, order='F')
    U = mpc_X[-m*N:].reshape(m,N, order='F')
    const = jnp.zeros((1,1))
    
    # Collision avoidance
    # barrier_average( weights[:,0], X[:,0], obstacle1X[:,0] )
    barrier = 0
    def main_barrier(X, weights, U, const_in):
        # const = jnp.zeros(N+1)
        def body(i, inputs):
            X, weights, U, const_in = inputs
            # barrier = barrier_average( weights[:,i], X[:,i], obstacle1X[:,0] )
            barrier = barrier_ci( weights[:,i], X[:,i], obstacle1X[:,0] )
            # barrier = jnp.sum(jnp.square(X[0:2,i] - obstacle1X[:,0])) - d_min**2
            const_next = const_in.at[i].set(barrier)
            return X, weights, U, const_next
        return lax.fori_loop( 0, N+1, body, (X, weights, U, const_in) )[3]
    const_in = jnp.zeros(N+1)
    const_temp = main_barrier(X, weights, U, const_in)
    const = jnp.append(const, const_temp)
        
    # Control input bounds
    # def main_control(U, const_in):
    #     # const = jnp.zeros((2,N))
    #     def body(i, inputs):
    #         U, const_in = inputs
    #         temp = control_bound[0]*control_bound[0] - U[0,i]*U[0,i]
    #         temp = jnp.append( temp, control_bound[1]*control_bound[1] - U[1,i]*U[1,i] )
    #         const_next = const_in.at[:,i].set(temp)
    #         return U, const_next
    #     return lax.fori_loop( 0, N, body, (U, const_in) )[1]
    # const_in = jnp.zeros((2,N))
    # const_temp = main_control(U, const_in).T.reshape(-1,1)
    # const = jnp.append(const, const_temp)
    
    return const[1:]
inequality_constraint(mpc_X)
inequality_constraint_grad = jit(jacrev(inequality_constraint, 0))



class CYIPOPT_Wrapper():
    '''
        Based on native interface https://cyipopt.readthedocs.io/en/stable/tutorial.html#scipy-compatible-interface
        Note: the above page has 2 interfaces: one based on scipy and one based on a direct wrapper(scroll down on the above webpage). Scipy interface is slower so I use the direct wrapper
    '''

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return objective(x)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        return objective_grad(x)

    def constraints(self, x):
        """Returns the constraints."""
        return jnp.append( equality_constraint(x), inequality_constraint(x) )

    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        return jnp.append( equality_constraint_grad(x), inequality_constraint_grad(x), axis=0 )

# Upper and lower bounds on optimization variables
lb = jnp.concatenate( ( -10 * jnp.ones(n*(N+1)*(2*n+1)),  -100000 * jnp.ones((N+1)*(2*n+1))  , -3 * jnp.ones(N*m)  ) )
ub = jnp.concatenate( (  10 * jnp.ones(n*(N+1)*(2*n+1)),  100000 * jnp.ones((N+1)*(2*n+1))  ,  3 * jnp.ones(N*m)  ) )

# Equality and inequality constraint function upper and lower bound
# Equality g(x) == 0
cl_equality = jnp.zeros( equality_constraint(mpc_X).size )
cu_equality = cl_equality
# Inequality  Infinity>=g(x)>=0 0
cl_inequality = jnp.zeros( inequality_constraint(mpc_X).size )
cu_inequality = 2.0e19 * jnp.ones( cl_inequality.size )
# Combine all in one
cl = jnp.append( cl_equality, cl_inequality )
cu = jnp.append( cu_equality, cu_inequality )

# run once to do grad and jit
equality_constraint_grad(mpc_X)
inequality_constraint_grad(mpc_X)
t0 = time.time()
print(f"Starting: ")

nlp = cyipopt.Problem(
   n=mpc_X.size,
   m=len(cl),
   problem_obj=CYIPOPT_Wrapper(),
   lb=lb,
   ub=ub,
   cl=cl,
   cu=cu,
)
nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 1e-3)
nlp.add_option('linear_solver', 'ma57')
nlp.add_option('print_level', 5)
nlp.add_option('max_iter', 2000)
t1 = time.time()
print(f"set problem in :{t1-t0}")

x, info = nlp.solve(mpc_X)
print(f"solved problem in :{time.time()-t1}")

sol_X = x[0:n*(2*n+1)*(N+1)].reshape(n*(2*n+1),N+1, order='F')
sol_weights = x[n*(2*n+1)*(N+1):n*(2*n+1)*(N+1)+(2*n+1)*(N+1)].reshape(2*n+1,N+1, order='F')
sol_U = x[-m*N:].reshape(m,N, order='F')
    
    
print(f"sol_X: {sol_X}")
print(f"sol_weights: {sol_weights}")
print(f"sol_U: {sol_U}")

# t2 = time.time()
# x, info = nlp.solve(x)
# print(f"second solve time: {time.time()-t2}")

fig, ax = plt.subplots(1,1)
ax.set_aspect('equal')
ax.set_xlim([-0.6, 3.0])
ax.set_ylim([-0.7, 3.0])
    




fig2, ax2 = plt.subplots(1,1)
for j in range(9):    
    ax2.plot(sol_X[n*j + 0,:], sol_X[n*j + 1,:], marker='h') # 'g*'Mean plot
circ = plt.Circle((obstacle1X[0],obstacle1X[1]),d_min,linewidth = 1, edgecolor='k',facecolor='k')
ax2.add_patch(circ)


# circ2 = plt.Circle((obstacle2X[0],obstacle2X[1]),d_min,linewidth = 1, edgecolor='k',facecolor='k')
# ax.add_patch(circ2)

print(f"ineq: {inequality_constraint(x)}")
print(f"final cost: {objective(x)}")
print(f"Weights sum: {jnp.sum(sol_weights, axis=0)}")

def barrier_ci_particle(X, obstacleX):
    dist = jnp.sum(jnp.square(X[0:2] - obstacleX)) - (2*d_min)**2
    return 10*dist

# Perform MC
num_particles = 50000
mc_states = np.repeat(np.asarray(robot_init_state), num_particles, axis=1)
mus = np.ones((2,1))
covs = np.ones((2,1))
weights_mc = np.ones((1,num_particles))/num_particles

violation_statistics = []

for i in range(N):
    mu, cov = get_mean_cov_np(mc_states, np.ones((1,num_particles))/num_particles)
    mus = np.append( mus, mu[0:2], axis=1 )
    covs = np.append( covs, np.diag(cov[0:2,0:2]).reshape(-1,1), axis=1 )
    confidence_ellipse(mu.reshape(-1,1), cov, ax, n_std=1.96, facecolor='cyan', alpha=0.2)
    num_violation = 0
    # Evaluate Constraint satisfaction    
    for j in range(num_particles):
        xdot_mu, xdot_cov = dynamics_xdot_noisy_np(mc_states[:,j].reshape(-1,1), sol_U[:,i].reshape(-1,1))
        xdot = np.array([  np.random.normal(xdot_mu[0,0], np.sqrt(xdot_cov[0,0])), np.random.normal(xdot_mu[1,0], np.sqrt(xdot_cov[1,1])), np.random.normal(xdot_mu[2,0], np.sqrt(xdot_cov[2,2])), np.random.normal(xdot_mu[3,0], np.sqrt(xdot_cov[3,3]))  ])#.reshape(-1,1)
        mc_states[:,j] = mc_states[:,j] + xdot * dt
        
        # compute statistics
        # num_violation += (barrier_ci_particle( mc_states[:,j], obstacle1X[:,0]  ) >= 0)
        if ( (np.sum(np.square(mc_states[0:2,j] - obstacle1X[:,0])) - (2*d_min)**2) >= 0 ):
            num_violation += 1.0
    num_violation = num_violation / num_particles
    violation_statistics.append(num_violation)    
print(f"violation_statics: {violation_statistics} \n")


confidence_ellipse(mu.reshape(-1,1), cov, ax, n_std=1.96, facecolor='cyan', alpha=0.2, label='MC 95% CI')
mus = mus[:,1:]
covs = covs[:,1:]
# for i in range(N):
#     confidence_ellipse(mus[0:2,i].reshape(-1,1), np.diag(covs[0:2,i]), ax, n_std=1.96, facecolor='cyan', alpha=0.2)

circ = plt.Circle((obstacle1X[0],obstacle1X[1]),2*d_min,linewidth = 1, edgecolor='k',facecolor='k')
ax.add_patch(circ)
# circ = plt.Circle((obstacle1X[0],obstacle1X[1]),2*d_min,linewidth = 1, edgecolor='k',facecolor='none', linestyle='--')
# ax.add_patch(circ)
ax.scatter(mus[0,:], mus[1,:], edgecolor='b', facecolor='none', label='MC Mean')

mus = np.ones((2,1)) 
for i in range(N+1):
    mu, cov = get_mean_cov_np(np.asarray(sol_X[:,i].reshape(4,2*n+1, order='F')), np.asarray(sol_weights[:,i].reshape(1,-1)))
    mus = np.append( mus, mu[0:2], axis=1 )
    confidence_ellipse(mu, cov, ax, n_std=1.96, facecolor='orange')
    
    
confidence_ellipse(mu, cov, ax, n_std=1.96, facecolor='orange', label='FORESEE 95% CI')
ax.plot(mus[0,1:], mus[1,1:], 'r*', label='FORESEE Mean')

ax.legend(loc='upper left')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
fig.savefig('mpc_ss_CI_genUT_obj2.png')
fig.savefig('mpc_ss_CI_genUT_obj2.eps')
plt.show()
    
    
# @jit
# def step(x,u,dt):
#     return x+u*dt



  # i  = 0
    # j = 0
    # cost = jnp.sum( 100*jnp.square(X[n*j+0:n*j+2,i] - goal[:,0]) )
    # return cost

    # for i in range(N):
    #     const = jnp.append( const, control_bound[0]*control_bound[0] - U[0,i]*U[0,i] )
    #     const = jnp.append( const, control_bound[1]*control_bound[1] - U[1,i]*U[1,i] )
 
 # const_next = const.at[i:n*(2*n+1):(i+1)*n*(2*n+1)].set(dynamics_const)   
    # const_weights_next = const_weights.at[i*(2*n+1):(i+1)*(2*n+1)].set( weight_error )  
# lb = jnp.concatenate((-20*jnp.ones(X.size), -control_bound[0]*jnp.ones(N), -control_bound[1]*jnp.ones(N)) )
# ub = jnp.concatenate(( 20*jnp.ones(X.size),  control_bound[0]*jnp.ones(N),  control_bound[1]*jnp.ones(N)) )

# lb = -100 * jnp.ones((N)*(n+m)+n)
# ub =  100 * jnp.ones((N)*(n+m)+n)

    # for i in range(N):             
    #     dynamics_const = X[:,i+1] - new_states.T.reshape(-1,1)
    #     const  = jnp.append( const, dynamics_const )
    #     weight_error = weights[:,i+1] - new_weights[0,:]
    #     const = jnp.append( const, jnp.append )


    # for i in range(N+1):
    #     barrier = 0;
    #     for j in range(9):
    #         barrier += weights[j,i]*jnp.sum(jnp.square(X[4*j+0:4*j+2,i] - obstacle1X[:,0])) - d_min**2
    #     const = jnp.append( const, barrier )
        # barrier = jnp.sum(jnp.square(X[0:2,i] - obstacle2X[:,0])) - d_min**2
        # const = jnp.append( const, barrier )
        
        
#CI: solved problem in :213.546466588974
#meanL 



# GenUT obj1, CI
# ineq: [ 3.13400000e+01  3.13046652e+01  3.07521882e+01  2.95578491e+01
#   2.77202511e+01  2.53017194e+01  2.23984024e+01  1.91306120e+01
#   1.56401574e+01  1.20902743e+01  8.66641893e+00  5.57517509e+00
#   3.03433684e+00  1.23856382e+00  2.81504781e-01  4.65867990e-02
#  -9.33829830e-09  1.46828472e-01  5.60524960e-01  1.27754377e+00
#   2.37858474e+00  3.80734684e+00  5.46988456e+00  7.26686107e+00
#   9.10437088e+00  1.08972980e+01  1.25697648e+01  1.40546620e+01
#   1.52934349e+01  1.62366681e+01  1.68454259e+01  1.71506400e+01
#   1.72143120e+01  1.70968512e+01  1.68525882e+01  1.65261218e+01
#   1.61521366e+01  1.57556205e+01  1.53530819e+01  1.49539264e+01
#   1.45621251e+01]
# final cost: 16150.134854453276
# Weights sum: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

# violation_statics_genut_ci_obj1: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99994, 0.9941, 0.96666, 0.94784, 0.94916, 0.9595, 0.97306, 0.9848, 0.99248, 0.99676, 0.9988, 0.9996, 0.99992, 0.99998, 0.99998, 0.99998, 0.99998, 0.99998, 0.99998, 0.99998, 0.99996, 0.99992, 0.99986, 0.99978, 0.99966, 0.99942, 0.99932, 0.99926, 0.99918] 

# GenUT, obj1, mean
# ineq: [ 3.33000000e+01  3.33005000e+01  3.27969457e+01  3.16535680e+01
#   2.98828181e+01  2.75528724e+01  2.47477241e+01  2.15645075e+01
#   1.81159244e+01  1.45326784e+01  1.09659461e+01  7.59001443e+00
#   4.60365569e+00  2.22417750e+00  6.56642950e-01 -9.99828467e-09
#  -9.99779376e-09  6.28553178e-01  1.97242237e+00  3.94037207e+00
#   6.38865389e+00  9.16107414e+00  1.21055348e+01  1.50812452e+01
#   1.79617566e+01  2.06360043e+01  2.30084837e+01  2.49992117e+01
#   2.65437873e+01  2.75936235e+01  2.81780914e+01  2.83900039e+01
#   2.83186517e+01  2.80432273e+01  2.76296054e+01  2.71295030e+01
#   2.65811801e+01  2.60110347e+01  2.54356356e+01  2.48639009e+01
#   2.42992576e+01]
# final cost: 15735.353006603105
# Weights sum: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# violation_statics_genut_mean_obj1: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99876, 0.93244, 0.6803, 0.5131, 0.50428, 0.60658, 0.78302, 0.91752, 0.97672, 0.99446, 0.9983, 0.9996, 0.9999, 0.99992, 0.99996, 0.99996, 0.99996, 0.99996, 0.99994, 0.99986, 0.99984, 0.99964, 0.99922, 0.99878, 0.99818, 0.9974, 0.99642, 0.99504, 0.99336] 

# Mean obj 2
# ineq: [ 3.33000000e+01  3.33005000e+01  3.27969457e+01  3.16535680e+01
#   2.98655333e+01  2.75188119e+01  2.47004850e+01  2.15065773e+01
#   1.80476158e+01  1.44517928e+01  1.08677375e+01  7.46786273e+00
#   4.45316934e+00  2.05791740e+00  5.35984628e-01 -1.01028103e-08
#  -9.94612356e-09  5.67009565e-01  1.90014808e+00  3.97580945e+00
#   6.58308795e+00  9.30778442e+00  1.17192093e+01  1.46554102e+01
#   1.79212593e+01  2.11553877e+01  2.40631198e+01  2.64358324e+01
#   2.81627407e+01  2.92344672e+01  2.97383293e+01  2.99398205e+01
#   2.96531915e+01  2.87712807e+01  2.73082917e+01  2.55239410e+01
#   2.43285332e+01  2.36921651e+01  2.35989898e+01  2.37145934e+01
#   2.38177362e+01]
# final cost: 15412.7132715157
# Weights sum: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# violation_statics_genut_mean_obj2: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9982, 0.91426, 0.6506, 0.51562, 0.50726, 0.60296, 0.77864, 0.9216, 0.98048, 0.99566, 0.99862, 0.99968, 0.99996, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99998, 0.99992, 0.99986, 0.99938, 0.99834, 0.99688, 0.99498, 0.99304, 0.99134, 0.9891] 

#CI obj 2
# ineq: [ 3.13400000e+01  3.13046652e+01  3.07521882e+01  2.95400284e+01
#   2.76771905e+01  2.52336831e+01  2.23077534e+01  1.90188873e+01
#   1.55066094e+01  1.19311502e+01  8.47505885e+00  5.34415958e+00
#   2.76320230e+00  9.53068536e-01  6.51736769e-02  3.89040439e-02
#  -1.01853829e-08  1.76651558e-01  6.63689743e-01  1.35239881e+00
#   2.30787215e+00  3.61233140e+00  4.83914191e+00  5.71139104e+00
#   7.36358026e+00  9.49251955e+00  1.18053044e+01  1.40328044e+01
#   1.59537702e+01  1.74173465e+01  1.83576083e+01  1.87978817e+01
#   1.88968589e+01  1.85320598e+01  1.76653956e+01  1.63407360e+01
#   1.46950102e+01  1.35647720e+01  1.28966130e+01  1.26606030e+01
#   1.23644737e+01]
# final cost: 15803.320594545808
# Weights sum: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# violation_statics_genut_ci_obj2: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99982, 0.9896, 0.95232, 0.94346, 0.94604, 0.95862, 0.97396, 0.9849, 0.9925, 0.99664, 0.99834, 0.99884, 0.99938, 0.99966, 0.99982, 0.9999, 0.99992, 0.99994, 0.9999, 0.99988, 0.99978, 0.99972, 0.9995, 0.99886, 0.99718, 0.99524, 0.99316, 0.99088, 0.98762] 