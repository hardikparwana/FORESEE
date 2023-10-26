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
        cost += jnp.sum( jnp.square( U[:,i] ) )      #
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
            new_states, new_weights = foresee_propagate( X[:,i].reshape(n,2*n+1, order='F'), weights[:,i].reshape(1,2*n+1), U[:,i].reshape(-1,1), dt )
            # new_states, new_weights = foresee_propagate_GenUT( X[:,i].reshape(n,2*n+1, order='F'), weights[:,i].reshape(1,2*n+1), U[:,i].reshape(-1,1), dt )
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
lb = jnp.concatenate( ( -10 * jnp.ones(n*(N+1)*(2*n+1)),  0 * jnp.ones((N+1)*(2*n+1))  , -3 * jnp.ones(N*m)  ) )
ub = jnp.concatenate( (  10 * jnp.ones(n*(N+1)*(2*n+1)),  1 * jnp.ones((N+1)*(2*n+1))  ,  3 * jnp.ones(N*m)  ) )

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
fig.savefig('mpc_ss_ci_obj1.png')
fig.savefig('mpc_ss_ci_obj1.eps')
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



# CI obj 2
# ineq: [ 3.13400000e+01  3.13046652e+01  3.07521615e+01  2.95397543e+01
#   2.76765775e+01  2.52327090e+01  2.23063756e+01  1.90169735e+01
#   1.55039595e+01  1.19276076e+01  8.47041366e+00  5.33804631e+00
#   2.75533005e+00  9.43759597e-01  5.67294922e-02  3.38141675e-02
#  -1.03000291e-08  1.82876276e-01  6.76346300e-01  1.37039972e+00
#   2.32487784e+00  3.62887681e+00  4.85967726e+00  5.73030745e+00
#   7.38800229e+00  9.52691131e+00  1.18518072e+01  1.40917865e+01
#   1.60242444e+01  1.74972520e+01  1.84440545e+01  1.88874666e+01
#   1.89870487e+01  1.86190079e+01  1.77444635e+01  1.64054754e+01
#   1.47427603e+01  1.35994920e+01  1.29214294e+01  1.26807804e+01
#   1.24515952e+01]
# final cost: 15805.209909961402
# Weights sum: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# violation_statics_ci_obj2: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9998, 0.9894, 0.95282, 0.94542, 0.94814, 0.96058, 0.97544, 0.98624, 0.993, 0.9966, 0.99836, 0.99902, 0.9995, 0.99984, 0.99992, 0.99996, 0.99998, 0.99998, 0.99998, 0.99998, 0.99994, 0.99988, 0.99966, 0.99892, 0.9976, 0.99564, 0.99282, 0.99018, 0.98734]


# Mean obj 2
# ineq: [ 3.33000000e+01  3.33005000e+01  3.27969457e+01  3.16535670e+01
#   2.98654970e+01  2.75187192e+01  2.47003385e+01  2.15064117e+01
#   1.80474332e+01  1.44516025e+01  1.08675571e+01  7.46773884e+00
#   4.45304960e+00  2.05794665e+00  5.35985611e-01 -9.96196962e-09
#  -9.96042661e-09  5.67175802e-01  1.90064499e+00  3.97629025e+00
#   6.58374192e+00  9.30294669e+00  1.17085061e+01  1.46509413e+01
#   1.79206802e+01  2.11566841e+01  2.40647278e+01  2.64367086e+01
#   2.81623804e+01  2.92328724e+01  2.97359314e+01  2.99368405e+01
#   2.96525300e+01  2.87737999e+01  2.73136328e+01  2.55290320e+01
#   2.43328695e+01  2.36961895e+01  2.36025584e+01  2.37157093e+01
#   2.38263032e+01]
# final cost: 15412.676311677367
# Weights sum: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# violation_statics_mean_obj2: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99798, 0.91612, 0.65376, 0.51694, 0.50758, 0.60238, 0.7759, 0.91908, 0.9784, 0.99482, 0.99832, 0.99954, 0.99988, 0.99994, 0.99994, 0.99994, 0.99996, 0.99996, 0.99996, 0.99988, 0.99986, 0.99964, 0.99936, 0.99802, 0.9964, 0.9944, 0.9925, 0.99056, 0.98844] 

# Mean obj1
# ineq: [ 3.33000000e+01  3.33005000e+01  3.27969457e+01  3.16535670e+01
#   2.98827813e+01  2.75527946e+01  2.47476118e+01  2.15643768e+01
#   1.81157859e+01  1.45325342e+01  1.09658086e+01  7.58989270e+00
#   4.60352926e+00  2.22405268e+00  6.56576381e-01 -9.85809149e-09
#  -9.94734042e-09  6.28539577e-01  1.97245492e+00  3.94046006e+00
#   6.38878481e+00  9.16125807e+00  1.21057981e+01  1.50815885e+01
#   1.79621465e+01  2.06364053e+01  2.30088910e+01  2.49996539e+01
#   2.65443024e+01  2.75942298e+01  2.81787545e+01  2.83906564e+01
#   2.83192032e+01  2.80435867e+01  2.76297047e+01  2.71293120e+01
#   2.65807102e+01  2.60103325e+01  2.54347686e+01  2.48629398e+01
#   2.42982590e+01]
# final cost: 15735.327833333396
# Weights sum: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# violation_statics_mean_obj1: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99834, 0.93024, 0.67984, 0.51576, 0.50342, 0.60886, 0.78312, 0.91884, 0.97768, 0.99448, 0.99856, 0.9995, 0.99986, 0.99994, 0.99996, 0.99996, 0.99996, 0.99994, 0.9999, 0.99984, 0.99972, 0.99958, 0.99934, 0.99894, 0.99824, 0.9977, 0.99676, 0.99564, 0.99412] 

# CI obj1
# ineq: [ 3.13400000e+01  3.13046652e+01  3.07521615e+01  2.95576392e+01
#   2.77197407e+01  2.53008242e+01  2.23970542e+01  1.91287277e+01
#   1.56375974e+01  1.20868383e+01  8.66188426e+00  5.56933818e+00
#   3.02713900e+00  1.23054972e+00  2.74712881e-01  4.39842770e-02
#  -1.00174658e-08  1.50267129e-01  5.67417742e-01  1.28603838e+00
#   2.38859759e+00  3.81945383e+00  5.48480946e+00  7.28531258e+00
#   9.12698219e+00  1.09245554e+01  1.26019141e+01  1.40916219e+01
#   1.53347458e+01  1.62814832e+01  1.68925363e+01  1.71985829e+01
#   1.72616661e+01  1.71427243e+01  1.68961702e+01  1.65670864e+01
#   1.61903691e+01  1.57911565e+01  1.53859925e+01  1.49843268e+01
#   1.45900969e+01]
# final cost: 16152.210339976502
# Weights sum: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# violation_statics_ci_obj1: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99994, 0.99454, 0.96706, 0.94944, 0.9501, 0.96088, 0.9744, 0.9858, 0.99344, 0.99754, 0.99918, 0.9997, 0.99986, 0.99992, 0.99998, 0.99998, 0.99998, 0.99998, 0.99996, 0.99996, 0.99996, 0.9999, 0.99984, 0.9998, 0.9997, 0.99964, 0.99956, 0.99944, 0.99926] 

