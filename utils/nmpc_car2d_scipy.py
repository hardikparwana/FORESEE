# from utils.initialize_track import *
import matplotlib.pyplot as plt
plt.ion()
from initialize_track import *
from robot_models.car2D import *

from jax import jit, grad, jacfwd, jacrev
import cyipopt

import pdb

x0 = track["center"][0,0]
y0 = track["center"][1,0]
x1 = track["center"][0,1]
y1 = track["center"][1,1]
heading = np.arctan2( y1 - y0, x1 - x0 )
Theta = 0

# x0 = 0.3715 # 0.37150648, #13.130269
# y0 = -0.1172 #  -0.11724903, #-2.2879753
# Theta = 1.72 #70



######## Sim parameters #########
dt = MPC_vars["Ts"]#  0.05
n = 7
m = 3
# f = 10.0
N = 10#MPC_vars["N"]  # MPC horizon
control_lb = [ -1, -2 * np.pi ]
control_ub = [  1,  2 * np.pi ]
r = widthTrack # track width

# MPC objective coefficents
ql = MPC_vars["qL"]  #1.0
qc = MPC_vars["qC"]  #  1.0
gamma = MPC_vars["qVtheta"] #  1.0

#################################
# pdb.set_trace()
# Car Initialization
# pdb.set_trace()
robot = Car2D( np.array([x0, y0, heading, 0.03, 0, 0, Theta]) ,dt, ax, ModelParams )
car_dynamics = car2D_dynamics(ModelParams)




# Declare Variables
print(f"{MPC_vars['invTx'].shape }, {(2*jnp.zeros((n,N+1))).shape} ")

# pdb.set_trace()
# X = (MPC_vars["invTx"]) @ (2*jnp.zeros((n,N+1)))
robot_init_state = robot.X # NOT normalized

X_bound_lb = np.ones((n,N+1)) * np.array([ -1, -1, -3, 0, -1, -1, 0 ]).reshape(-1,1)
X_bound_ub = np.ones((n,N+1)) * np.array([  1,  1,  3, 1,  1,  1, 1 ]).reshape(-1,1)
U_bound_lb = np.ones((m,N)) * np.array([ 0.05, -1, 0.0 ]).reshape(-1,1)
U_bound_ub = np.ones((m,N)) * np.array([  1.0,  1, 1.0 ]).reshape(-1,1)

mpc_X_lb = jnp.concatenate( (X_bound_lb.T.reshape(-1,1), U_bound_lb.T.reshape(-1,1)), axis=0 )[:,0]
mpc_X_ub = jnp.concatenate( (X_bound_ub.T.reshape(-1,1), U_bound_ub.T.reshape(-1,1)), axis=0 )[:,0]

X_guess = np.ones((n,N+1)) * robot_init_state 
U_guess = np.ones((m,N)) * np.array([ 1.0, 0.05, 0 ]).reshape(-1,1)

for t in range(N):
    X_guess[:,t+1] = X_guess[:,t] + dt * np.asarray(car_dynamics( X_guess[:,t].reshape(-1,1), U_guess[:,t].reshape(-1,1) )[:,0])
    dist = np.linalg.norm( X_guess[0:2,t+1] - X_guess[0:2,t] )
    X_guess[6, t+1] = X_guess[6, t] + dist
X_guess = np.asarray( X_guess )

path_plot = ax.plot( X_guess[0,:], X_guess[1,:], 'm*' )

# plt.show()

# exit()


X = (MPC_vars["Tx"]) @ X_guess
U = MPC_vars["Tu"] @ U_guess
# print(f"x: {X}")
# exit()

# append all to one array
mpc_X = jnp.concatenate( (X.T.reshape(-1,1), U.T.reshape(-1,1)), axis=0 )[:,0] # has to be a 1D array for ipopt
# plt.show()
# Functions for MPC
@jit
def compute_xy_from_path_length( path_length ):
    pred_x, grad_x = path_length_func_x(path_length)
    pred_y, grad_y = path_length_func_y(path_length)
    theta = jnp.arctan2( grad_y, grad_x )
    return jnp.array([ pred_x, pred_y, theta ]).reshape(-1,1)
    print(f"{pred_x}, {pred_y}, {theta}")
    return jnp.concatenate((pred_x, pred_y, theta)).reshape(-1,1)

@jit
def track_constraint(X, r): # should be >=0
    path_length = X[6,0]
    path_location = compute_xy_from_path_length( path_length )
    return (X[0:2] - path_location[0:2]).T @ (X[0:2] - path_location[0:2]) - r**2

@jit
def track_objective(X, u):
    path_location = compute_xy_from_path_length( X[6,0] )
    theta = path_location[2,0]
    lag_error = -jnp.cos( theta ) * ( X[0,0] - path_location[0,0] ) - jnp.sin( X[1,0] - path_location[1,0] )
    contouring_error = jnp.sin( theta ) * ( X[0,0] - path_location[0,0] ) - jnp.cos( theta ) * ( X[1,0] - path_location[1,0] )
    return ql * lag_error**2 + 20 * qc * contouring_error**2 - gamma * u[2,0]

@jit
def step( x,u,dt ):
    return x + car_dynamics(x, u) * dt

# def step_sum( x,u,dt ):
#     return jnp.sum(step(x,u,dt))
# step_grad = grad( step_sum )
# pdb.set_trace()

@jit
def objective(mpc_X):
    X = MPC_vars["invTx"] @ mpc_X[0:n*(N+1)].reshape(n,N+1, order='F')
    U = MPC_vars["invTu"] @ mpc_X[-m*N:].reshape(m,N, order='F')
    cost = 0
    for i in range(N):
        cost = cost + track_objective( X[:,i].reshape(-1,1), U[:,i].reshape(-1,1) )
        # cost += jnp.sum( jnp.square(X[0:2,i] - jnp.array([1,2])) )
    return cost
objective_grad = jit(grad(objective, 0))

@jit
def equality_constraint(mpc_X, robot_init_state):
    '''
        Assume of form g(x) = 0
        Returns g(x) as 1D array
    '''
    # return jnp.array([0.0])
    X = MPC_vars["invTx"] @ mpc_X[0:n*(N+1)].reshape(n,N+1, order='F') 
    U = MPC_vars["invTu"] @ mpc_X[-m*N:].reshape(m,N, order='F')
    const = jnp.zeros(1)
    
    # Initial state constraint
    init_state_error = X[:,0] - robot_init_state[:,0]
    const = jnp.append(const, init_state_error)
    # return const

    # Dynamic Constraint
    for i in range(N):
        dynamics_const = X[:,i+1] - step( X[:,i].reshape(-1,1), U[:,i].reshape(-1,1), dt )[:,0]
        const  = jnp.append( const, dynamics_const )
    return const[1:]
equality_constraint_grad = jit(jacrev(equality_constraint, 0))

@jit
def inequality_constraint(mpc_X):
    '''
        Assume of form g(x) >= 0
        Retruns g(x) as 1D array
    '''
    # return jnp.array([0.0])
    X = MPC_vars["invTx"] @ mpc_X[0:n*(N+1)].reshape(n,N+1, order='F')
    U = MPC_vars["invTu"] @ mpc_X[-m*N:].reshape(m,N, order='F')
    const = jnp.ones(1)
    return const
    # Track boundary constraint
    for i in range(N+1):
        const = jnp.append( const, track_constraint(X, r)[0,0] )

    # Control input bounds
    for i in range(N):
        const = jnp.append( const, control_ub[0] - U[0,i] )
        const = jnp.append( const, control_ub[1] - U[1,i] )
        const = jnp.append( const, U[0,i] - control_lb[0] )
        const = jnp.append( const, U[1,i] - control_lb[1] )

    return const[1:]

# inequality_constraint(0.64645*jnp.ones(22))
# inequality_constraint(mpc_X)
inequality_constraint_grad = jit(jacrev(inequality_constraint, 0))

# lb = -100 * jnp.ones((N)*(n+m)+n)
# ub =  100 * jnp.ones((N)*(n+m)+n)



# Scipy optimize
from scipy.optimize import minimize


        # {'type': 'ineq', 'fun': inequality_constraint, 'jac': inequality_constraint_grad} )

res = None



for t in range(300):

    robot_init_state = np.copy(robot.X)
    eq_cons = lambda z: equality_constraint(z, robot_init_state)
    eq_cons_grad = lambda z: equality_constraint_grad(z, robot_init_state)

    cons = ( {'type': 'eq', 'fun': eq_cons, 'jac': eq_cons_grad} )#,
    res = minimize( objective, mpc_X, method='SLSQP', constraints=cons, options={'maxiter': 10000} )

    print(res.message)

    sol_X = MPC_vars["invTx"] @ res.x[0:n*(N+1)].reshape(n,N+1, order='F')
    sol_U = MPC_vars["invTu"] @ res.x[-m*N:].reshape(m,N, order='F')

    robot.step(sol_U[:,0].reshape(-1,1))

    sol_X[:,1:] = sol_X[:,:-1]
    sol_U[:,1:] = sol_U[:,:-1]
    mpc_X = jnp.concatenate( (X.T.reshape(-1,1), U.T.reshape(-1,1)), axis=0 )[:,0]

    # plt.plot( sol_X[0,:], sol_X[1,:], 'm*' )
    path_plot[0].set_xdata( sol_X[0,:] )
    path_plot[0].set_ydata( sol_X[1,:] )

    fig.canvas.draw()
    fig.canvas.flush_events()

    

plt.plot(sol_X[0,:], sol_X[1,:], 'k')

# pdb.set_trace()
plt.show()


