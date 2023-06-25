import jax.numpy as np
import jax
from utils.utils import wrap_angle

def step(y, u, dt):
    """
    System of first order equations for a cart-pole system
    The policy commands the force applied to the cart
    (stable equilibrium point with the pole down at [~,0,0,0])
    # theta = 0 at stable equilibrium
    """
    return rk4_integration( y, u, dt )
    x, x_dot, theta, theta_dot = y[0,0], y[1,0], y[2,0], y[3,0]
    
    m1 = 0.5  # mass of the cart
    m2 = 0.5  # mass of the pendulum
    l = 0.5   # length of the pendulum
    b = 0.1   # friction coefficient
    g = 9.81  # acceleration of gravity
        
    den = 4*(m1+m2)-3*m2*np.cos(theta)**2
    
    dydt = np.array([x_dot,
            (2*m2*l*theta_dot**2*np.sin(theta)+3*m2*g*np.sin(theta)*np.cos(theta)+4*u[0,0]-4*b*x_dot)/den,
            theta_dot,
            (-3*m2*l*theta_dot**2*np.sin(theta)*np.cos(theta)-6*(m1+m2)*g*np.sin(theta)-6*(u[0,0]-b*x_dot)*np.cos(theta))/(l*den)]).reshape(-1,1)
    # print(f"dy_dt: {dydt.T}")
    new_state = y + dydt * dt
    new_state_clipped = np.array([ new_state[0,0], new_state[1,0], wrap_angle(new_state[2,0]), new_state[3,0] ]).reshape(-1,1)

    return new_state

def get_next_states_from_dynamics(states, controls, dt):
    new_states = rk4_integration( states[:,0].reshape(-1,1), controls[:,0].reshape(-1,1), dt )
    for i in range(1,states.shape[1]):
        new_states = np.append( new_states, rk4_integration( states[:,i].reshape(-1,1), controls[:,i].reshape(-1,1), dt ), axis=1 )
    return new_states
    

def rk4_integration(y, u, dt):
    k1 = dt * ( state_dot( y, u ) )
    k2 = dt * ( state_dot( y + k1/2.0, u ) )
    k3 = dt * ( state_dot( y + k2/2.0, u ) )
    k4 = dt * ( state_dot( y + k3, u ) )
    k = ( k1 + 2*k2 + 3*k3 + k4 ) / 6.0
    new_state = y + k
    return np.array([ new_state[0,0], new_state[1,0], wrap_angle(new_state[2,0]), new_state[3,0] ]).reshape(-1,1)

def step_using_xdot(state, state_dot, dt):
     state_next = state + state_dot * dt
     return np.array([ state_next[0,0], state_next[1,0], wrap_angle(state_next[2,0]), state_next[3,0] ]).reshape(-1,1)
    
def state_dot( y, u ):
    x, x_dot, theta, theta_dot = y[0,0], y[1,0], y[2,0], y[3,0]
    
    m1 = 0.5  # mass of the cart
    m2 = 0.5  # mass of the pendulum
    l = 0.5   # length of the pendulum
    b = 0.1   # friction coefficient
    g = 9.81  # acceleration of gravity
        
    den = 4*(m1+m2)-3*m2*np.cos(theta)**2
    
    dydt = np.array([x_dot,
            (2*m2*l*theta_dot**2*np.sin(theta)+3*m2*g*np.sin(theta)*np.cos(theta)+4*u[0,0]-4*b*x_dot)/den,
            theta_dot,
            (-3*m2*l*theta_dot**2*np.sin(theta)*np.cos(theta)-6*(m1+m2)*g*np.sin(theta)-6*(u[0,0]-b*x_dot)*np.cos(theta))/(l*den)]).reshape(-1,1)

    return dydt
    
def get_state_dot_noisy(state, action):
    X_dot = state_dot(state, action)
    error_square = 0.01 + np.square(X_dot) # /2  #never let it be 0!!!!
    cov = np.diag( error_square[:,0] )
    X_dot = X_dot + X_dot/2 #X_dot = X_dot + X_dot/6
    return X_dot, cov

    X_dot = state_dot(state, action)
    return X_dot, np.zeros((4,4))

def get_state_dot_noisy_mc(state, action, key):
    mu, cov = get_state_dot_noisy(state, action)
    key, subkey = jax.random.split(key)
    rnd1 = jax.random.normal( subkey, shape = (4,1) )
    return mu + np.sqrt( cov )@ rnd1, key

def get_state_dot_noisy_rk4(state, action, dt):
    k1 = dt * ( state_dot( state, action ) )
    k2 = dt * ( state_dot( state + k1/2.0, action ) )
    k3 = dt * ( state_dot( state + k2/2.0, action ) )
    k4 = dt * ( state_dot( state + k3, action ) )
    X_dot = ( k1 + 2*k2 + 3*k3 + k4 ) / 6.0 / dt

    return X_dot, np.zeros((4,4))
    

