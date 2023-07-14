import jax.numpy as np
import jax
from jax import jit
from utils.utils import wrap_angle
from diffrax import diffeqsolve, ODETerm, Tsit5


def step_without_wrap(y, u, dt):
    return rk4_integration_without_wrap( y, u, dt )

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

def rk4_integration_without_wrap(y, u, dt):
    k1 = dt * ( state_dot( y, u ) )
    k2 = dt * ( state_dot( y + k1/2.0, u ) )
    k3 = dt * ( state_dot( y + k2/2.0, u ) )
    k4 = dt * ( state_dot( y + k3, u ) )
    k = ( k1 + 2*k2 + 3*k3 + k4 ) / 6.0
    new_state = y + k
    return np.array([ new_state[0,0], new_state[1,0], new_state[2,0], new_state[3,0] ]).reshape(-1,1)

def step_using_xdot(state, state_dot, dt):
     state_next = state + state_dot * dt
     return np.array([ state_next[0,0], state_next[1,0], wrap_angle(state_next[2,0]), state_next[3,0] ]).reshape(-1,1)
    
def state_dot( y, u ):
    gravity = 9.8
    masscart = 0.7 #1.0
    masspole = 0.325 #0.1
    total_mass = masspole + masscart
    length = 0.5  # actually half the pole's length
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02  # seconds between state updates
        
    x = y[0,0]
    x_dot = y[1,0]
    theta = y[2,0]
    theta_dot = y[3,0]
    force = u[0,0]
    
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    # For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    temp = ( force + polemass_length * theta_dot**2 * sintheta ) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / ( length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)  )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    dydt = np.array([x_dot, xacc, theta_dot, thetaacc ]).reshape(-1,1)
        
    return dydt

def state_dot_diffrax(t, y, args):
    return np.append(state_dot( y[:-1].reshape(-1,1), y[-1].reshape(-1,1) ), np.array([[0.0]]), axis=0)
term = ODETerm(state_dot_diffrax)
solver = Tsit5()


def step(y, u, dt):
    """
    System of first order equations for a cart-pole system
    The policy commands the force applied to the cart
    (stable equilibrium point with the pole down at [~,0,0,0])
    # theta = 0 at stable equilibrium
    """
    return rk4_integration( y, u, dt )

@jit
def step_with_diffrax(y, u, dt):
    solution = diffeqsolve(term, solver, t0=0, t1=dt, dt0=dt/5, y0=np.append(y,u,axis=0))
    return solution.ys[0,:-1].reshape(-1,1)

@jit
def step_with_wrap_diffrax(y, u, dt):
    solution = diffeqsolve(term, solver, t0=0, t1=dt, dt0=dt/5, y0=np.append(y,u,axis=0))
    next_state = solution.ys[0,:-1].reshape(-1,1)
    next_state = np.array([ next_state[0,0], next_state[1,0], wrap_angle(next_state[2,0]), next_state[3,0] ]).reshape(-1,1)
    return next_state
    
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
    

