import jax.numpy as np
from utils.utils import wrap_angle

def step(state, action, params, dt): # 4 x 1 array
        x = state[0,0]
        x_dot = state[1,0]
        theta = state[2,0]
        theta_dot = state[3,0]
        force = action[0,0]
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        polemass_length = params[0]
        gravity = params[1]
        length = params[2]
        masspole = params[3]
        total_mass = params[4]
        # tau = params[5]

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = ( force + polemass_length * theta_dot**2 * sintheta ) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / ( length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)  )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass


        x = x + dt * x_dot
        x_dot = x_dot + dt * xacc
        theta = wrap_angle(theta + dt * theta_dot)
        theta_dot = theta_dot + dt * thetaacc
        
        return np.array([ x, x_dot, theta, theta_dot ]).reshape(-1,1)

def step_using_xdot(state, state_dot, dt):
     state_next = state + state_dot * dt
     return np.array([ state_next[0,0], state_next[1,0], wrap_angle(state_next[2,0]), state_next[3,0] ]).reshape(-1,1)
    
def state_dot( state, action, params ):
    x = state[0,0]
    x_dot = state[1,0]
    theta = state[2,0]
    theta_dot = state[3,0]
    force = action[0,0]
    
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    polemass_length = params[0]
    gravity = params[1]
    length = params[2]
    masspole = params[3]
    total_mass = params[4]
    tau = params[5]

    # For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    temp = ( force + polemass_length * theta_dot**2 * sintheta ) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / ( length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)  )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    return np.array([ x_dot, xacc, theta_dot, thetaacc ]).reshape(-1,1)
    
def get_state_dot_noisy(state, action, params):
    X_dot = state_dot(state, action, params)
    # error_square = 0.01 + np.square(X_dot) # /2  #never let it be 0!!!!
    # cov = np.diag( error_square[:,0] )
    # X_dot = X_dot + X_dot/2 #X_dot = X_dot + X_dot/6
    # return X_dot, cov
    return X_dot, np.zeros((4,4))

