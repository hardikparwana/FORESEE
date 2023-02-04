# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

""" 
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL contact:	Diego Romeres (romeres@merl.com)
"""

import numpy as np
import sympy as sym
import pickle as pkl


def pend(y, t, u):
    """
    System of first order equations for a pendulum system
    The policy commands the torque applied to the joint
    (stable equilibrium point with the pole down at [0,0])
    """
    theta, theta_dot = y
    
    m = 1.          # mass of the pendulum
    l = 1.          # lenght of the pendulum
    b = 0.1         # friction coefficient
    g = 9.81        # acceleration of gravity
    I = 1/3*m*l**2  # moment of inertia of a pendulum around extreme point

    dydt = [theta_dot, 
            (u -b*theta_dot - 1/2*m*l*g*np.sin(theta))/I]
    return dydt

def cartpole(y, t, u):
    """
    System of first order equations for a cart-pole system
    The policy commands the force applied to the cart
    (stable equilibrium point with the pole down at [~,0,0,0])
    """
    
    x, x_dot, theta, theta_dot = y
    
    m1 = 0.5  # mass of the cart
    m2 = 0.5  # mass of the pendulum
    l = 0.5   # length of the pendulum
    b = 0.1   # friction coefficient
    g = 9.81  # acceleration of gravity
        
    den = 4*(m1+m2)-3*m2*np.cos(theta)**2
    
    dydt = [x_dot,
            (2*m2*l*theta_dot**2*np.sin(theta)+3*m2*g*np.sin(theta)*np.cos(theta)+4*u-4*b*x_dot)/den,
            theta_dot,
            (-3*m2*l*theta_dot**2*np.sin(theta)*np.cos(theta)-6*(m1+m2)*g*np.sin(theta)-6*(u-b*x_dot)*np.cos(theta))/(l*den)]
    return dydt

