import jax.numpy as np

def wrap_angle(angle):
    return np.arctan2( np.sin(angle), np.cos(angle) )