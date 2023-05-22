import jax.numpy as np
from jax import random

def generate_psd_params(n,N):
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    diag = random.uniform(subkey, shape=(n,1))[:,0] + n
    key, subkey = random.split(key)
    off_diag = random.uniform(subkey, shape=(int( (n**2-n)/2.0 ),1))[:,0]
    params = np.append(diag, off_diag, axis = 0).reshape(1,-1)
    for i in range(1,N):
        # Diagonal elements
        key, subkey = random.split(key)
        params_temp = random.uniform( subkey, shape=( 1,int(n + (n**2 -n)/2.0)) )
        params = np.append( params, params_temp, axis = 0 )    
    return params