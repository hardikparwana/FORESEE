import jax
from jax import grad
import jax.numpy as jnp


# def f1(x):
#     return jax.scipy.linalg.sqrtm(x)[0,0]

def f1(x):
    return jnp.sqrt(x)[0,0]

f1grad = grad(f1)

print("hello")
f1(jnp.array([[0.0]]))

print(f1grad(jnp.array([[0.0]])))