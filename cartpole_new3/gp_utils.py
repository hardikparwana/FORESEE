import jax
jax.config.update("jax_enable_x64", True)
import optax as ox
import gpjax as jgp
import jaxkern as jk
from jaxutils import Dataset
import jax.numpy as np
from jax import jit, random
key = random.PRNGKey(2)

def initialize_gp(num_datapoints = 10):
    meanf = jgp.mean_functions.Zero()
    # kernel = jk.RBF() #* jk.Polynomial()
    # kernel = jk.RBF() * jk.RationalQuadratic()
    kernel = jk.RBF() * jk.Periodic()
    prior = jgp.Prior(mean_function=meanf, kernel = kernel)
    likelihood = jgp.Gaussian( num_datapoints=num_datapoints )
    posterior = prior * likelihood
    parameter_state = jgp.initialise(
        posterior, key#, kernel={"lengthscale": np.array([0.5])}
    )
    return likelihood, posterior, parameter_state

def train_gp(likelihood, posterior, parameter_state, train_x, train_y):
    D = Dataset(X=train_x, y=train_y)
    negative_mll = jit(posterior.marginal_log_likelihood(D, negative=True))
    optimiser = ox.adam(learning_rate=0.08)
    inference_state = jgp.fit(
        objective=negative_mll,
        parameter_state=parameter_state,
        optax_optim=optimiser,
        num_iters=1500,
    )
    learned_params, training_history = inference_state.unpack()
    return likelihood, posterior, learned_params, D

def predict_gp(likelihood, posterior, learned_params, D, test_x):
    latent_dist = posterior(learned_params, D)(test_x)
    predictive_dist = likelihood(learned_params, latent_dist)
    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()
    return predictive_mean, predictive_std

def initialize_gp_prediction_distribution(gp_params, train_x, train_y):
    meanf = jgp.mean_functions.Zero()
    # kernel = jk.RBF() #* jk.Polynomial()
    # kernel = jk.RBF() * jk.RationalQuadratic()
    kernel = jk.RBF() * jk.Periodic()
    prior = jgp.Prior(mean_function=meanf, kernel = kernel)
    likelihood = jgp.Gaussian( num_datapoints=train_x.shape[0] )
    posterior = prior * likelihood
    D = Dataset(X=train_x, y=train_y)
    latent_dist = posterior(gp_params, D)
    return latent_dist

def initialize_gp_prediction(gp_params, train_x, train_y):
    meanf = jgp.mean_functions.Zero()
    # kernel = jk.RBF() #* jk.Polynomial()
    # kernel = jk.RBF() * jk.RationalQuadratic()
    kernel = jk.RBF() * jk.Periodic()
    prior = jgp.Prior(mean_function=meanf, kernel = kernel)
    likelihood = jgp.Gaussian( num_datapoints=train_x.shape[0] )
    posterior = prior * likelihood
    D = Dataset(X=train_x, y=train_y)
    latent_dist = posterior(gp_params, D)
    return latent_dist

def predict_with_gp_params(gp_params, train_x, train_y, test_x):
    meanf = jgp.mean_functions.Zero()
    # kernel = jk.RBF() #* jk.Polynomial()
    # kernel = jk.RBF() * jk.RationalQuadratic()
    kernel = jk.RBF() * jk.Periodic()
    prior = jgp.Prior(mean_function=meanf, kernel = kernel)
    likelihood = jgp.Gaussian( num_datapoints=train_x.shape[0] )
    posterior = prior * likelihood
    D = Dataset(X=train_x, y=train_y)
    latent_dist = posterior(gp_params, D)(test_x)
    predictive_dist = likelihood(gp_params, latent_dist)
    predictive_mean = predictive_dist.mean()
    predictive_var = predictive_dist.stddev()**2
    return predictive_mean, predictive_var