import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jax.scipy.special import digamma

def pairwise_distances(x, y, ord=jnp.inf):
    """Compute pairwise distances between points in x and y."""
    return norm(x[:, None, :] - y[None, :, :], ord=ord, axis=-1)

def knn_distances(x, k, ord=jnp.inf):
    """
    Compute the distance to the k-th nearest neighbor for each point in x,
    using the specified norm. Ignores self-distances.
    """
    n = x.shape[0]
    print(x.shape)
    dists = pairwise_distances(x,x, ord=ord)
    dists = dists + jnp.eye(n) * 1e10  # mask self
    val, _ = jax.lax.top_k(-dists, k)  # Use -dist for smallest
    val_d = -val[:, -1]
    return val_d

def add_noise(x, noise_scale=1e-10, key=None):
    """
    Adds Gaussian noise to an input array.

    Parameters:
    - x: array to which noise is added
    - noise_scale: standard deviation of the noise
    - key: JAX PRNGKey (required)

    Returns:
    - Noisy version of x
    """

    noise = noise_scale * jax.random.normal(key, shape=x.shape)
    return x + noise

def ksg(x, y, k=3, ord=jnp.inf, key=None):
    """
    Kraskov-Stögbauer-Grassberger mutual information estimator.
    Differentiable and uses general norm (default L_inf).
    """
    n = x.shape[0]
    if key != None: # add noise if needed (recommended by KSG paper)
        x = add_noise(x,key=key)
        y = add_noise(y,key=key)

    # Joint space
    xy = jnp.concatenate([x, y], axis=1)

    # Get distances to k-th neighbor in joint space
    eps = knn_distances(xy, k, ord=ord) - 1e-15  # Make it strictly less

    # Get pairwise distances in marginal spaces
    dx = pairwise_distances(x, x, ord=ord) + jnp.eye(n) * 1e10
    dy = pairwise_distances(y, y, ord=ord) + jnp.eye(n) * 1e10

    # Count neighbors within epsilon (exclude equality due to strictness)
    nx = jnp.sum(dx < eps[:, None], axis=1)
    ny = jnp.sum(dy < eps[:, None], axis=1)

    # Mutual Information Estimation
    mi = digamma(k) + digamma(n) - jnp.mean(digamma(nx + 1) + digamma(ny + 1))
    return mi