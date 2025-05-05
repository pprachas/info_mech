import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jax.scipy.special import digamma
from jax.nn import softmax

def pairwise_distances(x, y, ord=jnp.inf):
    """Compute pairwise distances between points in x and y."""
    return norm(x[:, None, :] - y[None, :, :], ord=ord, axis=-1)

def knn_distances(x, k, ord=jnp.inf, alpha=None):
    """
    Compute the distance to the k-th nearest neighbor for each point in x,
    using the specified norm. Ignores self-distances.
    """
    n = x.shape[0]
    dists = pairwise_distances(x,x, ord=ord)
    dists = dists + jnp.eye(n) * 1e10  # mask self

    return jnp.sort(dists)[:,k-1]

def add_noise(x, noise_scale=1e-10, key=None):
    """
    Adds Gaussian noise to an input array.

    Args:
    - x: array to which noise is added
    - noise_scale: standard deviation of the noise
    - key: JAX PRNGKey (required)

    Returns:
    - Noisy version of x
    """

    noise = noise_scale * jax.random.normal(key, shape=x.shape)
    return x + noise

def ksg(x, y, k=3, ord=jnp.inf, key=None, alpha=None):
    """
    Kraskov-Stögbauer-Grassberger mutual information estimator.
    Differentiable and uses general norm (default L_inf).
    
    Args:
        x: Input vector
        y: Output vector
        k: number of nearest neighbor
        ord: order of L_p norm
        key: random key for noise
        alpha: parameter to sharpen sigmoid for soft thresholding. None to use "hard" threshold
    """
    n = x.shape[0]
    if key != None: # add noise if needed (recommended by KSG paper)
        x = add_noise(x,key=key)
        y = add_noise(y,key=key)

    # Joint space
    xy = jnp.concatenate([x, y], axis=1)

    # Get distances to k-th neighbor in joint space
    eps = knn_distances(xy, k, ord=ord, alpha=alpha)

    # Get pairwise distances in marginal spaces
    dx = pairwise_distances(x, x, ord=ord) + jnp.eye(n) * 1e10
    dy = pairwise_distances(y, y, ord=ord) + jnp.eye(n) * 1e10

    # Count neighbors within epsilon (exclude equality due to strictness)
    if alpha == None:
        # Perform hard (non-differentiable threshold)
        nx = jnp.sum(dx < eps[:, None], axis=1)
        ny = jnp.sum(dy < eps[:, None], axis=1)

    else:
        nx = jnp.sum(jax.nn.sigmoid(-alpha*(dx-eps[:, None])),axis=1)
        ny = jnp.sum(jax.nn.sigmoid(-alpha*(dy-eps[:, None])),axis=1)

    # Mutual Information Estimation
    mi = digamma(k) + digamma(n) - jnp.mean(digamma(nx + 1) + digamma(ny + 1))
    return mi