import jax
import jax.numpy as jnp

def jax_central_difference(f, x, y, alpha=None, epsilon=1e-5, key=0, wrt='x'):
    """
    Approximate the directional derivative of f(x, y) with respect to either x or y using central difference
    in a random direction.

    Args:
        f: A scalar function f(x, y) where x and y are arrays.
        x: Input array 1
        y: Input array 2
        epsilon: Small scalar for finite difference.
        key: JAX PRNG key for randomness.
        wrt: 'x' or 'y', specifying which argument to differentiate with respect to.

    Returns:
        A tuple: (approx_derivative_scalar, direction_vector)
    """
    key = jax.random.PRNGKey(0)

    # Select the argument to differentiate with respect to
    if wrt == 'x':
        v = jax.random.normal(key, shape=x.shape)
        v = v / jnp.linalg.norm(v)
        f_plus = f(x + epsilon * v, y, alpha=alpha)
        f_minus = f(x - epsilon * v, y, alpha=alpha)
    elif wrt == 'y':
        v = jax.random.normal(key, shape=y.shape)
        v = v / jnp.linalg.norm(v)
        f_plus = f(x, y + epsilon * v, alpha=alpha)
        f_minus = f(x, y - epsilon * v, alpha=alpha)
    else:
        raise ValueError("wrt must be 'x' or 'y'")

    fd_approx = (f_plus - f_minus) / (2 * epsilon)

    return fd_approx, v