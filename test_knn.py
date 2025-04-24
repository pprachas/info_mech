import numpy as np
from sklearn.neighbors import NearestNeighbors
import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from utils.mi import knn_distances

# Create synthetic data
np.random.seed(0)
n = 100
x_np = np.random.randn(n, 10)
y_np = x_np + 0.5 * np.random.randn(n, 10)
xy_np = np.hstack([x_np, y_np])

# Use sklearn to find k-th neighbor distances
k = 3
nbrs = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev')  # L_inf norm = Chebyshev
nbrs.fit(xy_np)
distances, indices = nbrs.kneighbors(xy_np)

# Exclude self-distance (first column), take k-th neighbor
sklearn_kth_dist = distances[:, k]

# Now compare to JAX version


xy_jnp = jnp.array(xy_np)
jax_kth_dist = knn_distances(xy_jnp, k)

# Compare results
print("Sklearn k-th neighbor distances:\n", sklearn_kth_dist[:10])
print("JAX k-th neighbor distances:\n", jax_kth_dist[:10])

# Check difference
diff = np.abs(sklearn_kth_dist - np.array(jax_kth_dist))/sklearn_kth_dist
print("\nMax difference:", np.max(diff))
print("Mean difference:", np.mean(diff))