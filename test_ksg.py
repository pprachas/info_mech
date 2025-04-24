import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from utils.mi import ksg
from npeet import entropy_estimators as ee
jax.config.update("jax_enable_x64", True)

# Example: x has 1D, y has 10D
n = 100  # Number of data points
x = np.random.randn(n, 10)  # 1D x
y = np.hstack([x, x**2, np.sin(x)] + [0.1 * np.random.randn(n, 10)])  # 10D y
k=2
# Convert to JAX arrays
x_jax = jnp.array(x)
y_jax = jnp.array(y)

# Calculate the MI using the ksg1_mi function (your current implementation)
key = random.key(0)
 # for small random noise
mi = ksg(x_jax, y_jax, k=k, key=key)

# Print the result to check for unexpected values

npeet_mi=ee.mi(x.tolist(), y.tolist(), k=k, base=np.e)
# use bmi's estimator 
print(f"Estimated MI my implementation: {mi}")
print(f"Estimate by npeet: {npeet_mi}")

percent_diff = np.abs((mi-npeet_mi)/npeet_mi)*100

print(f'Percent DIfference: {percent_diff} %')
