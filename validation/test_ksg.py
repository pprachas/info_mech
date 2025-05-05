import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from npeet import entropy_estimators as ee
from jax import grad
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt

# import custom scripts
import sys
sys.path.append('..')
from utils.mi import ksg

from utils.grad_check import jax_central_difference

jax.config.update("jax_enable_x64", True) # tell jax to use float64

# Example: x has 1D, y has 10D
n = 250 # Number of data points
x = np.random.randn(n, 10)  # 1D x
y = np.hstack([x] + [np.random.randn(n, 1)])  # 10D y

x = minmax_scale(x)
y = minmax_scale(y)

k=5
alphas = np.logspace(2,8,7)

# Convert to JAX arrays
x_jax = jnp.array(x)
y_jax = jnp.array(y)

# Calculate the MI using the ksg1_mi function (your current implementation)
key = random.key(0)
 # for small random noise

npeet_mi=ee.mi(x.tolist(), y.tolist(), k=k, base=np.e)
percent_diff = []
for alpha in alphas:
    mi = ksg(x_jax, y_jax, k=k, key=key, alpha=alpha)

    # Print the result to check for unexpected values

    # Compute percent difference
    percent_diff.append(np.abs((mi-npeet_mi)/npeet_mi)*100)
    print(f'alpha:{alpha}, percent:{np.abs((mi-npeet_mi)/npeet_mi)*100}')


    #-------------Test adjoint-------------------#
    eps_all = np.logspace(-12,-5,9)
    grad_mi_finite_all = []
    grad_mi_autodiff_all = []

    for eps in eps_all:
        grad_mi_finite,v = jax_central_difference(ksg, x_jax, y_jax, alpha=alpha, wrt='y', key=None, epsilon=eps)
        grad_mi_autodiff = jnp.vdot(grad(ksg,argnums=1)(x_jax,y_jax, alpha=alpha),v)

        percent_diff_grad = np.abs((grad_mi_finite-grad_mi_autodiff)/grad_mi_autodiff)*100

        grad_mi_finite_all.append(grad_mi_finite)
        grad_mi_autodiff_all.append(grad_mi_autodiff)

    grad_mi_finite_all = np.array(grad_mi_finite_all)
    grad_mi_autodiff_all = np.array(grad_mi_autodiff_all)

    percent_diff_grad = np.abs((grad_mi_finite_all-grad_mi_autodiff_all)/grad_mi_autodiff_all)*100

    # Plot comparison of finite difference and autograd
    plt.figure(figsize=(5,5))
    plt.semilogx(eps_all,percent_diff_grad, marker='s')
    plt.xlabel('h')
    plt.ylabel('Percent error between autodiff and finite difference')

    plt.title(rf'$\alpha = 10^{int(np.log10(alpha))}$')

# Plot percent difference 
plt.figure(figsize=(5,5))
plt.semilogx(alphas,percent_diff, marker='s')
plt.xlabel(r'$\alpha$')
plt.ylabel('Percent error between smooth and non-smooth ksg')

plt.title(rf'Percent Error')

plt.show()