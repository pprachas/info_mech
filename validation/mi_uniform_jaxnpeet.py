import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import jax
import jax.numpy as jnp
from npeet import entropy_estimators as ee
from sklearn.preprocessing import minmax_scale, StandardScaler
# import custom scripts
import sys
sys.path.append('..')
from utils.symbolic_int import  sym_sigma_y
from utils.analytical import sigma_y_uniform
from utils.mi import ksg


P = 1.0
x_num = 0
y_num = -np.logspace(1,8,500)
a = 100.0

# Sample uniform load
k=3
np.random.seed(0)
a = np.random.uniform(0,100,1000)
sigma_y = []

# get Lamdified analytical solution
a_lim = symbols('a_lim', real = True, positive=True)
x,y,s = symbols('x y s', real = True)
p = P/(2*a_lim)
soln = sym_sigma_y(p)
sigma_y = lambdify([x,y,a_lim],soln, modules=['numpy'])


mi_npeet = []
mi_jax = []
for ii in range(len(y_num)):
    sigma_y_num=sigma_y(x_num,y_num[ii],a)

    #---------Scale data----------#
    # Need this step due to KNN on joint space
    a_scaled = minmax_scale(a[:,None])
    sigma_y_num_scaled = minmax_scale(sigma_y_num[:,None])
    # NPEET MI estimator 
    mi_npeet.append(ee.mi(a_scaled.tolist(), sigma_y_num_scaled.tolist(), k=k, base=np.e))

    key = jax.random.key(0)
    a_jax = jnp.array(a_scaled)
    sigma_y_jax = jnp.array(sigma_y_num_scaled)
    # for small random noise
    mi_jax.append(ksg(a_jax, sigma_y_jax, k=k, key=key))


#mi_jax = np.array(mi_jax)

plt.figure()
plt.semilogx(np.abs(y_num), mi_jax, marker = '.', label = 'MI JAX')
plt.semilogx(np.abs(y_num), mi_npeet, label = 'MI NPEET')
plt.xlabel('Depth of block')
plt.ylabel('Mutual Information')
plt.legend()
plt.show()