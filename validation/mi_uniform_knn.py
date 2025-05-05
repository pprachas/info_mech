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
ks = [3,5,10]
np.random.seed(0)
a = np.random.uniform(0,100,1000)
sigma_y = []

# get Lamdified analytical solution
a_lim = symbols('a_lim', real = True, positive=True)
x,y,s = symbols('x y s', real = True)
p = P/(2*a_lim)
soln = sym_sigma_y(p)
sigma_y = lambdify([x,y,a_lim],soln, modules=['numpy'])

scaler = StandardScaler()

plt.figure()
for k in ks:
    mi_npeet = []
    mi_npeet_standard = []
    for ii in range(len(y_num)):
        sigma_y_num=sigma_y(x_num,y_num[ii],a)
        #---------Scale data----------#
        # min-max scaling
        a_scaled_minmax = minmax_scale(a[:,None])
        sigma_y_num_scaled_minmax = minmax_scale(sigma_y_num[:,None])

        # standard scaling 
        a_scaled_standard = scaler.fit_transform(a[:,None])
        sigma_y_num_scaled_standard = scaler.fit_transform(sigma_y_num[:,None])

        # NPEET MI estimator
        mi_npeet.append(ee.mi(a_scaled_minmax.tolist(), sigma_y_num_scaled_minmax.tolist(), k=k, base=np.e))
        mi_npeet_standard.append(ee.mi(a_scaled_standard.tolist(), sigma_y_num_scaled_standard.tolist(), k=k, base=np.e))
    
    plt.semilogx(np.abs(y_num), mi_npeet_standard, label = f'k={k}; standard scaling', marker = '.')

    plt.semilogx(np.abs(y_num), mi_npeet, label = f'k={k}; min-max scaling')

plt.xlabel('Depth of block')
plt.ylabel('Mutual Information')
plt.legend()
plt.show()