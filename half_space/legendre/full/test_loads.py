import numpy as np
import pandas as pd
from sympy import *
import matplotlib.pyplot as plt
import mpmath as mp


from symbolic import sym_legendre_series, sym_even_legendre_series, sym_sigma_y, mp_vectorize

mp.mp.dps = 70

num_coeff=3
num_samples = 10

# integrate

x,s = symbols('x s')
y,a_lim,m = symbols('y a_lim m', real = True, positive=True)
num_coeff_max=num_coeff # Use 6 term (even) legendre polynomials
c = symbols(f'c_1:{num_coeff_max+1}')


p = sym_even_legendre_series(num_coeff_max)

soln = sym_sigma_y(p, polynomial=True)

#-----------Lamdify and vectorize from numerical solution------------#
sigma_y = lambdify([*c,x,y,m,a_lim], soln, modules=['mpmath'])

sigma_y_vec = mp_vectorize(sigma_y)
#------------------Numerically Evaluate function---------------------#
a=100
m=1

rng = np.random.default_rng(0)

# sample coeffcients
coeffs_n = rng.uniform(high=10,low=-10, size=(num_samples,num_coeff))
coeffs=np.zeros((num_samples,num_coeff_max)) # max number of coefficients
coeffs[:,:num_coeff] = coeffs_n

x_nums =0.0
y_nums = np.logspace(-4,5,27)

plt.figure()
plt.title('at x=0')
sigma_y_res0 = sigma_y_vec(*coeffs[0], x_nums, y_nums,m, a).astype(float)
for coeff in coeffs:
    sigma_y_res = sigma_y_vec(*coeff, x_nums, y_nums,m, a).astype(float)
    plt.loglog(y_nums, np.abs(sigma_y_res-sigma_y_res0))
plt.xlabel('Depth')
plt.ylabel(r'$\sigma_{yy}$')

plt.show()