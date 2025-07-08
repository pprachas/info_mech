import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *
from sympy import lambdify
import time
impport mpmath as mp

import sys 
sys.path.append('..')

from utils.symbolic import sym_even_legendre_series, sym_sigma_y

# Set up symbols -- we let s be dummy varaible integrated in superposition equation
x, s = symbols('x s')
y, a_lim = symbols('y a_lim', positive=True)
num_coeff=2
c = symbols(f'c_1:{num_coeff+1}')
# Sample random coefficients for Legendre Polynomials

rng = np.random.default_rng(0)

plt.figure()

p=sym_even_legendre_series(num_coeff) # Applied load

print(p.free_symbols)
p_poly = Poly(p,s, extension=True) 
print(f'Applied Load: {p}')
print(f'Applied Load: {p_poly.as_expr()}')

# Integration by splitting polynomial
start = time.time()
sigma_y_split = sym_sigma_y(p, polynomial=True)
end = time.time()


print(f'Sigma_y split solution: {sigma_y_split}')
print(f'Code time: {end-start} seconds ')

# Integration not splitting
start = time.time()
sigma_y_full = sym_sigma_y(p, polynomial=False)
end = time.time()

print(f'Sigma_y full solution: {sigma_y_full}')
print(f'Code time: {end-start} seconds ')

# Check if both forms are equivalent
print(f'difference between 2 solutions: {simplify(sigma_y_full - sigma_y_split)}')


# Plot solution
rng = np.random.default_rng(0)
coeffs = rng.uniform(high=1,low=-1, size=num_coeff)


x_lin = np.linspace(-1000,1000,1000)
y_lin = np.linspace(0,2000,1000)

x_mesh,y_mesh = np.meshgrid(x_lin,y_lin)
a=500

sigma_y_leg = lambdify([c,x,y,a_lim], sigma_y_split, modules=['numpy'])

plt.plot(y_lin,np.real(sigma_y_leg(coeffs,0,y_lin,a)), label = 'Real Solution')
plt.plot(y_lin,np.imag(sigma_y_leg(coeffs,0,y_lin,a)), label = 'Imaginary Solution')

plt.figure()
fig=plt.contourf(x_mesh,y_mesh,(sigma_y_leg(coeffs,x_mesh,y_mesh,a)))
fig.axes.yaxis.set_inverted(True)
plt.colorbar(fig)

plt.show()