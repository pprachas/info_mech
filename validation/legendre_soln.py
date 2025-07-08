import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *
import mpmath as mp

import sys 
sys.path.append('..')

from utils.symbolic import sym_even_legendre_series, sym_sigma_y, mp_vectorize

mp.mp.dps=50

# plot settings
plt.style.use('jeff_style.mplstyle')

# Set up symbols -- we let s be dummy varaible integrated in superposition equation
x,s = symbols('x s')
y,m,a_lim = symbols('y m a_lim', real = True, positive=True)

num_coeff=4
c = symbols(f'c_1:{num_coeff+1}', real = True)
# Sample random coefficients for Legendre Polynomials

rng = np.random.default_rng(0)
coeffs = rng.uniform(high=10,low=-10, size=num_coeff)

x_lin = np.linspace(-200,200,100)
y_lin = np.linspace(1e-6,400,100)

x_mesh,y_mesh = np.meshgrid(x_lin,y_lin)

p=sym_even_legendre_series(num_coeff) # Applied load


p_poly = Poly(p,s) 
print(f'Applied Load: {p}')
print(f'Load Magnitude: {simplify(integrate(p,(s,-a_lim,a_lim)))}')


soln = sym_sigma_y(p, polynomial=True)

print(f'Sigma_y: {soln}')

sigma_y_leg = lambdify([*c,x,y,a_lim,m], soln, modules=['mpmath'])
load = lambdify([c,s,a_lim,m], p, modules=['numpy']) # using numpy is fine here

sigma_y_leg_vec = mp_vectorize(sigma_y_leg)
a=100
m_num=100

x_load = np.linspace(-a,a,100)

# Plot load
plt.figure()
plt.plot(x_load,load(coeffs,x_load,a,m_num))

# Plot sigma_yy
plt.figure(figsize=(3,3))
fig=plt.contourf(x_mesh,y_mesh,sigma_y_leg_vec(*coeffs,x_mesh,y_mesh,a, m_num).astype(float), cmap = 'OrRd')
fig.axes.yaxis.set_inverted(True)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.colorbar(fig)
plt.tight_layout()
plt.savefig('sigma_y.pdf')

plt.show()