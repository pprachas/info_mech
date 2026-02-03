import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *
import mpmath as mp

import sys 

from symbolic import sym_even_legendre_series, sym_legendre_series, sym_sigma_y, mp_vectorize

plt.style.use('jeff_style.mplstyle')

# Set up symbols -- we let s be dummy varaible integrated in superposition equation
x,s = symbols('x s')
y,a_lim,m = symbols('y a_lim m', real = True, positive=True)
num_coeff_max=6 # Use 6 term (even) legendre polynomials
c = symbols(f'c_1:{num_coeff_max+1}', real = True)

#--------Get Even Legendre Load----------------#
p=sym_even_legendre_series(num_coeff_max) # Applied load
load = lambdify([c,s,a_lim,m], p, modules=['numpy']) # using numpy is fine here

a=100
m_num=1
num_coeff = 5

# Load coefficients
coeffs = np.loadtxt(f'legendre/even/coeffs/legendre_coeffs{num_coeff}.txt')

x_load = np.linspace(-a,a,100)
colors = plt.cm.gray(np.linspace(0,1,40))

plt.figure(figsize=(3,3))

for ii in range(30):
    plt.plot(x_load,load(coeffs[ii],x_load,a,m_num), c=colors[ii])

plt.axis('off')
plt.savefig(f'loads_even{num_coeff}.pdf')

#--------Get full Legendre Load----------------#
p=sym_legendre_series(num_coeff_max) # Applied load
load = lambdify([c,s,a_lim,m], p, modules=['numpy']) # using numpy is fine here

# Load coefficients
coeffs = np.loadtxt(f'legendre/full/coeffs/legendre_coeffs{num_coeff}.txt')

a=100
m_num=100
x_load = np.linspace(-a,a,100)
colors = plt.cm.gray(np.linspace(0,1,40))

plt.figure(figsize=(3,3))

for ii in range(30):
    plt.plot(x_load,load(coeffs[ii],x_load,a,m_num), c=colors[ii])

plt.axis('off')
plt.savefig(f'loads_full{num_coeff}.pdf')

plt.show()
