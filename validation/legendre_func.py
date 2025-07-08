import numpy as np
import scipy
from scipy.fft import ifft
import matplotlib.pyplot as plt
from sympy import *
from sympy import lambdify

import sys 
sys.path.append('..')

from utils.symbolic import sym_even_legendre_series, sym_sigma_y, mp_vectorize

# Set up symbols -- we let s be dummy varaible integrated in superposition equation
s = symbols('s', real=True)
a_lim = symbols('a_lim', real = True, positive=True)
num_coeff=3
c = symbols(f'c_1:{num_coeff+1}', real = True)
# Sample random coefficients for Legendre Polynomials

rng = np.random.default_rng(0)

plt.figure()

p=sym_even_legendre_series(num_coeff)

int_domain = integrate(p,(s,-a_lim, a_lim)) # should equal to 1 

print(simplify(int_domain)) # should equal to 1 by definition

#---Generate and visualize a bunch of even Legendre series

p_num = lambdify((c,s, a_lim), p, modules='numpy')

lim = 100
x = np.linspace(-lim,lim,1000)

for ii in range(10):
    coeffs = np.random.uniform(low = -1, high = 1, size=num_coeff)
    print(coeffs)
    
    plt.plot(x,p_num(coeffs, x,lim))
    
plt.show()