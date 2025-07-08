import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *
import mpmath as mp

import sys 
sys.path.append('../..')

from utils.symbolic import sym_even_legendre_series, sym_sigma_y, mp_vectorize

# Set up symbols -- we let s be dummy varaible integrated in superposition equation
x,s = symbols('x s')
y,a_lim,m = symbols('y a_lim m', real = True, positive=True)
num_coeff_max=6 # Use 6 term (even) legendre polynomials
c = symbols(f'c_1:{num_coeff_max+1}', real = True)

p=sym_even_legendre_series(num_coeff_max) # Applied load

p_poly = Poly(p,s) 
print(f'Applied Load: {p}')
print(f'Load Magnitude: {simplify(integrate(p,(s,-a_lim,a_lim)))}') # check solution

# Solve Solution
soln = sym_sigma_y(p, polynomial=True)

# print(f'Sigma_y: {soln}')

# Save solution as a text file
f_name = 'legendre_soln.txt'
with open(f_name, 'w') as file:
    file.write(srepr(soln))

# reload file as check
with open(f_name, 'r') as file:
    loaded_soln_str = file.read()

loaded_soln = sympify(loaded_soln_str)
print(loaded_soln.free_symbols)

# Check equivalence between loaded soln and original solution (sanity check)
print(f'difference between loaded and orignal solution: {simplify(soln-loaded_soln, force=True)}')
