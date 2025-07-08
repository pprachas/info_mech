import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *

import sys 
sys.path.append('..')

from utils.symbolic import sym_even_legendre_series, sym_sigma_y

# Set up symbols -- we let s be dummy varaible integrated in superposition equation

x,y,s,a_lim = symbols('x y s a_lim')

p1 = s**2
p2 = 1
p= p1+p2

a=100

print(f'Applied Load: {p}')
print(f'Load Magnitude: {simplify(integrate(p,(s,-a_lim,a_lim)))}')


soln_poly = sym_sigma_y(p, polynomial=False).rewrite(atan)

# soln_poly_split = sym_sigma_y(p, polynomial=True)

soln_p1 = sym_sigma_y(p1, polynomial=False)
soln_p2 = sym_sigma_y(p2, polynomial=False)


# check solution

diff = soln_p1+soln_p2-soln_poly

print(f'Difference between solutions:')
print(simplify(diff))

print('solution:')
print(simplify(soln_poly.subs({x:0, a_lim:1})))
print(simplify(soln_poly.subs({x:0, a_lim:1})).as_real_imag())

soln_poly_num = lambdify([x,y,a_lim], soln_poly)


x_lin = np.linspace(-1000,1000,100)
y_lin = np.linspace(0,5000,100)

x_mesh,y_mesh = np.meshgrid(x_lin,y_lin)

plt.figure()
plt.title('Full solution')
fig=plt.contourf(x_mesh,y_mesh,soln_poly_num(x_mesh,y_mesh,a))
fig.axes.yaxis.set_inverted(True)
plt.savefig('full.png')

# at x=0
y_log=np.logspace(-3,9,1000)
plt.figure()
plt.title('at x=0')
fig=plt.loglog(y_log,np.abs(soln_poly_num(0,y_log,a)))

plt.savefig('poly_x0.png')

plt.show()