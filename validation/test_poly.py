import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *
from sympy.integrals.rationaltools import ratint
import matplotlib.pyplot as plt
import mpmath as mp

import sys 
sys.path.append('..')

from utils.symbolic import sym_even_legendre_series, sym_sigma_y, complex_to_atan, mp_vectorize

# Set up symbols -- we let s be dummy varaible integrated in superposition equation
x,s = symbols('x s')
y,a_lim = symbols('y a_lim',postive=True)
mp.mp.dps=50


p= s**6
integrand = p/(y**2+(x-s)**2)**2

soln = simplify((-2/pi)*(y)**3*Integral(integrand,(s)))
soln=simplify(soln.doit())
# soln_def = soln.subs(s,a_lim)- soln.subs(s,-a_lim)
#soln_int=soln.simplify()# .doit(hint='ratint')

soln= (expand_log(soln, force=True))

print(f'complex form:{soln}')


soln_atan = complex_to_atan(soln)

print(f'real form:{soln_atan}')

soln = simplify(soln.subs(s,a_lim) - soln.subs(s,-a_lim)) # definite integral
soln_atan = simplify(soln_atan.subs(s,a_lim) - soln_atan.subs(s,-a_lim)) # definite integral

print(soln_atan.evalf(subs={x:0, y:1e6, a_lim:1}))
print(f'at x=0 and a_lim=1:{simplify(soln_atan)}')

# Lambdify function with numpy
soln_num = lambdify([x,y,a_lim], soln, modules='numpy')
soln_atan_num = lambdify([x,y,a_lim], soln_atan, modules='numpy')

x_num= 0
y_num = np.logspace(-3,6,1000)

plt.figure()
plt.loglog(y_num,np.abs(np.real(soln_num(x_num,y_num,1))), marker = '.', label = 'complex solution')
plt.loglog(y_num,np.abs(soln_atan_num(x_num,y_num,1)), label = 'atan solution')
plt.title('Numpy solution')
plt.xlabel('Depth')
plt.ylabel(r'$\sigma_{yy}$')
plt.legend()
plt.savefig('compare_atan.png')

# plt.figure()
# plt.plot(y,np.abs(np.imag(soln_num(x,y,1))))
# plt.title('Complex Component')
# plt.xlabel('Depth')
# plt.ylabel(r'$\sigma_{yy}$')

# use mpmath
soln_num = lambdify([x,y,a_lim], soln.n(50), modules='mpmath')
soln_atan_num = lambdify([x,y,a_lim], soln_atan.n(50), modules='mpmath')

soln_num_vec = mp_vectorize(soln_num)
soln_atan_num_vec = mp_vectorize(soln_atan_num)

plt.figure()
plt.loglog(y_num,np.abs(np.real(soln_num_vec(x_num,y_num,1))), marker = '.', label = 'complex solution')
plt.loglog(y_num,np.abs(soln_atan_num_vec(x_num,y_num,1)), label = 'atan solution')
plt.title('MP math solution')
plt.xlabel('Depth')
plt.legend()
plt.ylabel(r'$\sigma_{yy}$')
plt.savefig('mp_math_soln.png')

print(soln.evalf(subs={x:0, y:1e6, a_lim:1}))
print(soln_atan.evalf(subs={x:0, y:1e6, a_lim:1}))
print(soln_atan_num(mp.mpf(0),mp.mpf(1e6),mp.mpf(1)))
plt.show()

