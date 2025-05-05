import numpy as np
import matplotlib.pyplot as plt
from sympy import *
# import custom scripts
import sys
sys.path.append('..')

from utils.analytical import sigma_y_point, sigma_y_uniform
from utils.symbolic_int import  sym_sigma_y


x_lin = np.linspace(-1000,1000,5000)
y_lin = np.linspace(0,1000,5000)

x_mesh,y_mesh = np.meshgrid(x_lin,y_lin)

#------uniform load solution--------#
P = 1
a = 500

plt.figure()
fig = plt.contourf(x_mesh,y_mesh,np.log(np.abs(sigma_y_uniform(P,a,x_mesh,y_mesh))))
fig.axes.yaxis.set_inverted(True)
plt.colorbar(fig)
plt.title('Uniform load (By hand)')

#--------Compare solutions-----#
a_lim = symbols('a_lim', real = True, positive=True)
x,y,s = symbols('x y s', real = True)
p = P/(2*a_lim)
soln = sym_sigma_y(p)
# params = sorted(list(soln.free_symbols))
# print(params)
sigma_y = lambdify([x,y,a_lim],soln, modules=['numpy'])

plt.figure()
fig = plt.contourf(x_mesh,y_mesh,np.log(np.abs(sigma_y(x_mesh,y_mesh,a))))
fig.axes.yaxis.set_inverted(True)
plt.colorbar(fig)
plt.title('Uniform load (Sympy)')

# # uniform load

y = np.linspace(0,1000,1000)
plt.figure()
plt.semilogy(y_lin,np.abs(sigma_y_uniform(P,a,0,y_lin)), label = 'analytical', marker = 's')
plt.title('Uniform load')
plt.xlabel(r'$y$-coordinate')
plt.ylabel(r'$\sigma_{yy}$')
plt.tight_layout()
plt.legend()

#---------Compare with FE solution------------#
fea_sigma_y = np.loadtxt('fenicsx_scripts/FEA_results.txt')
plt.semilogy(-fea_sigma_y[0,:],np.abs(fea_sigma_y[1,:]), label = 'fenicsx', marker='o', fillstyle=None, markersize = 3)
plt.semilogy(y_lin,np.abs(sigma_y(0,y_lin,a)), label = 'sympy')
plt.legend()
plt.savefig('compare_uniform.png')