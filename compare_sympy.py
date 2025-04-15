import numpy as np
import matplotlib.pyplot as plt
from sympy import *

from utils.analytical import sigma_y_point, sigma_y_uniform
from utils.symbolic_int import  sym_sigma_y


x = np.linspace(-1000,1000,5000)
y = np.linspace(0,1000,5000)

x,y = np.meshgrid(x,y)

#------uniform load solution--------#
P = 1
a = 500

plt.figure()
fig = plt.contourf(x,y,np.log(np.abs(sigma_y_uniform(P,a,x,y))))
fig.axes.yaxis.set_inverted(True)
plt.colorbar(fig)
plt.title('Uniform load (By hand)')

#--------Compare with sympy solution-----#
p = P/(2*a)

plt.figure()
fig = plt.contourf(x,y,np.log(np.abs(sym_sigma_y(p,x,y,a))))
fig.axes.yaxis.set_inverted(True)
plt.colorbar(fig)
plt.title('Uniform load (Sympy)')

# # uniform load

y = np.linspace(0,1000,1000)
plt.figure()
plt.semilogy(y,np.abs(sigma_y_uniform(P,a,0,y)), label = 'analytical', marker = 's')
plt.semilogy(y,np.abs(sym_sigma_y(p,0,y,a)), label = 'sympy')
plt.title('Uniform load')
plt.xlabel(r'$y$-coordinate')
plt.ylabel(r'$\sigma_{yy}$')
plt.tight_layout()
plt.legend()


# plt.show()

plt.show()