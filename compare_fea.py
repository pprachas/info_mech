import numpy as np
import matplotlib.pyplot as plt

from utils.analytical import sigma_y_point, sigma_y_uniform


x = np.linspace(-1000,1000,5000)
y = np.linspace(0,1000,5000)

x,y = np.meshgrid(x,y)

#-------Point load solution-------#
P = 1.0

plt.figure()
fig = plt.contourf(x,y,np.log(-sigma_y_point(P,x,y)))
fig.axes.yaxis.set_inverted(True)
plt.colorbar(fig)
plt.title('Normal force point load')

#------uniform load solution--------#
P = 1.0
a = 500.0

plt.figure()
fig = plt.contourf(x,y,np.log(np.abs(sigma_y_uniform(P,a,x,y))))
fig.axes.yaxis.set_inverted(True)
plt.colorbar(fig)
plt.title('Uniform load')

#--------Compare with FE solution-----#
# point load 
sigma_point_y_FE = np.loadtxt('point_load.txt')
y_fea = np.linspace(0,1000,1000)
y = np.linspace(0,1000,1000)

plt.figure()
plt.semilogy(y,-sigma_y_point(P,0,y), label = 'analytical', marker = 's')
plt.semilogy(y_fea[1:],-sigma_point_y_FE[1:], label = 'FE')
plt.title('Point load')
plt.xlabel(r'$y$-coordinate')
plt.ylabel(r'$\sigma_{yy}$')
plt.legend()

# uniform load
sigma_point_y_FE = np.loadtxt('uniform_load.txt')
y_fea = np.linspace(0,1000,1000)
y = np.linspace(0,1000,1000)
plt.figure()
plt.semilogy(y,np.abs(sigma_y_uniform(P,a,0,y)), label = 'analytical', marker = 's')
plt.semilogy(y_fea,-sigma_point_y_FE, label = 'FE')
plt.title('Uniform load')
plt.xlabel(r'$y$-coordinate')
plt.ylabel(r'$\sigma_{yy}$')
plt.tight_layout()
plt.legend()


plt.show()