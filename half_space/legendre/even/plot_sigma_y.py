import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sympy import *
from symbolic import sym_even_legendre_series

# ------------------------- Configuration -------------------------#
num_coeff = 3
num_samples = 5000
num_sensors = 4  # Number of sensors to select
a = 100  # Normalization factor for coordinates

# ------------------------- Load Data -------------------------#
# Load stress data from CSV and reshape
df = pd.read_csv(f'sigma_y/legendre_coeffs{num_coeff}.csv', index_col=[0, 1], header=[0])
sigma_y = df.to_numpy().reshape(num_samples, -1, df.shape[1]).transpose(2, 1, 0)

print(sigma_y.shape)

# Extract spatial coordinates
x = df.columns.to_numpy().astype(float)
y = df.index.get_level_values(1)[:sigma_y.shape[1]].to_numpy()

x_grid, y_grid = np.meshgrid(x/a, y/a)

# plot stress
for ii in range(3):
    fig,ax=plt.subplots(figsize=(1.0, 1.5))
    plt.contourf(x/a,y/a, sigma_y[:,:,ii].T, cmap = 'Greys')
    # plt.contour(x/a,y/a, sigma_y[:,:,ii].T, colors = (0.5,0.0,0.0), linewidths = 0.2)
    plt.plot(x_grid,y_grid, marker = '.', ls = 'none', c=(0.75,0.5,0.5), markersize=1.75, mew = 0.0)
    ax.invert_yaxis() # only invert one since that are all shared
    ax.set_yscale('log')
    ax.axis('off')
    plt.savefig(f'halfspace_sigma{ii}.pdf')

# plot load with (fake) elastic body
x,s = symbols('x s')
y,a_lim,m = symbols('y a_lim m', real = True, positive=True)
num_coeff_max=6 # Use 6 term (even) legendre polynomials
c = symbols(f'c_1:{num_coeff_max+1}', real = True)

#--------Get Even Legendre Load----------------#
p=sym_even_legendre_series(num_coeff_max) # Applied load
load = lambdify([c,s,a_lim,m], p, modules=['numpy']) # using numpy is fine here

x_load = np.linspace(-a,a,100)
# Load coefficients
coeffs = np.loadtxt(f'coeffs/legendre_coeffs{num_coeff}.txt')
m_num = 1.0
for ii in range(3):
    fig,ax=plt.subplots(figsize=(1.0, 1.5))
    rect = Rectangle((-200,-400), 400,400, color = '0.6', ec = 'k')
    ax.add_patch(rect)
    plt.plot(x_load,load(coeffs[ii],x_load,a,m_num)*20, c= (0.5,0.0,0.0), lw = 0.75)
    plt.fill_between(x_load,load(coeffs[ii],x_load,a,m_num)*20,0, 
    color = (0.9,0.5,0.5), hatch = '||||||', edgecolor=(0.5,0.0,0.0), alpha = 0.6)
    ax.axis('off')

    plt.savefig(f'halfspace_load{ii}.pdf')


plt.show()