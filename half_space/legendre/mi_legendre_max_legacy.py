import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *
import mpmath as mp
import pandas as pd
from pathlib import Path
from npeet import entropy_estimators as ee
from sklearn.preprocessing import StandardScaler

import sys 
sys.path.append('../..')

from utils.symbolic import sym_even_legendre_series, sym_sigma_y, mp_vectorize

plt.style.use('jeff_style.mplstyle')

num_coeff=2
num_samples = 5000 # number of load samples

df = pd.read_csv(f'sigma_y/legendre_coeffs{num_coeff}.csv', index_col=[0,1], header=[0])

sigma_y = df.to_numpy()
# change index to (x,y,load)
sigma_y=sigma_y.reshape(num_samples,-1,sigma_y.shape[1]).transpose(2,1,0) 

# get x and y data values
x=df.columns.to_numpy().astype(float)
y=df.index.get_level_values(1)[:sigma_y.shape[1]].to_numpy()
a = 100

coeffs = np.loadtxt(f'coeffs/legendre_coeffs{num_coeff}.txt')[:,:num_coeff]

# compute mutual information
scaler = StandardScaler()
coeff_scaled = scaler.fit_transform(coeffs)

mi = []
for ii in range(len(x)): # len(x)
    for jj in range(len(y)):
        scaler.fit(sigma_y[ii,jj,:,None])
        sigma_y_scaled = scaler.transform(sigma_y[ii,jj,:,None])
        sigma_y_scaled[np.isclose(sigma_y[ii,jj,:,None]-scaler.mean_, 0, atol=1e-12)]=0
        mi_sample = ee.mi(coeff_scaled,sigma_y_scaled,base=np.e,k=5)
        # shuffle_mean,_ = ee.shuffle_test(ee.mi,coeff_scaled,sigma_y_scaled, ns=100, k=5)
        mi.append(mi_sample)

mi = np.array(mi)
#-----------meshgrid style-------------#
x_grid,y_grid = np.meshgrid(x/a,y/a)

mi_grid = mi.reshape(len(x),len(y)).T
mi_grid_max = np.unravel_index(np.argmax(mi_grid),mi_grid.shape)

levels = np.linspace(np.min(mi_grid), np.max(mi_grid), 15)

fig,ax = plt.subplots(figsize=(3,3))
contour = ax.contourf(x_grid,y_grid,mi_grid, levels = levels, cmap = 'YlOrRd')
ax.set_yscale('log')
ax.invert_yaxis()
ax.set_xlabel(r'$\displaystyle{\sfrac{x}{a}}$')
ax.set_ylabel(r'$\displaystyle{\sfrac{y}{a}}$')
#ax.set_title('MI at each location in halfspace')
plt.colorbar(contour,ax=ax)
plt.plot(x_grid,y_grid,marker = '.', ls = 'none', c='k', markersize=1.0)
plt.tight_layout()

plt.savefig('mi_legendre.pdf')

#-----------------Plot sigma_yy distribution------------#
# plt.figure()
# for ii in range(num_samples):
#     plt.loglog(y,np.abs(sigma_y[0,:,ii]))
# plt.title('Normal stress at x=0')
# plt.xlabel('Depth of Block')
# plt.ylabel('Sigma_yy')

# # match to sigma_yy shape
# mi_sigma_y = mi.reshape(len(x),len(y))

# x_idx = 0

#--------------------2 sensors-------------------------#
mi2 = []

# compute normalized mi at origin sensor 
scaler.fit(sigma_y[mi_grid_max[1],mi_grid_max[0],:,None])
sigma_og1_scaled = scaler.transform(sigma_y[mi_grid_max[1],mi_grid_max[0],:,None])
sigma_og1_scaled[np.isclose(sigma_y[mi_grid_max[1],mi_grid_max[0],:,None]-scaler.mean_, 0, atol=1e-12)]=0

for ii in range(len(x)): # len(x)
    for jj in range(len(y)):
        scaler.fit(sigma_y[ii,jj,:,None])
        sigma_y_scaled = scaler.transform(sigma_y[ii,jj,:,None])
        sigma_y_scaled[np.isclose(sigma_y[ii,jj,:,None]-scaler.mean_, 0, atol=1e-12)]=0

        mi_sample = ee.mi(coeff_scaled,np.hstack([sigma_og1_scaled,sigma_y_scaled]),base=np.e,k=5)
        # shuffle_mean,_ = ee.shuffle_test(ee.mi,coeff_scaled,sigma_y_scaled, ns=100, k=5)
        mi2.append(mi_sample)

mi2_grid = np.array(mi2).reshape(len(x),len(y)).T
mi2_grid_max = np.unravel_index(np.argmax(mi2_grid),mi2_grid.shape)


cmi = mi2_grid-mi_grid[mi_grid_max] # conditional mutual information

levels = np.linspace(np.min(cmi), np.max(cmi), 15)

fig,ax = plt.subplots(figsize=(3,3))
contour = ax.contourf(x_grid,y_grid,cmi, levels = levels, cmap = 'YlOrRd')
ax.set_yscale('log')
ax.invert_yaxis()
ax.set_xlabel(r'$\displaystyle{\sfrac{x}{a}}$')
ax.set_ylabel(r'$\displaystyle{\sfrac{y}{a}}$')
#ax.set_title(r'Conditional Mutual Information $I(X;Y_N|Y_{max1})$')
#plt.plot(x_grid,y_grid,marker = '.', ls = 'none', c='k')
plt.colorbar(contour,ax=ax)
plt.plot(x_grid,y_grid,marker = '.', ls = 'none', c='k', markersize=1.0)
plt.plot(x_grid[mi_grid_max],y_grid[mi_grid_max],marker = 'X', ls = 'none', c='blue', markersize=10)
plt.tight_layout()
plt.savefig('info_gain.pdf')

#--------------------3 sensors-------------------------#
mi3 = []

# compute normalized mi at sensor 2
scaler.fit(sigma_y[mi2_grid_max[1],mi2_grid_max[0],:,None])
sigma_og2_scaled = scaler.transform(sigma_y[mi2_grid_max[1],mi2_grid_max[0],:,None])
sigma_og2_scaled[np.isclose(sigma_y[mi2_grid_max[1],mi2_grid_max[0],:,None]-scaler.mean_, 0, atol=1e-12)]=0

for ii in range(len(x)): # len(x)
    for jj in range(len(y)):
        scaler.fit(sigma_y[ii,jj,:,None])
        sigma_y_scaled = scaler.transform(sigma_y[ii,jj,:,None])
        sigma_y_scaled[np.isclose(sigma_y[ii,jj,:,None]-scaler.mean_, 0, atol=1e-12)]=0

        mi_sample = ee.mi(coeff_scaled,np.hstack([sigma_og1_scaled, sigma_og2_scaled,sigma_y_scaled]),base=np.e,k=5)
        # shuffle_mean,_ = ee.shuffle_test(ee.mi,coeff_scaled,sigma_y_scaled, ns=100, k=5)
        mi3.append(mi_sample)

mi3_grid = np.array(mi3).reshape(len(x),len(y)).T
cmi2 = mi3_grid-cmi[mi2_grid_max]-mi_grid[mi_grid_max] # conditional mutual information
levels = np.linspace(np.min(cmi2), np.max(cmi2), 15)

fig,ax = plt.subplots(figsize=(3,3))
contour = ax.contourf(x_grid,y_grid,cmi2, levels = levels, cmap = 'YlOrRd')
ax.set_yscale('log')
ax.invert_yaxis()
ax.set_xlabel(r'$\displaystyle{\sfrac{x}{a}}$')
ax.set_ylabel(r'$\displaystyle{\sfrac{y}{a}}$')
#ax.set_title(r'Conditional Mutual Information $I(X;Y_N|Y_{max1}, Y_{max2})$')
#plt.plot(x_grid,y_grid,marker = '.', ls = 'none', c='k')
plt.colorbar(contour,ax=ax)
plt.plot(x_grid,y_grid,marker = '.', ls = 'none', c='k', markersize=1.0)
plt.plot(x_grid[mi_grid_max],y_grid[mi_grid_max],marker = 'X', ls = 'none', c='blue', markersize=10)
plt.plot(x_grid[mi2_grid_max],y_grid[mi2_grid_max],marker = 'X', ls = 'none', c='green', markersize=10)

plt.tight_layout()
plt.savefig('info_gain1.pdf')



plt.show()

