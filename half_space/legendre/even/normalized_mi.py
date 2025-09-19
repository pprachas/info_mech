import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator


import sys 
from custom_ee import entropy_r

num_samples = 5000 # number of load samples

c = ['0.1','0.3','0.5','0.7']
# Plot 1
fig1,ax1 = plt.subplots()
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.axhline(1.0, ls = ':', c = 'k', label = 'Max Theoretical I(X;Y)/H(X)')
ax1.set_ylabel('I(X;Y)/H(X)')
ax1.set_xlabel('Number of Sensors')

# Plot 2
fig2,ax2 = plt.subplots()

ax2.set_ylabel('I(X;Y)')
ax2.set_xlabel('H(Y)')
ax2.axline((0, 0), slope=1, ls = ':', c = 'k')

for num_coeff in range(1,5):
    #-------------------------Load Data--------------------------------------------------------#
    df = pd.read_csv(f'sigma_y/legendre_coeffs{num_coeff}.csv', index_col=[0,1], header=[0])

    sigma_y = df.to_numpy()
    # change index to (x,y,load)
    sigma_y=sigma_y.reshape(num_samples,-1,sigma_y.shape[1]).transpose(2,1,0) 

    # get x and y data values
    x=df.columns.to_numpy().astype(float)
    y=df.index.get_level_values(1)[:sigma_y.shape[1]].to_numpy()
    a = 100

    # get legendre coefficients
    coeffs = np.loadtxt(f'coeffs/legendre_coeffs{num_coeff}.txt')[:,:num_coeff]
    # load indices for sensor location
    idx = np.loadtxt(f'sensor_loc/coeffs{num_coeff}.txt').astype(int)
    #------------------------scale data-----------------------#
    scaler = StandardScaler()
    coeff_scaled = scaler.fit_transform(coeffs)

    sensors = sigma_y[idx.T[1], idx.T[0],:].T
    sensor_scaled=scaler.fit_transform(sensors)

    mi_all = []
    hx_all = []
    hy_all = []
    hxy_all = []
    for ii in range(len(idx)): #len(idx)
        mi, hx, hy, hxy = entropy_r(coeff_scaled, sensor_scaled[:,0:ii+1],k=5 ,base = np.e)
        mi_all.append(mi)
        hx_all.append(hx)
        hy_all.append(hy)
        hxy_all.append(hxy)

    mi_all = np.array(mi_all)
    hx_all = np.array(hx_all)
    hy_all = np.array(hy_all)
    hxy_all = np.array(hxy_all)

    nmi = mi_all/hx_all

    ax1.plot(np.arange(1,6),nmi, marker = 'o', label = f'{num_coeff} Legendre Coefficients')

    

    ax2.scatter(hy_all[:num_coeff],mi_all[:num_coeff], marker = 'o',label = f'{num_coeff} Legendre Coefficients', c = c[num_coeff-1])
    ax2.scatter(hy_all[num_coeff:],mi_all[num_coeff:], marker = '*',label = f'{num_coeff} Legendre Coefficients extra', c = c[num_coeff-1])

ax1.legend()
ax2.legend()
plt.show()