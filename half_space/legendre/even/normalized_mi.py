import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as ticker

import sys 
from custom_ee import entropy_r

plt.style.use('../../jeff_style.mplstyle')


num_samples = 5000 # number of load samples

color = ['0.2', (0.5,0.5,0.5), (0.9,0.5,0.5), (0.4,0.0,0.0)]
# Plot 1
fig1,ax1 = plt.subplots(figsize=(6.5/2,6.5/2))
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.axhline(1.0, ls = ':', c = 'k', label = 'Max Theoretical I(X;Y)/H(X)')
ax1.set_ylabel(r'$I(X;Y)/H(X)$', fontsize = 12)
ax1.set_xlabel('Number of Sensors', fontsize=12)
ax1.set_title(r'$\max \; I(X;Y) = H(X)$', fontsize = 12)

# Plot 2
fig2,ax2 = plt.subplots(figsize=(6.5/2,6.5/2))

ax2.set_ylabel(r'$I(X;Y)$', fontsize=12)
ax2.set_xlabel(r'$H(Y)$', fontsize=12)
ax2.set_title(r'$I(X;Y) = H(Y)$', fontsize=12)
ax2.axline((0, 0), slope=1, ls = ':', c = 'k')
ax2.set_xlim([1,7.5])
ax2.set_ylim([1,7.5])
ax2.set_aspect('equal')
tick_spacing = 1
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

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

    ax1.plot(np.arange(1,6)[:num_coeff],nmi[:num_coeff], marker = 'o', mew = 1.0, 
            mec = color[num_coeff-1], ls = 'none', fillstyle = 'none')
    ax1.plot(np.arange(1,6)[num_coeff:],nmi[num_coeff:], marker = '*', mew = 1.0, 
            mec = color[num_coeff-1], ls = 'none', fillstyle = 'none')
    ax1.plot(np.arange(1,6),nmi,c = color[num_coeff-1])




    ax2.plot(hy_all[:num_coeff],mi_all[:num_coeff], marker = 'o',
    label = f'{num_coeff} Legendre Coefficients', mew=1.0,mec = color[num_coeff-1], fillstyle='none',ls = 'none')
    ax2.plot(hy_all[num_coeff:],mi_all[num_coeff:], marker = '*',
    label = f'{num_coeff} Legendre Coefficients extra',mew=1.0, mec = color[num_coeff-1], fillstyle='none',ls = 'none')

fig1.tight_layout()
fig1.savefig('HX_max.pdf')
fig2.tight_layout()
fig2.savefig('HY_IXY.pdf')


# Custom handles
handles = [
    Line2D([0],[0], ls = 'none',marker='o', mfc='none', mew = 1.5, mec = 'k', markersize=8, label = r'Number of Sensors $\leq d_x$'),
    Line2D([0],[0], ls = 'none', marker='*', mfc='none', mew = 1.5, mec = 'k', label = r'Number of Sensors $> d_x$')]

plt.figure()
plt.legend(handles=handles,
    title = 'Number of Sensors',
    fontsize = 12,
    title_fontsize = 12,
    ncol=2)
plt.savefig('marker_legemd.pdf')

custom_handles = [
    Patch(facecolor = color[0], label = '1'), Patch(facecolor = color[1], label = '2'), Patch(facecolor = color[2], label = '3'),
    Patch(facecolor = color[3], label = '4')
]

labels = [h.get_label() for h in custom_handles]


fig_leg = plt.figure()
fig_leg.legend(custom_handles,
               labels,
               title = r'$d_x$',
               fontsize=12, 
               title_fontsize = 12,
               ncol = 4)


fig_leg.tight_layout()
plt.savefig('normi_colors.pdf')

plt.show()