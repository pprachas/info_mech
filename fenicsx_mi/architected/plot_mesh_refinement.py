import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

import sys

from custom_ee import entropy_r
from postprocess import normalize_signal
from fea import import_mesh

plt.style.use('jeff_style.mplstyle')

geoms = ['pore','slit']

# Load and congregate all sigma_yy files
for geom in geoms:
    # import files
    f_path = f'mesh_refinement/sigma/{geom}'

    sigma_y = []
    for ii in range(6):
        sigma_y_all = []
        for run in range(5):
            f_file = f'{f_path}/{geom}9_{ii}/run{run}.npz'
            sigma_y_all.append(np.load(f_file)['sigma_points'])

        sigma_y.append(np.vstack(sigma_y_all))

    sigma_y = np.array(sigma_y)

    # Load Coefficients
    load_case = 'even'
    num_coeff = 6


    coeffs = np.loadtxt(f'../../half_space/legendre/{load_case}/coeffs/legendre_coeffs{num_coeff}.txt')[:,:num_coeff] # legendre coefficients

    # get number of elements
    num_elements = []
    for num in range(6):
        f_mesh = f'mesh_refinement/{geom}/{geom}9_{num}.xdmf'
        domain = import_mesh(f_mesh)
        num_elements.append(domain.topology.index_map(2).size_local)

    num_elements = np.array(num_elements)

    # Compute mi
    scaler = StandardScaler()
    coeff_scaled = scaler.fit_transform(coeffs)

    norm_mi = []
    for sigma_points in sigma_y:
        sigma_points_scaled = normalize_signal(sigma_points,scaler)

        mi_sample,Hx_sample,_,_ = entropy_r(coeff_scaled,sigma_points_scaled,base=np.e,k=5, vol=False)

        norm_mi.append(mi_sample/Hx_sample)

    norm_mi = np.array(norm_mi)

    fig,ax1 = plt.subplots(1,2,figsize=(6.5,3))
    if geom == 'slit':
        fig.suptitle('Slit Geometry')
    elif geom == 'pore':
        fig.suptitle('Pore Geometry')
    
    # Raw values
    ax1[0].plot(num_elements,sigma_y[:,0,0], c = 'k', marker = 'o', markersize = 5)
    ax1[0].axvline(num_elements[-2], ls = ':', color = 'k')
    ax1[0].set_xlabel('Number of Elements')
    ax1[0].set_ylabel(r'$\sigma_{22}$')

    ax2 = ax1[0].twinx() 
    ax2.set_ylabel(r'I(X;Y)/h(X)', color=(0.5,0.,0.)) 
    ax2.tick_params(axis='y', labelcolor=(0.5,0.,0.))
    ax2.plot(num_elements, norm_mi, marker = 's',color=(0.5,0.,0.), markersize = 5)
    ax2.axvline(num_elements[-2], ls = ':', color = 'k')

    # change of values
    percent_sigma_y = np.abs(np.diff(sigma_y[:,0,0])/sigma_y[:,0,0][1:])*100
    ax1[1].plot(num_elements[1:],percent_sigma_y, c = 'k', marker = 'o', markersize = 5)
    ax1[1].axvline(num_elements[-2], ls = ':', color = 'k')
    ax1[1].set_xlabel('Number of Elements')
    ax1[1].set_ylabel(r'Percent Change of $\sigma_{22}$')

    ax2 = ax1[1].twinx() 

    percent_normmi = np.abs(np.diff(norm_mi)/norm_mi[1:])*100
    ax2.plot(num_elements[1:],percent_normmi, color = (0.5,0.,0.), marker = 's', markersize = 5)
    ax2.axvline(num_elements[-2], ls = ':', color = 'k')
    ax2.set_xlabel('Number of Elements')
    ax2.set_ylabel('Percent Change of I(X;Y)/H(X)', color = (0.5,0.,0.))
    ax2.tick_params(axis='y', labelcolor=(0.5,0.,0.))

    print(f'{geom}: {percent_sigma_y}')

    
    plt.tight_layout()
    plt.savefig(f'{geom}_refinement.pdf')
N = np.arange(-2,4)

char_lengths = 20*np.sqrt(2)**N
print(char_lengths)
plt.show()