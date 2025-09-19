import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *
from pathlib import Path
from npeet import entropy_estimators as ee
from sklearn.preprocessing import StandardScaler

import sys 
sys.path.append('../..')

H = 100
R_all = [H/2.5, H/5, H/10, H/20]
num_coeff=4
num_samples = 5000 # number of load samples

coeffs = np.loadtxt(f'../../half_space/legendre/full/coeffs/legendre_coeffs{num_coeff}.txt')[:,:num_coeff] # legendre coefficients

# compute mutual information
scaler = StandardScaler()
coeff_scaled = scaler.fit_transform(coeffs)

Path('mi').mkdir(parents=True, exist_ok=True)
for R in R_all:
    # Load all stress 
    sigma_yy = []
    for ii in range(100):
        sigma_yy.append(np.loadtxt(f'sigma_yy/R{int(R)}/sigma_yy{ii}.txt'))
    
    sigma_yy = np.vstack(np.array(sigma_yy)).T

    mi = []
    for ii in range(len(sigma_yy)): #len(sigma_yy)
            scaler.fit(sigma_yy[ii,:,None])
            sigma_yy_scaled = scaler.transform(sigma_yy[ii,:,None])

            sigma_yy_scaled[np.isclose(sigma_yy[ii,:,None]-scaler.mean_, 0, atol=1e-12)]=0
            mi_sample = ee.mi(coeff_scaled,sigma_yy_scaled,base=np.e,k=5)
            # shuffle_mean,_ = ee.shuffle_test(ee.mi,coeff_scaled,sigma_y_scaled, ns=100, k=5)
            mi.append(mi_sample)

    mi = np.array(mi)
    np.savetxt(f'mi/R{int(R)}.txt',mi)