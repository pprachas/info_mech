import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *
from pathlib import Path
from npeet import entropy_estimators as ee
from sklearn.preprocessing import StandardScaler

import sys 
sys.path.append('../..')
from utils.custom_ee import entropy_r


L = 100
H_all = [4*L,2*L,L,L/2,L/4]
num_coeff = 4
num_samples = 5000 # number of load samples

coeffs = np.loadtxt(f'../../half_space/legendre/full/coeffs/legendre_coeffs{num_coeff}.txt')[:,:num_coeff] # legendre coefficients

# compute mutual information
scaler = StandardScaler()
coeff_scaled = scaler.fit_transform(coeffs)

mi = []
Hx = []
for H in H_all:
    # Load all stress 
    sigma_yy = np.loadtxt(f'pointwise_sigma/H{int(H)}.txt')
    
    sigma_yy = np.vstack(np.array(sigma_yy))

    scaler.fit(sigma_yy)
    sigma_yy_scaled = scaler.transform(sigma_yy)

    sigma_yy_scaled[np.isclose(sigma_yy-scaler.mean_, 0, atol=1e-12)]=0
    mi_sample,Hx_sample,_,_ = entropy_r(coeff_scaled,sigma_yy_scaled,base=np.e,k=5, vol=False)
            # shuffle_mean,_ = ee.shuffle_test(ee.mi,coeff_scaled,sigma_y_scaled, ns=100, k=5)
    mi.append(mi_sample)
    Hx.append(Hx_sample)

mi = np.array(mi)
Hx = np.array(Hx)

plt.figure()
plt.title('Depth vs MI')
plt.plot(np.array(H_all)/L,mi/Hx, marker = 'o')
plt.xlabel('Depth/Width')
plt.ylabel('I(X;Y)/H(X)')
plt.savefig('pointwise_mi.png')
plt.show()