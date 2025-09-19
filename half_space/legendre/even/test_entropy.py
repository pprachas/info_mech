import numpy as np
import scipy
import pandas as pd
from npeet import entropy_estimators as ee
from sklearn.preprocessing import StandardScaler
from postprocess import normalize_signal
from custom_ee import entropy_r, entropy


num_coeff=4
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

scaler = StandardScaler()
coeff_scaled = normalize_signal(coeffs,scaler)

entropy_Y = []
entropy_X = []
mi = []
cXY = []
cYX = []
HXY = []
r = []


for ii in range(1): #sigma_y.shape[0]
        for jj in range(1): #sigma_y.shape[1]
            sigma_y_scaled = normalize_signal(sigma_y[ii,jj,:,None],scaler)
            entropy_sample = entropy(sigma_y_scaled,base=np.e,k=5)
            # shuffle_mean,_ = ee.shuffle_test(ee.mi,coeff_scaled,sigma_y_scaled, ns=100, k=5)
            mi_sample = ee.mi(coeff_scaled,sigma_y_scaled,base=np.e,k=5)
            entropy_Y.append(entropy_sample)
            entropy_X.append(ee.entropy(coeff_scaled,base=np.e,k=5))
            cXY.append(ee.centropy(coeff_scaled,sigma_y_scaled,k=5,base=np.e))
            cYX.append(ee.centropy(sigma_y_scaled,coeff_scaled,k=5,base=np.e))
            xy = np.c_[coeff_scaled,sigma_y_scaled]
            HXY.append(entropy(xy, k=5, base=np.e))
            mi.append(mi_sample)
            mi_r,HX_r,HY_r,HXY_r = entropy_r(coeff_scaled,sigma_y_scaled,base=np.e,k=5)
            


print(f'Number of coefficients {num_coeff}')      
print(f'H(Y): {entropy_Y}')
print(f'H(X): {entropy_X}')
print(f'H(X|Y): {cXY}')
print(f'H(Y|X): {cYX}')
print(f'H(X,Y): {HXY}')
print(f'I(X;Y):{mi}')
print(f'H_r(X) {HX_r}')
print(f'H_r(Y) {HY_r}')
print(f'H_r(XY) {HXY_r}')
print(f'H(X,Y) - H(X) {HXY_r-HX_r}')