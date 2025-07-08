import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns

import sys 
sys.path.append('../..')

num_coeff=4
num_samples = 5000 # number of load samples

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


#--------------------loop for number of sensors--------------#
score = []
score_dummy = []
for ii in range(len(idx)):
#------------------------Test-train Split--------------------#
    X_train, X_test, y_train, y_test = train_test_split(sensor_scaled[:,0:ii+1], coeff_scaled, test_size=0.2, random_state=42)


    #------------------------Setup regression-----------------------#
    regression = LinearRegression()
    regression.fit(X_train,y_train)
    y_pred = regression.predict(X_test)
    score.append(mean_squared_error(y_test, y_pred))

    # do regression on random sensor locations
    score_dummy.append([])
    seed = 0
    rng = np.random.default_rng(seed)
    for jj in range(20):
        idx1 = rng.integers(low=0, high=len(x), size=ii+1)
        idx2 = rng.integers(low=0, high=len(y), size=ii+1)

        sensor = sigma_y[idx1, idx2,:].T
        X_train, X_test, y_train, y_test = train_test_split(sensor, coeff_scaled, test_size=0.2, random_state=42)

        regression.fit(X_train,y_train)
        y_pred = regression.predict(X_test)
        score_dummy[ii].append(mean_squared_error(y_test, y_pred))

score = np.array(score)
score_dummy = np.array(score_dummy)

#setup dataframe for violin plot
df = pd.DataFrame(score_dummy.T)
df.columns += 1 # make index star from 0
print(df)

plt.figure()
#plt.plot(np.arange(0,len(score))+1,score_dummy, c='k', alpha = 0.5)
ax = sns.violinplot(df,cut=0, inner = 'quartile', native_scale=True, color = '0.8', density_norm='width')

# only show median line
for ii,line in enumerate(ax.lines):
  line.set_linestyle('-')
  if (ii-1)%3 != 0: 
    line.set_linestyle('')

sns.stripplot(df, ax = ax, jitter = True, native_scale=True, color = (0.9,0.5,0.5))
plt.plot(np.arange(0,len(score))+1,score, lw=3, marker = 's',markersize= 8, ls = 'none', mec = (0.5,0,0), mew = 1.5, zorder = 10, fillstyle='none')
plt.title(f'{num_coeff} Legendre Coefficients')
plt.xlabel('Number of Sensors')
plt.ylabel('MSE')
plt.savefig('test.pdf')
plt.show()


    
    


