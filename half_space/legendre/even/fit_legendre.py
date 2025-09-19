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
from sklearn.pipeline import Pipeline
import seaborn as sns
from npeet import entropy_estimators as ee
from custom_ee import entropy_r
from postprocess import normalize_signal

num_coeff = 4
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

sensor = sigma_y[idx.T[1], idx.T[0],:].T
sensor_scaled=scaler.fit_transform(sensor)
regression = LinearRegression()
pipe = Pipeline([
    ('scaler', scaler),         # First step: scale the features
    ('regression', regression)    # Second step: apply the regression model
])

#--------------------loop for number of sensors--------------#
score = []
score_dummy = []
IXY = []
IXXhat = []

IXXhat_dummy = []

#---------------Pipeline----------------#
pipe = Pipeline([
    ('scaler', StandardScaler()),     # This scales X again — optional if already done
    ('regression', LinearRegression())
])

for ii in range(len(idx)): # range(len(idx))
#------------------------Test-train Split--------------------#
  X_train, X_test, y_train, y_test = train_test_split(sensor[:, 0:ii+1], coeffs, test_size=0.2, random_state=42)

  scaler_Y = StandardScaler().fit(y_train)

  y_train_standardized = scaler_Y.transform(y_train)
  pipe.fit(X_train, y_train_standardized)
  y_pred = pipe.predict(X_test)
  #------------------------Compute Metrics------------------------#
  y_test_standardized = scaler_Y.transform(y_test)
  score.append(mean_squared_error(y_test_standardized, y_pred))

  # Mutual Information
  #IXY.append(ee.mi(normalize_signal(sensor_scaled[:, 0:ii+1], StandardScaler()), normalize_signal(coeff_scaled, StandardScaler()), base=np.e, k=5))   # I(X;X̂)
  IXXhat.append(entropy_r(y_test_standardized, y_pred, base=np.e, k=5)[0])   # I(X;X̂)

    # do regression on random sensor locations
  score_dummy.append([])
  IXXhat_dummy.append([])
  seed = ii
  rng = np.random.default_rng(seed)
  for jj in range(20):
      idx1 = rng.integers(low=0, high=len(x), size=ii+1)
      idx2 = rng.integers(low=0, high=len(y), size=ii+1)


      sensor_dummy = sigma_y[idx1, idx2, :].T
      X_train, X_test, y_train, y_test = train_test_split(sensor_dummy, coeffs, test_size=0.2, random_state=42)
      scaler_Y = StandardScaler().fit(y_train)

      y_train_standardized = scaler_Y.transform(y_train)
      pipe.fit(X_train, y_train_standardized)
      y_pred = pipe.predict(X_test)

      #------------------------Compute Metrics------------------------#
      y_test_standardized = scaler_Y.transform(y_test)

      score_dummy[ii].append(mean_squared_error(y_test_standardized, y_pred))
      IXXhat_dummy[ii].append(entropy_r(y_test_standardized, y_pred, base=np.e, k=5)[0])   # I(X;X̂)



score = np.array(score)
score_dummy = np.array(score_dummy)

IXY = np.array(IXY)
IXXhat = np.array(IXXhat)

IXXhat_dummy = np.array(IXXhat_dummy)

#setup dataframe for violin plot
df = pd.DataFrame(score_dummy.T)
df.columns += 1 # make index star from 0

plt.figure()
#plt.plot(np.arange(0,len(score))+1,score_dummy, c='k', alpha = 0.5)
ax = sns.violinplot(df,cut=0, inner = 'quartile', native_scale=True, color = '0.8', density_norm='width')

# only show median line
for ii,line in enumerate(ax.lines):
  line.set_linestyle('-')
  if (ii-1)%3 != 0: 
    line.set_linestyle('')

# Violin plot
sns.stripplot(df, ax = ax, jitter = True, native_scale=True, color = (0.9,0.5,0.5))
plt.plot(np.arange(0,len(score))+1,score, marker = 's',markersize= 8, ls = 'none', mec = (0.5,0,0), mew = 1.5, zorder = 10, fillstyle='none')
plt.title(f'{num_coeff} Legendre Coefficients')
plt.xlabel('Number of Sensors')
plt.ylabel('MSE')
plt.savefig('test.pdf')

# Rate-distortion
plt.figure()

score = np.maximum(score,1e-12)
score_dummy = np.maximum(score_dummy,1e-12)


plt.plot(score, IXXhat, marker = 's',markersize= 8, ls = 'none', mec = (0.5,0,0), mew = 1.5, zorder = 10, fillstyle='none')
plt.plot(score_dummy, IXXhat_dummy, marker = '.', ls = 'none', color = (0.9,0.5,0.5))


a = 1.0  # example: x ~ U(-1, 1)

D = np.logspace(-3, np.log10(score_dummy.max()), 200)

d_D = 1
# shannon lower bound
R = 0.5 * num_coeff * np.log(2/(np.pi*np.e*D))

R[R < 0] = 0


plt.plot(D,R, ls = ':', c = 'k', label = 'Shannon Lower Bound')

plt.xlabel('Distortion')
plt.ylabel('Rate')
plt.legend()

print(score.min())
plt.show()