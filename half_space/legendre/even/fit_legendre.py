import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import sys
import joblib
from custom_ee import entropy_r

#--------------Set up function to give distortion for mlp with best parameters------------------#
def best_mlp(X,y, num_coeff):
  # Set MLP as regressor
  mlp = MLPRegressor(random_state=42, max_iter=5000, early_stopping=True) 


  param_grid = {
      'hidden_layer_sizes': [(100,), (100,100), (100, 100, 100) , (100,100,100,100)], # Number of neurons in hidden layers
      'activation': ['relu'], # Activation function for the hidden layer
      'solver': ['adam'], # Solver for weight optimization
      'tol': [1e-6, 1e-8,1e-10],
      'alpha': [1e-4,1e-3,1e-2,1e-1], # L2 regularization term
      'learning_rate': ['adaptive'] # Learning rate schedule
  }

  grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

  #------------------------Test-train Split--------------------#
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

  scaler_X = StandardScaler().fit(X_train)
  scaler_Y = StandardScaler().fit(y_train)

  X_train = scaler_X.transform(X_train)
  X_test = scaler_X.transform(X_test)
  y_train = scaler_Y.transform(y_train)
  y_test = scaler_Y.transform(y_test)

  if num_coeff == 1:
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

  grid_search.fit(X_train, y_train)


  print(f"Best parameters: {grid_search.best_params_}")
  best_mlp_model = grid_search.best_estimator_

  y_pred = best_mlp_model.predict(X_test)

  mse = mean_squared_error(y_test,y_pred)

  if num_coeff == 1:
    Ixx = entropy_r(y_test[:,None], y_pred[:,None], base=np.e, k=5)[0]
  else:
    Ixx = entropy_r(y_test, y_pred, base=np.e, k=5)[0]

  # Get train scores
  grid_search.cv_results_['mean_train_score']


  return mse, Ixx, grid_search.cv_results_['mean_train_score'], y_pred


#----------------set up folder to save results
Path('./RD/greedy').mkdir(parents=True, exist_ok=True)
Path('./RD/random').mkdir(parents=True, exist_ok=True)

num_coeff = int(sys.argv[1])
num_samples = 5000 # number of load samples

#-------------------------Load Data--------------------------------------------------------#
df = pd.read_csv(f'sigma_y/legendre_coeffs{num_coeff}.csv', index_col=[0,1], header=[0])

sigma_y = df.to_numpy()
# change index to (x,y,load)
sigma_y=sigma_y.reshape(num_samples,-1,sigma_y.shape[1]).transpose(2,1,0) 

# get x and y data values
x_data=df.columns.to_numpy().astype(float)
y_data=df.index.get_level_values(1)[:sigma_y.shape[1]].to_numpy()
a = 100 # width of load

# get legendre coefficients
coeffs = np.loadtxt(f'coeffs/legendre_coeffs{num_coeff}.txt')[:,:num_coeff]
# load indices for sensor location
idx = np.loadtxt(f'sensor_loc/coeffs{num_coeff}.txt').astype(int)


sensors = sigma_y[idx.T[1], idx.T[0],:].T
y = coeffs

mse_random = []
mse_greedy = []
y_pred_greedy = []

MI_random = []
MI_greedy = []
train_scores = []

for ii in range(len(idx)): 
  #--------greedy sensor results-----#
  X = sensors[:,0:ii+1]

  mse,Ixx, train_score, y_pred = best_mlp(X,y, num_coeff)
  mse_greedy.append(mse)
  MI_greedy.append(Ixx)
  train_scores.append(train_score)
  y_pred_greedy.append(y_pred)
  print(y_pred.shape)

  #------random sensor results------#
  mse_random.append([])
  MI_random.append([])
  for jj in range(10): # 10
    seed = jj
    rng = np.random.default_rng(seed)

    idx1 = rng.choice(a=len(x_data), size = ii+1)
    idx2 = rng.choice(a=len(y_data), size = ii+1)

    X = sigma_y[idx1, idx2, :].T
    mse,Ixx,_,_ = best_mlp(X,y, num_coeff)
    mse_random[ii].append(mse)
    MI_random[ii].append(Ixx)

mse_greedy = np.array(mse_greedy)
mse_random = np.array(mse_random)

MI_greedy = np.array(MI_greedy)
MI_random = np.array(MI_random)

train_scores = np.array(train_scores)

y_pred_greedy = np.array(y_pred_greedy)

np.savetxt(f'RD/greedy/D{num_coeff}.txt', mse_greedy)
np.savetxt(f'RD/random/D{num_coeff}.txt', mse_random)

np.savetxt(f'RD/greedy/R{num_coeff}.txt', MI_greedy)
np.savetxt(f'RD/random/R{num_coeff}.txt', MI_random)

np.savez(f'RD/greedy/y_pred{num_coeff}.npz', y_pred=y_pred_greedy)


np.savetxt(f'RD/greedy/train_scores{num_coeff}.txt',train_scores)
