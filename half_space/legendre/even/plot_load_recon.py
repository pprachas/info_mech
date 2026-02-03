import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from symbolic import sym_even_legendre_series
from sympy import *
from sklearn.preprocessing import StandardScaler

plt.style.use('../../jeff_style.mplstyle')

num_coeff = 4
# Load predicted coefficients
y_pred = np.load(f'RD/greedy/y_pred{num_coeff}.npz')['y_pred']

# Load ground truth (make sure test set)
coeffs = np.loadtxt(f'coeffs/legendre_coeffs{num_coeff}.txt')[:,:num_coeff]

_, _, y_train, y_test = train_test_split(coeffs,coeffs, test_size=0.2, random_state=42) # test values are dummy
scaler_Y = StandardScaler().fit(y_train)
print(y_pred.shape, y_train.shape)
y_scaled = []
for ii in range(5):
    y_scaled.append(scaler_Y.inverse_transform(y_pred[ii])) # unnormalize to recontruct load

y_pred = np.array(y_scaled)
# Create sympy equation
x,s = symbols('x s')
y,a_lim,m = symbols('y a_lim m', real = True, positive=True)
num_coeff_max=4 # Use 6 term (even) legendre polynomials
c = symbols(f'c_1:{num_coeff_max+1}', real = True)

# Create Legendre Even Load
p=sym_even_legendre_series(num_coeff_max) # Applied load
load = lambdify([c,s,a_lim,m], p, modules=['numpy']) # using numpy is fine here

a = 100 # width of load
x_load = np.linspace(-a,a,100)
m_num=1


print(y_test.shape)
print(y_pred.shape)

print(y_pred[:,0,:], y_test[0,:])

for ii in range(5):
    plt.figure(figsize = (1,1))
    plt.plot(x_load,load(y_test[0,:], x_load, a,m_num), c = (0.5,0,0), lw=1.5) # ground turth
    plt.plot(x_load,load(y_pred[ii,0,:], x_load, a,m_num), c = '0.4') # reconstruction
    plt.savefig(f'load_recon{ii+1}.pdf')
plt.show()