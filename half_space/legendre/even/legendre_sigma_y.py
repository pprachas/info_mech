import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *
import mpmath as mp
import pandas as pd
from pathlib import Path

from symbolic import sym_even_legendre_series, sym_sigma_y, mp_vectorize

Path('./coeffs').mkdir(parents=True, exist_ok=True)

mp.mp.dps = 70
num_samples = 5000 # number of load samples
#------------------------Sympy Solution---------------------------#
# Set up symbols -- we let s be dummy varaible integrated in superposition equation
x,s = symbols('x s')
y,a_lim,m = symbols('y a_lim m', real = True, positive=True)
num_coeff_max=6
num_coeff=int(sys.argv[1])
c = symbols(f'c_1:{num_coeff_max+1}', real = True)

# load file solution
f_name = 'legendre_soln.txt'
with open(f_name, 'r') as file:
    soln_str = file.read()

soln = sympify(soln_str)
#-----------Lamdify and vectorize from numerical solution------------#
sigma_y = lambdify([*c,x,y,m,a_lim], soln, modules=['mpmath'])

sigma_y_vec = mp_vectorize(sigma_y)
#------------------Numerically Evaluate function---------------------#
a=100
m=1000

rng = np.random.default_rng(0)

# sample coeffcients
coeffs_n = rng.uniform(high=10,low=-10, size=(num_samples,num_coeff))
coeffs=np.zeros((num_samples,num_coeff_max)) # max number of coefficients
coeffs[:,:num_coeff] = coeffs_n

# Save coefficients

np.savetxt(f'coeffs/legendre_coeffs{num_coeff}.txt',coeffs)

x_nums = np.arange(0,2*a + a/10,a/10)
y_nums = np.logspace(-4,9,27)

print(y_nums)

df= []
for coeff in coeffs:
    sigma_y_res = []
    for y_num in y_nums:
        sigma_y_res.append(sigma_y_vec(*coeff, x_nums, y_num,m, a).astype(float))

    
    df.append(pd.DataFrame(sigma_y_res, index=y_nums, columns=x_nums))

# Save dataframe
f_path = 'sigma_y'
path = Path(f_path)
path.mkdir(parents=True, exist_ok=True)
df_all=pd.concat(df,  keys=range(num_samples), axis=0)

df_all.to_csv(f'sigma_y/legendre_coeffs{num_coeff}.csv')

index_col = [ii for ii in range(num_samples)]

print(index_col)
df1 = pd.read_csv(f'sigma_y/legendre_coeffs{num_coeff}.csv', index_col=[0,1], header=[0])

print(df1)