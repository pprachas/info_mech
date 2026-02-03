import numpy as np
import pandas as pd
from sympy import *

from symbolic import sym_legendre_series, sym_even_legendre_series, sym_sigma_y, mp_vectorize


num_coeff=1
num_samples = 3

# integrate

x,s = symbols('x s')
y,a_lim,m = symbols('y a_lim m', real = True, positive=True)
num_coeff_max=1 # Use 6 term (even) legendre polynomials
c = symbols(f'c_1:{num_coeff_max+1}')


p = sym_legendre_series(num_coeff_max)

soln = sym_sigma_y(p, polynomial=True)

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

w = 2*a + a/10
x_nums = np.arange(-w,w,a/10)
y_nums = np.logspace(-4,9,27)

df= []
for coeff in coeffs:
    sigma_y_res = []
    for y_num in y_nums:
        sigma_y_res.append(sigma_y_vec(*coeff, x_nums, y_num,m, a).astype(float))

    
    df.append(pd.DataFrame(sigma_y_res, index=y_nums, columns=x_nums))

# Save dataframe
df_all=pd.concat(df,  keys=range(num_samples), axis=0)
sigma_y = df_all.to_numpy().reshape(3, -1, df_all.shape[1]).transpose(2, 1, 0)


df = pd.read_csv(f'sigma_y_old/legendre_coeffs{num_coeff}.csv', index_col=[0, 1], header=[0])
sigma_y_old = df.to_numpy().reshape(5000, -1, df.shape[1]).transpose(2, 1, 0)[:,:,0:3]

print(np.sum(sigma_y-sigma_y_old))
