import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *
import mpmath as mp
import pandas as pd
from pathlib import Path

import sys 
sys.path.append('../..')

from utils.symbolic import sym_even_legendre_series, sym_sigma_y, mp_vectorize

num_samples = 5000 # number of load samples
#------------------------Sympy Solution---------------------------#
# Set up symbols -- we let s be dummy varaible integrated in superposition equation
x,s = symbols('x s')
y,a_lim,m = symbols('y a_lim m', real = True, positive=True)
num_coeff_max=6
c = symbols(f'c_1:{num_coeff_max+1}', real = True)

# load file solution
f_name = '../integration_soln.txt'
with open(f_name, 'r') as file:
    soln_str = file.read()

soln = sympify(soln_str)
#-----------Lamdify and vectorize from numerical solution------------#
sigma_y_vec = lambdify([*c,x,y,m,a_lim], soln, modules=['numpy'])

#------------------Numerically Evaluate function---------------------#
a_max=100
m=1000
rng = np.random.default_rng(0)

# unifrom load -- all coefficients are 0:
coeff = np.array([0]*6)
# sample coeffcients
a_all = rng.uniform(high=a_max,low=0, size=(num_samples))


# Save coefficients
np.savetxt(f'a.txt',a_all)

x_nums = np.arange(0,2*a_max + a_max/10,a_max/10)
y_nums = np.logspace(-4,9,27)


df= []
for a in a_all:
    sigma_y_res = []
    for y_num in y_nums:
        sigma_y_res.append(sigma_y_vec(*coeff, x_nums, y_num,m, a).astype(float))

    
    df.append(pd.DataFrame(sigma_y_res, index=y_nums, columns=x_nums))

# Concatenate dataframes
df_all=pd.concat(df,  keys=range(num_samples), axis=0)
# Save dataframe
df_all.to_csv(f'sigma_y.csv')

index_col = [ii for ii in range(num_samples)]

print(index_col)
df1 = pd.read_csv(f'sigma_y.csv', index_col=[0,1], header=[0])

print(df1)