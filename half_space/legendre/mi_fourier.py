import numpy as np
import scipy
from scipy.fft import ifft
import matplotlib.pyplot as plt
from sympy import *
from sympy import lambdify

import sys 
sys.path.append('../..')

from utils.symbolic import sym_fourier_series, sym_sigma_y

# sample random coefficients
s = symbols('s', real=True)

interval = [0,100.0]

# coeff = np.random.uniform(size=(2,2)) # 5 term fourier series since a_0 is given

coeff=[[1,0]]
f = sym_fourier_series(interval,coeff)

sigma_y = sym_sigma_y(cos(s))

print(sigma_y.doit())