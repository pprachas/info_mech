import numpy as np
import scipy
from scipy.fft import ifft
import matplotlib.pyplot as plt
from sympy import *
from sympy import lambdify

import sys 
sys.path.append('..')

from utils.symbolic import sym_fourier_series, sym_sigma_y

# sample random coefficients
x, s = symbols('x s')

interval = [0,10.0]

coeff = np.random.uniform(size=(2,2)) # 5 term fourier series since a_0 is given

# get symbolic fourier series
f = sym_fourier_series(interval,coeff)

print(f'Original Function: {f}')

# Validate with sympy fourier series -- should return the same function back
f_fourier=fourier_series(f,(x,interval[0],interval[1]))

print(f'Sympy fourier series:{f_fourier.truncate()}')

# Compute area under the curve -- should be 1 by construction
print(f'Area under the curve: {simplify(integrate(f,(x,interval[0], interval[1])))}')

# plot function
plt.figure()
plot(f,(x,interval[0], interval[1]))
plt.savefig('fourier_test.png')