from sympy import *
import numpy as np


def sym_sigma_y(p):
    '''
    Computes the reaction force on an arbitrary load in an elastic half-space (in the y direction) from integrating the airy stress function derived superposition principle from Flamant's problem .

    Args:
        p: symbolic function function of distributed normal load must be in terms of f(x, y, s)
        x_vec: numerical vector of x-coordinates to evaluate stress
        y_vec: numerical vector of x-coordinates to evaluate stress
        a_num: width of the distributed load

    Returns:
        vector for sigma_y at evaluated (x,y) points
    '''
    p=nsimplify(p) # deal with potential floats

    x,y,s = symbols('x y s', real = True) # make sure symbols are real
    a_lim = symbols('a_lim', real = True, positive=True) # a is always positive

    # sigma_y
    integrand = ((-2/pi)*(y**3*p))/(y**2+(x-s)**2)**2

    sigma_y_sym = integrate(integrand,(s,-a_lim,a_lim)) # symbolic integration

    return sigma_y_sym #sigma_y(x_vec,y_vec,a_num)
