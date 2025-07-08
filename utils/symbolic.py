from sympy import *
import numpy as np
from sympy.integrals.rationaltools import ratint
from itertools import product
import mpmath as mp

def complex_to_atan(expr):
    '''
    Converts complex log expression to real atan and log expressions for readibility and numerical stability
    '''
    a = Wild('a', exclude=[I])
    b = Wild('b', exclude=[I])

    c = Wild('c', exclude=[I])
    d=Wild('d', exclude=[I])   

    pattern = I*c*(log(a - I*b) - log(a + I*b)) # pattern to match
    pattern2= (c + I*d)*log(a - I*b) + (c - I*d)*log(a + I*b)

    matches_log = expr.find(pattern) # fins expressions that nmatch this pattern
    matches_log2 = expr.find(pattern2)
    # Sequencially replace complex log with atan
    for match_log in matches_log:
        sub=match_log.match(pattern)
        if sub != None:
            a_val, b_val, c_val = sub[a], sub[b], sub[c]
            real_expr = -2*c_val*atan(a_val/b_val)
            expr = expr.subs(match_log, real_expr)
    
    for match_log in matches_log2:
        sub=match_log.match(pattern2)
        if sub != None:
            a_val, b_val, c_val, d_val = sub[a], sub[b], sub[c], sub[d]
            real_expr = c_val*log(a_val**2+b_val**2) - 2*d_val*atan(a_val/b_val)
            expr = expr.subs(match_log, real_expr)


    return simplify(expr)


def sym_sigma_y(p, polynomial = False):
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

    x,y,s,a_lim = symbols('x y s a_lim')

    # Evaluate indefinite integral
    if polynomial == True:
        # Break polynomial up into small pieces to integrate (way faster this way)
        p_poly = Poly(p,s, extension=True) 
        sigma_y_sym=0
        print(f'Partitioned Polynomial into {len(p_poly.as_expr().args)} terms:')
        for count,mono in enumerate(p_poly.as_expr().args):   
            print(f'Integrating term {count+1} ({mono})...')  
            # sigma_y
            integrand = simplify(mono/(y**2+(x-s)**2)**2)

            integrated = integrate(integrand,s)

            integrated = complex_to_atan(simplify(expand_log(integrated,force=True)))
            print(integrated)
            sigma_y_sym += integrated # symbolic integration
        
        soln = (-2/pi)*y**3*sigma_y_sym
    else:
        #Directly integrate expression (required for constant loads; not recommended for polynomial expressions)
        integrand = p/((y**2+(x-s)**2)**2)
        sigma_y_sym = (-2/pi)*y**3*integrate(integrand,s)
    
        soln = simplify(expand_log(sigma_y_sym, force=True))
        # convert complex logs to atans fpor numerical stability
        soln=complex_to_atan(soln)

    return simplify(soln.subs(s,a_lim) - soln.subs(s,-a_lim)) #return definte integral

def sym_fourier_series(interval, coeffs):
    """
    Get sympy function of nth term fourier series
    coeffs: a list or vector of [a0, a1, b1, a2, b2]
    Args:
        interval: interval [a,b] for load 
        coeffs: Vector of coefficients in the form of list of fourier coefficients [[a_1 , b_1], ..., [a_n,b_n]]

    """
    s = symbols('s', real=True)

    L = Rational((interval[1]-interval[0]),2)

    a_0 = Rational(1,L) #a_0 by definition of fourier with area under curve of 1

    f = (1/2)*a_0 
    for n,(a_n,b_n) in enumerate(coeffs):
        n+=1 # by definition n starts at 1

        f += a_n*cos(pi*s*n/L) + b_n*sin(pi*s*n/L)
    
    
    return f

def sym_even_legendre_series(num_coeff):
    s = symbols('s') # independent variable
    c = symbols(f'c_1:{num_coeff+1}')
    m = symbols('m', postive=True) # Load magnitude  

    a_lim = symbols('a_lim', positive=True) # a is always positive

    p = m/(2*a_lim) # by definition the zeroth term for orthogonal polynomials to enfore \int p(s) = 1

    for n,c_n in enumerate(c):
        p += c_n*legendre(2*(n+1),s)
    
    # transform vairables to interval; legendre polymoial are in range [-1,1]
    p=p.subs(s,s/a_lim)
    
    return simplify(p)
        
def mp_vectorize(func):
    """
    Vectorized evaluator for mpmath functions with arbitrary number of arguments,
    where each argument may be a scalar or a sequence (e.g., list, tuple, NumPy array).

    Evaluates the function over the full Cartesian product of the input arguments.

    Args:
        func: A function that accepts N `mpf`-convertible arguments.

    Returns:
        A callable that takes N inputs (scalars or sequences) and returns an
        N-dimensional NumPy array of `mpmath.mpf` results.
    """
    def wrapper(*args):
        # Convert inputs to object arrays
        arrays = [np.array(arg, dtype=object) for arg in args]

        # Broadcast all arrays to the same shape
        bcast_args = np.broadcast_arrays(*arrays)

        # Prepare empty result array
        result = np.empty(bcast_args[0].shape, dtype=object)

        # Iterate over each index and apply func
        for idx in np.ndindex(result.shape):
            # Directly convert each argument to mpf
            mp_args = [mp.mpf(arr[idx]) for arr in bcast_args]
            result[idx] = func(*mp_args)

        return result

    return wrapper