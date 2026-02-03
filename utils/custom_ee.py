import numpy as np
from numpy import log
from npeet.entropy_estimators import *
from scipy.special import digamma

# Written in the same format as npeet -- npeet functions are used here


def entropy_r(x,y,k=3,base=2, vol=False):
    '''
    Relative differential entropy. The base distribution is a uniform hypercube.
    This code is modified from NPEET
    '''
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    
    x, y = np.asarray(x), np.asarray(y)
    _,dx = x.shape
    _,dy = y.shape
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    # x = add_noise(x)
    # y = add_noise(y)
    points = [x, y]

    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points)
    tree_x = build_tree(x)
    tree_y = build_tree(y)
    dvec = query_neighbors(tree, points, k)
    dvec_x = query_neighbors(tree_x, x, k)
    dvec_y = query_neighbors(tree_y, y, k)

    r_approx = np.mean(dvec**(dx+dy))

    dvec_scaled = dvec/r_approx**(1/(dx+dy))
    Elog_dvec_scaled = log(dvec_scaled).mean() 

    digamma_nx, digamma_ny, digamma_k, digamma_N = (
        avgdigamma(x, dvec),
        avgdigamma(y, dvec),
        digamma(k),
        digamma(len(x)),
    )
    
    if vol == True:
        # Volume approximation
        cx = 2**dx # unit cube length is is 2r
        cy = 2**dy # unit cube length is is 2r
        Volx = cx*(len(x)/k*r_approx)**(dx/(dx+dy))
        Voly = cy*(len(y)/k*r_approx)**(dy/(dx+dy))
        Volxy = cx*cy*len(x)/k*r_approx

        return (
            (-digamma_nx - digamma_ny + digamma_k + digamma_N) / log(base), #mi
            (digamma_N - digamma_nx + dx*Elog_dvec_scaled)/log(base), # H(X)
            (digamma_N - digamma_ny + dy*Elog_dvec_scaled)/log(base), # H(Y)
            (digamma_N - digamma_k + (dx+dy)*Elog_dvec_scaled)/log(base),  #(H(X,Y))
            Volx, # approximate of support of X (vol(X))
            Voly, # approximate of support of Y (vol(Y))
            Volxy # approximate of support of XY (vol(XY))
        )
        
    else:
         return (
            (-digamma_nx - digamma_ny + digamma_k + digamma_N) / log(base), #mi
            (digamma_N - digamma_nx + dx*Elog_dvec_scaled)/log(base), # H(X)
            (digamma_N - digamma_ny + dy*Elog_dvec_scaled)/log(base), # H(Y)
            (digamma_N - digamma_k + (dx+dy)*Elog_dvec_scaled)/log(base)  #(H(X,Y))
         )


