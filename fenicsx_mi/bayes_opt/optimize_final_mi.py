import numpy as np
from ufl import sym, grad, Identity, tr, inner, Measure, TestFunction, TrialFunction
import ufl
import petsc4py.PETSc as PETSc

from mpi4py import MPI
from dolfinx import fem, io, plot
import dolfinx.fem.petsc
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import create_rectangle, CellType
import os, sys

import pyvista

from symbolic import sym_legendre_series
from fea import eval_points, run_linear_fea_traction_batch, import_mesh, compute_stress, compute_reaction_force

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from custom_ee import entropy_r
from postprocess import normalize_signal

# Read mesh
L = 100
H = 100
num_coeff = 6

sigma_points = []

flag = sys.argv[1]

f_mesh = f'optimize/opt{flag}.xdmf'
load_case = 'full'

coeffs = np.loadtxt(f'../../half_space/legendre/{load_case}/coeffs/legendre_coeffs{num_coeff}.txt')[:,:num_coeff] # legendre coefficients

# points to evaluate traction
points_x = np.linspace(-L/2,L/2,num_coeff)
points = np.zeros((num_coeff,3))
points[:,0] = points_x

# Import mesh
domain=import_mesh(f_mesh)

# solve displacement
a,l,V,u,bcs,u0 = run_linear_fea_traction_batch(domain,L,H, coeffs, load_case)

# Compute function space for stress:
W = fem.functionspace(domain, ('DG', 1, (2,2)))
E = fem.Constant(domain, 1.0)
nu = fem.Constant(domain, 0.0)


for ii in range(len(u)):
    #------------Interpolate stress--------------#
    sigma_u = compute_stress(W,u[ii], E, nu)
    #-----------Extract sigma_yy component----#
    sigma_xx,sigma_xy,sigma_yx, sigma_yy = sigma_u.split() 

    sigma_points.append(eval_points(domain,sigma_yy,points))

sigma_points= np.array(sigma_points).squeeze()

#-----------Compute MI---------------#
scaler = StandardScaler()
coeff_scaled = scaler.fit_transform(coeffs)

sigma_points_scaled = normalize_signal(sigma_points, scaler)

mi,Hx,_,_ = entropy_r(coeff_scaled,sigma_points_scaled,base=np.e,k=5, vol=False)

norm_mi = mi/Hx

print(f'Normalized Information: {norm_mi}')
