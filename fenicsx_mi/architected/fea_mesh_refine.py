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

import sympy as sp
from symbolic import sym_legendre_series
from fea import eval_points, run_linear_fea_traction_batch, import_mesh, compute_stress, compute_reaction_force

from pathlib import Path

# Read mesh
L = 100
H = 100

sigma_points = []


num= int(sys.argv[1])

run = int(sys.argv[2])
geom = ['pore','slit'][int(sys.argv[3])]

f_mesh = f'mesh_refinement/{geom}/{geom}9_{num}.xdmf'
load_case = 'even'
num_coeff = 6


coeffs_all = np.loadtxt(f'../../half_space/legendre/{load_case}/coeffs/legendre_coeffs{num_coeff}.txt')[:,:num_coeff] # legendre coefficients
coeffs = np.split(coeffs_all,5,axis=0)[run]


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

f_path = f'mesh_refinement/sigma/{geom}/{geom}9_{num}'
Path(f_path).mkdir(parents=True, exist_ok=True)


print(sigma_points)

print(sigma_points.shape)
np.savez_compressed(f'{f_path}/run{run}.npz', sigma_points=sigma_points)