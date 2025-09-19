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
sys.path.append('../..')
from utils.symbolic import sym_legendre_series
from utils.fea import eval_points, run_fea, import_mesh, compute_stress, compute_reaction_force
from utils.meshing import rectangle_mesh

from pathlib import Path


sigma_yy_all = []
# geometry
L = 100
H=int(sys.argv[1])
run = int(sys.argv[2])
# loiad coefficients
num_coeff=4
coeff_all = np.loadtxt(f'../../half_space/legendre/full/coeffs/legendre_coeffs{num_coeff}.txt')[:,:num_coeff] # legendre coefficients
coeff_run = np.split(coeff_all,100,axis=0)[run]

# Import mesh
f_mesh = f'mesh/rectangle{H}.xdmf'
domain=import_mesh(f_mesh)

for coeff_idx in range(len(coeff_run)):
    coeff = coeff_run[coeff_idx]

    a,l,u,bcs,u0 = run_fea(domain,L,H, coeff)

    #------------Interpolate stress--------------#
    W = fem.functionspace(domain, ('DG', 1, (2,2)))
    E = fem.Constant(domain, 1.0)
    nu = fem.Constant(domain, 0.0)
    sigma_u = compute_stress(W,u, E, nu)
    #-----------Extract sigma_yy component----#
    W0 = fem.functionspace(domain,('DG',1))
    sigma_yy = fem.Function(W0)
    # dummy = sigma_u[1,1]
    # sigma_yy.interpolate(dummy)
    sigma_xx,sigma_xy,sigma_yx, sigma_yy = sigma_u.split() 
    sigma_yy_all.append(sigma_yy.collapse().x.array)

sigma_yy_all = np.array(sigma_yy_all)

f_path = f'sigma_yy/H{H}'
Path(f_path).mkdir(parents=True, exist_ok=True)
np.savetxt(f'{f_path}/sigma_yy{run}.txt', sigma_yy_all)