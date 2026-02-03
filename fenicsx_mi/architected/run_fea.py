import numpy as np

from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile, gmshio
import os, sys

import pyvista

import sympy as sp
sys.path.append('../..')
from utils.symbolic import sym_legendre_series
from utils.fea import eval_points, run_linear_fea_traction_batch, import_mesh, compute_stress, compute_reaction_force

from pathlib import Path

# Read mesh
L = 100
H = 100

sigma_points = []


num_array = int(sys.argv[1])

run = int(sys.argv[2])

geom = ['slit','pore'][int(sys.argv[3])]
num_coeff= 6 
load_case = 'full'


f_mesh = f'mesh/{geom}/{geom}{num_array}.xdmf'
load_case = 'full'



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

# f_path = f'pointwise_sigma/pores/coeff{num_coeff}/{load_case}/phi2/array{num_array}/c1-{c1}_c2-{c2}'

f_path = f'pointwise_sigma/{geom}{num_array}/'
Path(f_path).mkdir(parents=True, exist_ok=True)
np.savez_compressed(f'{f_path}/{geom}{run}.npz', sigma_points=sigma_points)

# np.savez_compressed(f'{f_path}/lattice{run}.npz', sigma_points=sigma_points)

# test = np.load(f'{f_path}/lattice{run}.npz')['sigma_points']
# print(test.shape)

