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

import matplotlib.pyplot as plt


import sympy as sp
sys.path.append('../..')
from utils.symbolic import sym_legendre_series
from utils.fea import eval_points, run_fea, import_mesh, compute_stress, compute_reaction_force
from utils.meshing import rectangle_mesh

sample = int(sys.argv[1])

react_all = []
max_stress_all = []
num_nodes_mesh = []

n_nodes_prev = 3
for ii in range(10):

    L = 100
    H = int(L*4)

    # Create mesh
    n_nodes = 2*n_nodes_prev - 1 # number of nodes on the boundaries with quad mesh
    f_mesh = f'mesh/rectangle{int(H)}.xdmf'
    rectangle_mesh(L,H,n_nodes,f_mesh)

    num_coeff=3
    coeff = np.loadtxt(f'../../half_space/legendre/full/coeffs/legendre_coeffs{num_coeff}.txt')[sample,:num_coeff] # legendre coefficients

    domain=import_mesh(f_mesh)
    a,l,u,bcs,u0 = run_fea(domain,L,H, coeff)

    #------------Interpolate stress--------------#
    W = fem.functionspace(domain, ('DG', 1, (2,2)))
    E = fem.Constant(domain, 1.0)
    nu = fem.Constant(domain, 0.0)
    sigma_u = compute_stress(W,u, E, nu)
    sigma_xx,sigma_xy,sigma_yx, sigma_yy = sigma_u.split() # split into components


    # check reaction force
    V = fem.functionspace(domain, ('Lagrange', 2,(2,)))

    reaction_force = compute_reaction_force(V,a,u,l,bcs,u0)
    react_all.append(reaction_force)
    # max magnitiude sigma_y:
    max_stress_all.append(np.max(np.abs(sigma_yy.x.array[:])))

    # get number of nodes
    num_nodes_mesh.append(domain.topology.index_map(0).size_global)
    plt.figure()
    plt.plot(num_nodes_mesh,max_stress_all)
    plt.xlabel('number of nodes')
    plt.ylabel(r'max $|\sigma_yy|$')
    plt.title(r'Convergence max $|\sigma_yy|$')
    plt.savefig(f'refinement{sample}_max_stress.png')

    plt.figure()
    plt.plot(num_nodes_mesh,react_all)
    plt.xlabel('number of nodes')
    plt.ylabel('Reaction force')
    plt.title('Convergence Reaction Force')
    plt.savefig(f'refinement{sample}_reaction_force.png')

    n_nodes_prev = n_nodes

    # COnvergence Criteria

    if len(react_all) > 1:
        # percent differences
        react_diff = np.abs(1-react_all[-1])
        stress_diff = np.abs((max_stress_all[-2]-max_stress_all[-1])/max_stress_all[-2])

        print(react_diff,stress_diff)

        tol = 0.01
        if react_diff < tol and stress_diff < tol:
            print(n_nodes)
            break
        

