import numpy as np
from dolfinx import fem, plot
from dolfinx.io import XDMFFile
from mpi4py import MPI
import sys 
from collections import deque

import ufl
import dolfinx
from ufl import sqrt, atan2, cos, sin, pi,as_vector, as_tensor, conditional
from fea import eval_points, run_linear_fea_traction, import_mesh, compute_stress, compute_reaction_force

from pathlib import Path
from mesh_pore import generate_pore
import pyvista 
import matplotlib.pyplot as plt
import sys

num_coeff = 1
num_array = 3

L = 100
H = 100 

geom = ['pore', 'slit'][int(sys.argv[1])]
f_path = f'principal/{geom}/{geom}{num_array}'
Path(f_path).mkdir(parents=True, exist_ok=True)
f_mesh =  f'mesh/{geom}/{geom}{num_array}.xdmf'

coeffs = np.array([1e-20]) # essientially 0 but 0 gives interpolation errors in FEniCSx

domain = import_mesh(f_mesh)

load_case = 'even'
a,l,V,u,bcs,u0 = run_linear_fea_traction(domain,L,H, coeffs, load_case)

# Compute function space for stress:
W1 = fem.functionspace(domain, ('DG', 1, (2,2)))
E = fem.Constant(domain, 100.0)
nu = fem.Constant(domain, 0.0)

sigma_u = compute_stress(W1,u, E, nu)
#-----------Extract sigma_yy component----#
sigma_xx,sigma_xy,sigma_yx, sigma_yy = sigma_u.split() 

shear = (sigma_xy + sigma_yx) / 2 # make sure that it is symmetric

sigma_p1_ufl = (sigma_xx+sigma_yy)/2 + sqrt(((sigma_xx-sigma_yy)/2)**2 + shear**2)
sigma_p2_ufl = (sigma_xx+sigma_yy)/2 - sqrt(((sigma_xx-sigma_yy)/2)**2 + shear**2)

#------------Principal Components-----------------#
W = fem.functionspace(domain,('DG',1))
W_vec = fem.functionspace(domain, ('DG', 1, (2,)))

sigma_p1_expr = fem.Expression(sigma_p1_ufl, W.element.interpolation_points())
sigma_p1 = fem.Function(W)
sigma_p1.interpolate(sigma_p1_expr)

sigma_p2_expr = fem.Expression(sigma_p2_ufl, W.element.interpolation_points())
sigma_p2 = fem.Function(W)
sigma_p2.interpolate(sigma_p2_expr)

theta = 0.5*atan2((2*sigma_xy),(sigma_xx-sigma_yy))

# Rotation transformation

v1_ufl = as_vector([cos(theta), sin(theta)])
v2_ufl = as_vector([cos(theta+pi/2), sin(theta+pi/2)])

# assign correct directions
p1_dir_ufl_ub = conditional(sigma_xx > sigma_yy , v1_ufl,v2_ufl)
p2_dir_ufl_ub = conditional(sigma_xx < sigma_yy , v1_ufl,v2_ufl)

# bias dowwards
p1_dir_ufl = conditional(p1_dir_ufl_ub[1] <= 0 , p1_dir_ufl_ub,-p1_dir_ufl_ub)
p2_dir_ufl = conditional(p2_dir_ufl_ub[1] <= 0 , p2_dir_ufl_ub,-p2_dir_ufl_ub)

p1_dir_expr = fem.Expression(p1_dir_ufl, W_vec.element.interpolation_points())
p1_dir = fem.Function(W_vec)
p1_dir.interpolate(p1_dir_expr)

p2_dir_expr = fem.Expression(p2_dir_ufl, W_vec.element.interpolation_points())
p2_dir = fem.Function(W_vec)
p2_dir.interpolate(p2_dir_expr)

# maximum magnitude
max_dir_ufl_ub = conditional(abs(sigma_p1) > abs(sigma_p2),p1_dir, p2_dir)
max_dir_ufl = conditional(max_dir_ufl_ub[1] <= 0 , max_dir_ufl_ub,-max_dir_ufl_ub) # bias direction downwards
max_dir_ufl_mask = conditional(abs(sigma_p1) > abs(sigma_p2),0,1) # color mask for visualization

max_dir_expr = fem.Expression(max_dir_ufl, W_vec.element.interpolation_points())
max_dir_mask_expr = fem.Expression(max_dir_ufl_mask, W_vec.element.interpolation_points())
max_dir = fem.Function(W_vec)
max_dir.interpolate(max_dir_expr)

max_dir_mask = fem.Function(W)
max_dir_mask.interpolate(max_dir_mask_expr)

# bias max direction
sigma_max_ufl = conditional(abs(sigma_p1) > abs(sigma_p2),sigma_p1, sigma_p2)
sigma_max_expr = fem.Expression(sigma_max_ufl, W.element.interpolation_points())
sigma_max = fem.Function(W)
sigma_max.interpolate(sigma_max_expr)

# smooth vector projection
V_vec = fem.functionspace(domain, ('CG', 2, (2,))) # smooth CG function
u_proj = ufl.TrialFunction(V_vec)
v_proj = ufl.TestFunction(V_vec)

a_proj = ufl.inner(u_proj,v_proj)*ufl.dx
l_proj = ufl.inner(max_dir,v_proj)*ufl.dx

petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "cholesky", 
    "pc_factor_mat_solver_type": "mumps"
    }

problem = dolfinx.fem.petsc.LinearProblem(a_proj, l_proj, petsc_options = petsc_options)
max_dir_proj = problem.solve()


# Smooth mask projection
V = fem.functionspace(domain,('CG',2)) # smooth CG function
u_proj = ufl.TrialFunction(V)
v_proj = ufl.TestFunction(V)

a_proj = ufl.inner(u_proj,v_proj)*ufl.dx
l_proj = ufl.inner(max_dir_mask,v_proj)*ufl.dx

petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "cholesky", 
    "pc_factor_mat_solver_type": "mumps"
    }

problem = dolfinx.fem.petsc.LinearProblem(a_proj, l_proj, petsc_options = petsc_options)
max_dir_mask_proj = problem.solve()

pyvista.start_xvfb()
#------plot maximum magnitude streamline projected orientation----------------#
p = pyvista.Plotter(off_screen=True, shape=(1,1),lighting=None)
topology, cells, geometry = plot.vtk_mesh(V_vec) # make sure function space is the same as stress function space
grid = pyvista.UnstructuredGrid(topology, cells, geometry)

vector = np.zeros((geometry.shape[0],3))
vector[:,:len(p1_dir)] = max_dir_proj.x.array.reshape(geometry.shape[0], len(p2_dir))
grid.point_data['vector'] = vector

streamline1=grid.streamlines('vector',pointa=(-L/2,L-1e-3,0),pointb=(L/2,L-1e-3,0),integration_direction='both',
                                initial_step_length=0.5,max_time=10000.0,terminal_speed=1e-6, 
                                max_error = 1e-3,n_points=50, compute_vorticity = False,
                                interpolator_type = 'c', integrator_type = 4) # step size 5 for slit; step size 0.5 for pore 

p.add_mesh(grid, color='k',  opacity=0.3)
p.add_mesh(streamline1, color = 'r', line_width = 3.0)

points_x = np.linspace(-L/2,L/2,6)
points = np.zeros((6,3))
points[:,0] = points_x

points_x2 = np.linspace(-L/2,L/2,4)
points2 = np.zeros((4,3))
points2[:,0] = points_x2

p.add_points(points, color = 'k', point_size=20)

p.view_xy()
p.image_scale = 25
p.save_graphic(f'{f_path}/max_proj_streamline.pdf')

#-----------Plot results maximum orientation-------------------#

p = pyvista.Plotter(off_screen=True, shape=(1,1),lighting=None)

topology, cells, geometry = plot.vtk_mesh(W_vec) # make sure function space is the same as stress function space
grid = pyvista.UnstructuredGrid(topology, cells, geometry)

vector = np.zeros((geometry.shape[0],3))
vector[:,:len(p2_dir)] = max_dir.x.array.reshape(geometry.shape[0], len(p2_dir))

grid.point_data['vector'] = vector
grid.point_data['mask'] = max_dir_mask.x.array

arrow_geom = pyvista.Arrow(tip_length=0.7, tip_radius=0.3, shaft_radius=0.05)
glyphs=grid.glyph(orient='vector', geom = arrow_geom, factor=5.0, tolerance = 0.03)

p.add_mesh(grid, color='grey',  opacity=0.3)
p.add_mesh(glyphs, scalars = 'mask', cmap = 'coolwarm')

p.view_xy()
p.image_scale = 25
p.save_graphic(f'{f_path}/max_DG.pdf')
#-----------Plot results projected maximum orientation-------------------#

p = pyvista.Plotter(off_screen=True, shape=(1,1),lighting=None)

topology, cells, geometry = plot.vtk_mesh(V_vec) # make sure function space is the same as stress function space
grid = pyvista.UnstructuredGrid(topology, cells, geometry)

vector = np.zeros((geometry.shape[0],3))

vector[:,:len(p2_dir)] = max_dir_proj.x.array.reshape(geometry.shape[0], len(p2_dir))

grid.point_data['vector'] = vector
grid.point_data['mask'] = max_dir_mask_proj.x.array > 0 # make binary mask

glyphs=grid.glyph(orient='vector', geom = arrow_geom, factor=5.0, tolerance = 0.03)


p.add_mesh(grid, color='grey',  opacity=0.3)
p.add_mesh(glyphs, scalars = 'mask', cmap = 'coolwarm')

p.view_xy()
p.image_scale = 25
p.save_graphic(f'{f_path}/max_CG.pdf')

#-----------Plot results p1-------------------#
p = pyvista.Plotter(off_screen=True, shape=(1,1),
                    lighting=None)

topology, cells, geometry = plot.vtk_mesh(W_vec) # make sure function space is the same as stress function space
grid = pyvista.UnstructuredGrid(topology, cells, geometry)

vector = np.zeros((geometry.shape[0],3))
vector[:,:len(p1_dir)] = p1_dir.x.array.reshape(geometry.shape[0], len(p1_dir))

grid.point_data['vector'] = vector
grid.point_data['p1'] = sigma_p1.x.array
grid.set_active_vectors('vector')

glyphs=grid.glyph(orient='vector', geom = arrow_geom, factor=5.0, tolerance = 0.03)

p.add_mesh(grid, color='grey',  opacity=0.5)
p.add_mesh(glyphs, color = (0.23, 0.299, 0.754))
p.view_xy()
p.image_scale = 25  
p.save_graphic(f'{f_path}/p1.pdf')

#-----------Plot results p2-------------------#
p = pyvista.Plotter(off_screen=True, shape=(1,1),
                    lighting=None)

topology, cells, geometry = plot.vtk_mesh(W_vec) # make sure function space is the same as stress function space
grid = pyvista.UnstructuredGrid(topology, cells, geometry)

vector = np.zeros((geometry.shape[0],3))
vector[:,:len(p2_dir)] = p2_dir.x.array.reshape(geometry.shape[0], len(p2_dir))
grid.point_data['vector'] = vector
grid.point_data['p2'] = sigma_p2.x.array


# grid.set_active_vectors('vector')

glyphs=grid.glyph(orient='vector', geom = arrow_geom, factor=5.0, tolerance = 0.03)

p.add_mesh(grid, color='grey',  opacity=0.3)
p.add_mesh(glyphs, color = (0.706, 0.016, 0.15))

p.view_xy()
p.image_scale = 25
p.save_graphic(f'{f_path}/p2.pdf')
