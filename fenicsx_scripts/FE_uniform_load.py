import numpy as np
from ufl import sym, grad, Identity, tr, inner, Measure, TestFunction, TrialFunction
import ufl
import petsc4py.PETSc as PETSc

from mpi4py import MPI
from dolfinx import fem, io, plot
import dolfinx.fem.petsc
from dolfinx.mesh import create_rectangle, CellType
import os, sys

import pyvista

L = 1e4
H = 1e4

a = 500.
P = -1/(2*a)
n_el = 100

domain = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([-L/2,-H]), np.array([L/2, 0])],
    [n_el,n_el],
    cell_type=CellType.triangle,
    diagonal = dolfinx.cpp.mesh.DiagonalType.crossed
)

dim = domain.topology.dim
degree = 2 
V = fem.functionspace(domain, ("Lagrange", degree, (dim,))) # Function space as Lagrange shape functions with dimensions dim

#-----------Marking Facets for Boundary conditions----------------#
def left(x):
    return np.isclose(x[0], -L/2, 1e-3)
def right(x):
    return np.isclose(x[0], L/2, 1e-3)
def top_f(x): # only for prescribed force
    return np.isclose(x[1], 0, 1e-3) & ((x[0] >= -a) & (x[0] <= a))
def bot(x):
    return np.isclose(x[1], -H, 1e-3)

# getting entities
left_facets = dolfinx.mesh.locate_entities_boundary(domain,dim-1,left)
right_facets = dolfinx.mesh.locate_entities_boundary(domain,dim-1,right)
top_f_facets = dolfinx.mesh.locate_entities_boundary(domain,dim-1,top_f)
bot_facets = dolfinx.mesh.locate_entities_boundary(domain,dim-1,bot)

num_facets = domain.topology.index_map(dim - 1).size_local # get number of facets in system
markers = np.zeros(num_facets, dtype=np.int32) # vector of markers -- initialize as 0
# mark facets
markers[left_facets] = 1 
markers[right_facets] = 2
markers[top_f_facets] = 3
markers[bot_facets] = 4

facet_marker = dolfinx.mesh.meshtags(domain, dim - 1, np.arange(num_facets, dtype=np.int32), markers) # mark facets

# Mark Dirichlet BCs
Vx, _ = V.sub(0).collapse() # x dof subspace
Vy, _ = V.sub(1).collapse() # y dof subspace
left_dofs = dolfinx.fem.locate_dofs_topological((V.sub(0), Vx), dim-1, facet_marker.find(1))
right_dofs = dolfinx.fem.locate_dofs_topological((V.sub(0), Vx), dim-1, facet_marker.find(2))
bot_dofs = dolfinx.fem.locate_dofs_topological((V.sub(1), Vy), dim-1, facet_marker.find(4))

u0x = fem.Function(Vx)
u0y = fem.Function(Vy)

bcs = [
    fem.dirichletbc(u0x, left_dofs, V.sub(0)),
    fem.dirichletbc(u0x, right_dofs, V.sub(0)),
    fem.dirichletbc(u0y, bot_dofs, V.sub(1))
]

#------------------Kinematics-----------------#
def epsilon(v):
    return sym(grad(v))

#---------------Constitutive Model------------#
E = fem.Constant(domain, 1.0)
nu = fem.Constant(domain, 0.0)

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)

def sigma(v):
    return lmbda * tr(epsilon(v)) * Identity(dim) + 2 * mu * epsilon(v)

#---------------Define weak form----------------#

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_marker) # measures

t = fem.Constant(domain, (0.0, P)) # traction force

# setup function spaces
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
# Bilinear form
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx 
L = ufl.dot(t, v) * ds(3) # no body force --  only traction

test = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L))
print(test[np.nonzero(test[:])])

#--------solve problem------------------#
uh= fem.Function(V, name="Displacement") 
problem = fem.petsc.LinearProblem(a, L, u=uh, bcs=bcs)
problem.solve()
print('Starting linear solve')
uh.x.scatter_forward()
print('Linear solve done!')
#------------Interpolate stress--------------#
W = fem.functionspace(domain, ('DG', 1))
sigma_expr = fem.Expression(sigma(uh)[1,1], W.element.interpolation_points()) # only interpolate sigma_yy
sigma_sol = fem.Function(W)
sigma_sol.interpolate(sigma_expr)
sigma_sol.x.scatter_forward()

#----------Plotter----------------------#
pyvista.start_xvfb()
p = pyvista.Plotter(off_screen=True, window_size=(800, 600), shape=(1,1),
                    lighting=None)

topology, cells, geometry = plot.vtk_mesh(W) # make sure function space is the same as stress function space
grid = pyvista.UnstructuredGrid(topology, cells, geometry)

#grid['u'] = uh.x.array.reshape((geometry.shape[0], 2)) # add displacement to plotter

grid.point_data['sigma_yy'] = sigma_sol.x.array # add stress data
grid.set_active_scalars('sigma_yy')

p.add_mesh(grid, show_edges=True)
p.view_xy()

p.screenshot('sigma_yy.png')

#--------Plot BCs------------#
pyvista.start_xvfb()
p = pyvista.Plotter(off_screen=True, window_size=(800, 600), shape=(1,1),
                    lighting=None)

exclude_entities = facet_marker.find(0)
marker_idx = np.full_like(facet_marker.values, True, dtype=np.bool_)
marker_idx[exclude_entities] = False

topology, cells, geometry = plot.vtk_mesh(domain,facet_marker.dim, facet_marker.indices[marker_idx]) # make sure function space is the same as stress function space
grid = pyvista.UnstructuredGrid(topology, cells, geometry)
grid.cell_data["Marker"] = facet_marker.values[marker_idx]
p.add_mesh(grid)
p.view_xy()

p.screenshot('facets.png')

#----------Extract solution at x=0------------#
bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
# points to extract solution
y_points = y = np.linspace(0,-1000,100)
x_points = np.zeros(len(y))
z_points = np.zeros(len(y))
points = np.stack([x_points, y_points, z_points], axis=-1)

# determine cell that the points live in
potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, points)

colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, potential_colliding_cells, points)

# get only one cell
points_on_proc = []
cells = []
for ii, point in enumerate(points):
    if len(colliding_cells.links(ii)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(ii)[0])

points_on_proc = np.array(points_on_proc, dtype=np.float64).reshape(-1, 3)
cells = np.array(cells, dtype=np.int32)
sigma_values = sigma_sol.eval(points_on_proc, cells)

print(np.array([y_points,sigma_values.reshape(-1)]))

print(uh.x)
print(dolfinx.la.norm(uh.x), len(uh.x.array))

np.savetxt('FEA_results.txt',np.array([y_points,sigma_values.reshape(-1)]))