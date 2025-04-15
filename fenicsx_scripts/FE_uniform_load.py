import numpy as np
from ufl import sym, grad, Identity, tr, inner, Measure, TestFunction, TrialFunction

from mpi4py import MPI
from dolfinx import fem, io, plot
import dolfinx.fem.petsc
from dolfinx.mesh import create_rectangle, CellType
import os, sys

import pyvista

L = 1e4
H = 1e4
n_el = 100
domain = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([-L/2,-H]), np.array([L/2, 0])],
    [n_el,n_el],
    cell_type=CellType.quadrilateral,
)
cells, types, x = plot.vtk_mesh(domain)
grid = pyvista.UnstructuredGrid(cells, types, x)

# plotter.add_mesh(grid, style="wireframe")
if sys.platform == "linux" and (os.getenv("CI") or pyvista.OFF_SCREEN):
    pyvista.start_xvfb(0.05)

plotter = pyvista.Plotter()
plotter.show()


