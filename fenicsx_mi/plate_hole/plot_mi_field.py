import numpy as np
from dolfinx import fem, plot
from dolfinx.io import XDMFFile
import sys

sys.path.append('../..')

from utils.fea import import_mesh
import pyvista

H = 100
R_all = [H/2.5] #[H/2.5,H/5,H/10,H/20]

for R in R_all:
# Import mesh
    f_mesh = f'mesh/plate_hole{int(R)}.xdmf'
    domain = import_mesh(f_mesh)


    # Import MI results

    mi = np.loadtxt(f'mi/R{int(R)}.txt')

    print(mi.shape)

    #------Plot MI with interpolation functions------#
    W = fem.functionspace(domain,('DG',1))


    pyvista.start_xvfb()
    p = pyvista.Plotter(off_screen=True, window_size=(800, 600), shape=(1,1),
                        lighting=None)

    topology, cells, geometry = plot.vtk_mesh(W) # make sure function space is the same as stress function space
    grid = pyvista.UnstructuredGrid(topology, cells, geometry)

    #grid['u'] = uh.x.array.reshape((geometry.shape[0], 2)) # add displacement to plotter

    grid.point_data['mi'] = mi # add stress data
    grid.set_active_scalars('mi')

    p.add_mesh(grid)
    p.view_xy()


    p.save_graphic(f'mi/R{int(R)}.pdf')


