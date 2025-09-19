import numpy as np
from dolfinx import fem, plot
from dolfinx.io import XDMFFile
import sys

sys.path.append('../..')

from utils.fea import import_mesh
import pyvista

L = 100
H_all = [4*L,2*L,L,L/2,L/4]

for H in H_all:
# Import mesh
    f_mesh = f'mesh/rectangle{int(H)}.xdmf'
    domain = import_mesh(f_mesh)


    # Import MI results

    mi = np.loadtxt(f'mi/H{int(H)}.txt')

    print(mi.shape)

    #------Plot MI with interpolation functions------#
    W = fem.functionspace(domain,('DG',1))


    pyvista.start_xvfb()
    p = pyvista.Plotter(off_screen=True, shape=(1,1),
                        lighting=None)

    topology, cells, geometry = plot.vtk_mesh(W) # make sure function space is the same as stress function space
    grid = pyvista.UnstructuredGrid(topology, cells, geometry)

    #grid['u'] = uh.x.array.reshape((geometry.shape[0], 2)) # add displacement to plotter

    grid.point_data['mi'] = mi # add stress data
    grid.set_active_scalars('mi')

    p.add_mesh(grid, scalar_bar_args={'vertical': True, 'font_family': 'times'})
    p.show_bounds(font_family = 'times',font_size=12)
    p.view_xy()

    p.save_graphic(f'mi/H{int(H)}.pdf')


