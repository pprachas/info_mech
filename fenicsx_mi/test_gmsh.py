import gmsh
import pyvista as pv 
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from dolfinx.plot import vtk_mesh

gmsh.initialize()

# points in square
L = 100
H = 100

num_nodes = 25 # number of nodes per direction

box=gmsh.model.occ.addRectangle(-L/2,0,0,L,H,0)
gmsh.model.occ.synchronize()

# gmsh.model.mesh.setTransfiniteAutomatic()
lines = gmsh.model.getEntities(1)
for line in lines:
    gmsh.model.mesh.setTransfiniteCurve(line[1],num_nodes)
gmsh.model.mesh.setTransfiniteSurface(box)
gmsh.model.addPhysicalGroup(2,[box])

gmsh.option.setNumber('Mesh.Algorithm', 8)
gmsh.option.setNumber('Mesh.RecombineAll', 1)
gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)

gmsh.model.mesh.generate(2)

f_name = 'test.msh'
gmsh.write(f_name)


# gmsh to xdmf
mesh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model(), MPI.COMM_WORLD, rank=0,gdim=2)

mesh.name = 'mesh'

cell_markers.name = 'cells'
facet_markers.name = 'markers'

f_name = 'test.xdmf'
# Convert gmsh to xdmf
with XDMFFile(MPI.COMM_WORLD, f_name, 'w') as file:
    file.write_mesh(mesh)
    file.write_meshtags(cell_markers, mesh.geometry)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh.topology.create_connectivity(0, 1)
    file.write_meshtags(facet_markers, mesh.geometry)

with XDMFFile(MPI.COMM_WORLD, f_name, 'r') as xdmf:
    mesh = xdmf.read_mesh(name='mesh')

tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, x = vtk_mesh(mesh, tdim)

grid = pv.UnstructuredGrid(topology, cell_types, x)

pv.start_xvfb()
p = pv.Plotter(off_screen=True,window_size=[800, 800])
p.add_mesh(grid)
surface = grid.separate_cells().extract_surface(nonlinear_subdivision=4)
edges = surface.extract_feature_edges()
p.add_mesh(edges, style="wireframe", color="black", line_width=3)

p.view_xy()

p.screenshot("test_mesh.png")