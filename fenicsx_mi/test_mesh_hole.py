import gmsh
import pyvista as pv
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

gmsh.initialize()

L = 100
H = 100
n_nodes = 21
R = 30 # radius of hole

rat = (H/2-R)/(H/2)


#
# create box
p_rec_np = [[L/2,H],[-L/2,H],[-L/2,0],[L/2,0]]





# point for center of the circle
p_center = gmsh.model.occ.addPoint(0, H/2, 0, -1)

# points for arclengths and rectangle
thetas = np.linspace(np.pi/4,7*np.pi/4,4)
p_arc = [] # points that form the beginning and end of arcs
p_rec = [] # points for rectangle

for ii in range(len(thetas)):
    p_arc.append(gmsh.model.occ.addPoint(R*np.cos(thetas[ii]),(H/2) + R*np.sin(thetas[ii]) , 0, -1))
    p_rec.append(gmsh.model.occ.addPoint(p_rec_np[ii][0],p_rec_np[ii][1], 0,-1))
# construct lines and surfaces
square = []
circle = []
connect = []
for ii in range(len(thetas)):
    connect.append(gmsh.model.occ.addLine(p_arc[ii-1],p_rec[ii-1]))
    square.append(gmsh.model.occ.addLine(p_rec[ii-1],p_rec[ii]))
    circle.append(gmsh.model.occ.addCircleArc(p_arc[ii],p_center,p_arc[ii-1]))

surf = []
for ii in range(len(thetas)):
    loop = gmsh.model.occ.addCurveLoop([connect[ii-1], square[ii-1], connect[ii], circle[ii-1]])
    surf.append(gmsh.model.occ.addPlaneSurface([loop]))

gmsh.model.occ.synchronize()

for ii in range(len(thetas)):
    gmsh.model.mesh.setTransfiniteCurve(connect[ii-1],int(n_nodes*rat))
    gmsh.model.mesh.setTransfiniteCurve(square[ii-1],n_nodes)
    gmsh.model.mesh.setTransfiniteCurve(circle[ii-1],n_nodes)
    gmsh.model.mesh.setTransfiniteSurface(surf[ii-1])
    
    
gmsh.model.occ.fragment([gmsh.model.getEntities(1)[0]],gmsh.model.getEntities(1)[1:])

gmsh.model.occ.fragment([gmsh.model.getEntities(2)[0]],gmsh.model.getEntities(2)[1:])

gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2,[entry[1] for entry in gmsh.model.getEntities(2)])
gmsh.option.setNumber('Mesh.Algorithm', 8)
gmsh.option.setNumber('Mesh.RecombineAll', 1)
gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
gmsh.model.mesh.generate(2)

# convert to xdmf using dolifinx functions
mesh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model(), MPI.COMM_WORLD, rank=0,gdim=2)

mesh.name = 'mesh'

cell_markers.name = 'cells'
facet_markers.name = 'markers'
f_name = 'test_hole.xdmf'

with XDMFFile(MPI.COMM_WORLD, f_name, 'w') as file:
    file.write_mesh(mesh)
    file.write_meshtags(cell_markers, mesh.geometry)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh.topology.create_connectivity(0, 1)
    file.write_meshtags(facet_markers, mesh.geometry)
gmsh.finalize()



