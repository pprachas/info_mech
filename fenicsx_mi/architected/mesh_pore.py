import numpy as np
import matplotlib.pyplot as plt 
import gmsh
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from pathlib import Path

def generate_pore(L = 100, c1=0.0, c2=0.0, phi = 0.3,num_array = 3, f_name = 'pore.xdmf', char_length_rat = 20):
    gmsh.initialize()
    gmsh.model.add('pores')
    L = 100 # length of solid

    # pore shape
    L0 = L/num_array # length of array

    # Rectangle
    # rec = np.array([[-L0/2,-L0/2],[-L0/2,L0/2], [L0/2,L0/2],[L0/2,-L0/2],[-L0/2,-L0/2]])
    # Pore
    # c1_all = np.linspace(-0.2,0.2,3)
    # c2_all = np.linspace(-0.2,0.2,3)

    theta = np.linspace(0,2*np.pi,1000)
    r0 = (L0*np.sqrt(2*phi))/np.sqrt(np.pi*(2+c1**2+c2**2))
    r = r0*(1+c1*np.cos(4*theta) + c2*np.cos(8*theta))

    x_all = r*np.cos(theta)
    y_all = r*np.sin(theta)

    # --- starting corner at -L/2, 0 --- #
    x0 = -L/2 + L0/2
    y0 = L0/2
    #-------- Dummy unit cell centered around 0,0------------#
    rectangle = gmsh.model.occ.addRectangle(0,0,0,L0/2,L0/2)

    points = []
    for ii in range(len(x_all)):
        points.append(gmsh.model.occ.addPoint(x_all[ii],y_all[ii],0.0))

    curve = gmsh.model.occ.addBSpline(points)
    curve_loop = gmsh.model.occ.addCurveLoop([curve])

    pore = gmsh.model.occ.addPlaneSurface([curve_loop])

    cell_portion,_ = gmsh.model.occ.cut([(2,rectangle)], [(2,pore)])

    cell_x = gmsh.model.occ.copy(cell_portion)

    # reflect over x
    gmsh.model.occ.mirror(cell_x, 0, 1, 0,0)
    cell_x,_ = gmsh.model.occ.fragment(cell_portion,cell_x)

    # need to copy -> flip - fragemnt
    cell_y = gmsh.model.occ.copy(cell_x)
    gmsh.model.occ.mirror(cell_y, 1, 0, 0,0)
    cell,_ = gmsh.model.occ.fragment(cell_x,cell_y)

    #------ build array of cells ------#
    cells = []
    for ii in range(num_array):
        for jj in range(num_array):
            copied_cell = gmsh.model.occ.copy(cell)
            dx = x0 + ii * L0
            dy = y0 + jj * L0
            gmsh.model.occ.translate(copied_cell, dx, dy, 0)
            cells.extend(copied_cell)
    if num_array == 1:
        fused = cells
    else:
        fused,_ = gmsh.model.occ.fragment(cells,[])

    gmsh.model.occ.synchronize()
    #--- physical group --- #
    gmsh.model.addPhysicalGroup(2, [c[1] for c in fused])

    # --- meshing options --- #
    gmsh.model.occ.synchronize()
    
    lc = L0 / (char_length_rat)
    for dim, tag in gmsh.model.getEntities(0):
        gmsh.model.mesh.setSize([(0, tag)], lc)

    # gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", size_factor)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)  
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    gmsh.model.mesh.generate(2)

    # --- convert to dolfinx mesh --- #
    mesh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model(), MPI.COMM_WORLD, 0, gdim=2)

    mesh.name = "mesh"
    cell_markers.name = "cells"
    facet_markers.name = "markers"

    with XDMFFile(MPI.COMM_WORLD, f_name, "w") as file:
        file.write_mesh(mesh)
        file.write_meshtags(cell_markers, mesh.geometry)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        mesh.topology.create_connectivity(0, 1)
        file.write_meshtags(facet_markers, mesh.geometry)
    gmsh.finalize()

if __name__ == "__main__":
    num_arrays = np.linspace(1,9,9, dtype=int)
    phi_all = [0.3]
    # c1 = 0 and c2 = 0 is circular pores
    c1_all = [0.0]
    c2_all = [0.0]
    char_length_rat = 40

    for num_array in num_arrays:
        for count,phi in enumerate(phi_all):
            for ii,c1 in enumerate(c1_all):
                for jj,c2 in enumerate(c2_all):
                    f_name = f'mesh/pore/pore{num_array}.xdmf'
                    generate_pore(f_name = f_name,num_array = num_array, c1=c1, c2=c2, phi=phi, char_length_rat=char_length_rat)
  