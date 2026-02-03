import gmsh
from dolfinx.io import gmshio, XDMFFile
from mpi4py import MPI
import numpy as np

def generate_slit_mesh(
    L=100, H=100,
    num_x=3, num_y=1,
    gap_fraction=0.05,
    f_name="slit_mesh.xdmf",
    char_length_rat=20,
):
    """
    Generate 2D mesh [-L/2,L/2] x [0,H] with rectangular slits.
    Uniform gap between all rectangles, including edges.
    """
    gmsh.initialize()
    gmsh.model.add("slit_domain")

    domain = gmsh.model.occ.addRectangle(-L/2, 0, 0, L, H)

    # uniform gaps
    gap_x = gap_fraction * L
    gap_y = gap_fraction * H

    # compute rectangle size so gaps are uniform and edges equal
    rect_width = (L - (num_x + 1) * gap_x) / num_x
    rect_height = (H - (num_y + 1) * gap_y) / num_y

    slits = []
    for ii in range(num_x):
        x0 = -L/2 + gap_x + ii*(rect_width + gap_x)
        for jj in range(num_y):
            y0 = 0 + gap_y + jj*(rect_height + gap_y)
            slit = gmsh.model.occ.addRectangle(x0, y0, 0, rect_width, rect_height)
            slits.append((2, slit))

    # subtract slits
    cut, _ = gmsh.model.occ.cut([(2, domain)], slits)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [c[1] for c in cut])

    # mesh options
    lc = min(gap_x,gap_y) / char_length_rat
    gmsh.model.occ.synchronize()

    for dim, tag in gmsh.model.getEntities(0):
        gmsh.model.mesh.setSize([(0, tag)], lc)

    # gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", size_factor)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)  
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    gmsh.model.mesh.generate(2)


    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model(), MPI.COMM_WORLD, 0, gdim=2
    )
    

    mesh.name = "mesh"
    cell_tags.name = "cells"
    facet_tags.name = "facets"

    with XDMFFile(MPI.COMM_WORLD, f_name, "w") as file:
        file.write_mesh(mesh)
        file.write_meshtags(cell_tags, mesh.geometry)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        file.write_meshtags(facet_tags, mesh.geometry)

    gmsh.finalize()
    return mesh, cell_tags, facet_tags

if __name__ == "__main__":

    num_arrays = np.linspace(1,9,9, dtype=int)

    char_length_rat = 40
    for num_array in num_arrays:
        f_name = f'mesh/slit/slit{num_array}.xdmf'
        generate_slit_mesh(f_name = f_name,num_x = num_array, num_y=1, char_length_rat=char_length_rat)
