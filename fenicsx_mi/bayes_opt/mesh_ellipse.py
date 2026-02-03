import gmsh
from dolfinx.io import gmshio, XDMFFile
from mpi4py import MPI
import numpy as np

def generate_slit_ellipse(
    L=100, H=100,
    num_x=3,
    a=10, b=10,                   # can now be scalar or length-num_x
    f_name="ellipse_mesh.xdmf",
    char_length=None
):
    """
    Generate 2D mesh [-L/2,L/2] x [0,H] with a single row of num_x elliptical slits.
    a, b may be scalars or arrays of length num_x.
    """
    gmsh.initialize()
    gmsh.model.add("slit_domain")

    # --- broadcast a, b to arrays of length num_x ---
    a_vec = np.array(a, float)
    if   a_vec.ndim == 0:
        a_vec = np.full(num_x, a_vec)
    elif a_vec.ndim == 1 and a_vec.size == num_x:
        pass
    else:
        raise ValueError("a must be scalar or length num_x")

    b_vec = np.array(b, float)
    if   b_vec.ndim == 0:
        b_vec = np.full(num_x, b_vec)
    elif b_vec.ndim == 1 and b_vec.size == num_x:
        pass
    else:
        raise ValueError("b must be scalar or length num_x")
    # --------------------------------------------------

    # Base rectangle
    domain = gmsh.model.occ.addRectangle(-L/2, 0, 0, L, H)

    # Evenly spaced slit centers in x, single row at y = H/2
    x_centers = np.linspace(-L/2 + L/(2*num_x),
                              L/2  - L/(2*num_x),
                              num_x)
    y_center = H/2

    slits = []
    for ii, cx in enumerate(x_centers):
        cy = y_center
        ai = a_vec[ii]
        bi = b_vec[ii]

        if ai >= bi: # ellipse with major axis aligned with x-axis
            e = gmsh.model.occ.addEllipse(cx, cy,0, ai,bi)
        else: # ellsipse with major axis aligned with y-axis
            e = gmsh.model.occ.addEllipse(cx, cy,0, bi,ai,zAxis = [0,0,1],xAxis = [0,1,0])# swap major axis 

        loop = gmsh.model.occ.addCurveLoop([e])
        slit = gmsh.model.occ.addPlaneSurface([loop])
        slits.append((2, slit))

    # subtract slits
    cut, _ = gmsh.model.occ.cut([(2, domain)], slits)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [c[1] for c in cut])

    # mesh options
    if char_length is None:
        left_gap   = x_centers[0]- a_vec[0]+L/2
        right_gap  = (L/2) - (x_centers[-1] + a_vec[-1])
        bottom_gap = y_center - b_vec.min()
        top_gap = H - (y_center    + b_vec.min())
        min_gap = min(left_gap, right_gap, bottom_gap, top_gap)
        char_length = min_gap / 7

        print(left_gap,right_gap)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)

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

L = 100
num = 9

a = [(L/(2*num)-L/30), L/(2.5*num),L/(2.5*num), L/(2.5*num), L/(2.5*num), L/(2.5*num), L/(2.5*num), L/(2.5*num), (L/(2*num)-L/30)]

print(a)
if __name__ == "__main__":
    generate_slit_ellipse(num_x=9, a=a, b=L/2-L/10)

# generate_slit_ellipse(num_x=4, a=5,            b=3)        # all slits 5×3
# generate_slit_ellipse(num_x=4, a=[2,4,6,8],   b=3)        # varying a
# generate_slit_ellipse(num_x=4, a=[2,4,6,8],   b=[1,2,3,4]) # fully varying