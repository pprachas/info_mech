import numpy as np
import sys
from pathlib import Path
sys.path.append('../..')
from utils.meshing import plate_hole_mesh


L = 100
H = 100
R_all = [H/2.5, H/5, H/10, H/20]
Path('mesh').mkdir(parents=True, exist_ok=True)


# Create mesh
n_nodes = 65
for R in R_all:
    f_mesh = f'mesh/plate_hole{int(R)}.xdmf'
    plate_hole_mesh(L,H,R,n_nodes,f_mesh)