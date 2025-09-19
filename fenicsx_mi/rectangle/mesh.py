import numpy as np
import sys
from pathlib import Path
sys.path.append('../..')
from utils.meshing import rectangle_mesh


L = 100
H_all = [4*L,2*L,L,L/2,L/4]
Path('mesh').mkdir(parents=True, exist_ok=True)


# Create mesh
n_nodes = 100
for H in H_all:
    f_mesh = f'mesh/rectangle{int(H)}.xdmf'
    rectangle_mesh(L,H,n_nodes,f_mesh)