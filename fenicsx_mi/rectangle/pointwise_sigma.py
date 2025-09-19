import numpy as np
from dolfinx import fem
import sys 
sys.path.append('../..')

from utils.fea import import_mesh, eval_points
from pathlib import Path


L = 100
H_all = [4*L,2*L,L,L/2,L/4]

Path('pointwise_sigma').mkdir(parents=True, exist_ok=True)

for H in H_all:
    sigma_yy = []
    for ii in range(100):
        sigma_yy.append(np.loadtxt(f'sigma_yy/H{int(H)}/sigma_yy{ii}.txt'))
    
    sigma_yy = np.vstack(np.array(sigma_yy))

    # Load Mesh
    f_mesh = f'mesh/rectangle{int(H)}.xdmf'
    domain = import_mesh(f_mesh)

    # project to to function space
    W = fem.functionspace(domain,('DG',1))
    sigma_yy_int = fem.Function(W)

    sigma_points = []

    for sigma in sigma_yy:
        sigma_yy_int.x.array[:] = sigma
        # get stress at points

        points = np.array([[-L/2,0,0],[-L/4,0,0],[L/4,0,0],[L/2,0,0]])

        sigma_points.append(eval_points(domain,sigma_yy_int,points))
    
    sigma_points = np.array(sigma_points).squeeze()

    print(sigma_points.shape)
    np.savetxt(f'pointwise_sigma/H{int(H)}.txt', sigma_points)
