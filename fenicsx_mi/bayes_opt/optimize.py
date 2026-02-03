import numpy as np
from mesh_ellipse import generate_slit_ellipse
from fea import import_mesh, run_linear_fea_traction_batch, compute_stress, eval_points
from dolfinx import fem
import ufl
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization, acquisition
from bayes_opt.parameter import wrap_kernel
from postprocess import normalize_signal

import sys
from custom_ee import entropy_r
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from pathlib import Path
from scipy.optimize import NonlinearConstraint

seed = 42
rng = np.random.default_rng(seed)

Path(f'optimize/').mkdir(parents=True, exist_ok=True)
flag = int(sys.argv[1])

def eval_function(**kwargs):
    L = 100
    H = 100
    num_array = 9
    num_coeff = 6

    a = []
    b = []

    for ii in range(num_array):
        a.append(kwargs[f'a{ii}'])
        b.append(kwargs[f'b{ii}'])

    # 1) generate mesh
    generate_slit_ellipse(
        f_name=f'optimize/opt{flag}.xdmf',
        num_x=num_array, # 9 slits
        a=a,
        b=b
    )
    domain = import_mesh(f'optimize/opt{flag}.xdmf')

    # 2) sample some FEA runs
    coeffs = rng.uniform(low=-10, high=10, size=(500, num_coeff)) # 500 samples
    sigma_points = []
    
    a,l,V,u_all,bcs,u0 = run_linear_fea_traction_batch(domain, L, H, coeffs)
    W = fem.functionspace(domain, ('DG', 1, (2,2)))
    E = fem.Constant(domain, 1.0)
    nu = fem.Constant(domain, 0.0)
    for u in u_all:
        sigma_u = compute_stress(W, u, E, nu)
        W0 = fem.functionspace(domain, ('DG',1))
        sigma_xx, sigma_xy, sigma_yx, sigma_yy = sigma_u.split()
        points_x = np.linspace(-L/2, L/2, num_coeff)
        pts = np.zeros((num_coeff,3))
        pts[:,0] = points_x
        sigma_points.append(eval_points(domain, sigma_yy, pts))
    sigma_points = np.array(sigma_points).squeeze()

    # 3) compute mutual information
    scaler = StandardScaler()
    coeff_scaled = scaler.fit_transform(coeffs)

    sigma_scaled = normalize_signal(sigma_points,scaler)

    mi,Hx,_,_ = entropy_r(coeff_scaled, sigma_scaled, base=np.e, k=5, vol=False)
    if flag == 1:
        return mi/Hx
    else:
        return -np.abs(mi/Hx)

# Constraint Function
def constraint_function(**kwargs):
    # this is not needed but here to avoid error
    a = []
    b = []
    num_array = 9

    for ii in range(num_array):
        a.append(kwargs[f'a{ii}'])
        b.append(kwargs[f'b{ii}'])

    domain = import_mesh(f'optimize/opt{flag}.xdmf')
    volume = fem.assemble_scalar(fem.form(fem.Constant(domain,1.0)*ufl.dx(domain=domain))) # volume of domain
    V0 = 100*100
    return volume/V0

vol_frac = 0.2
constraint = NonlinearConstraint(constraint_function, vol_frac, 1) 

# 4) build pbounds: gap fractions + 5×2 slopes flattened
pbounds = {}

L = 100
num = 9

print((L/(2*num)-L/30))
pbounds[f'a0'] = (L/100, (L/(2*num)-L/30)) # minumum edge width is L/30
pbounds[f'a{num-1}'] = (L/100,L/(2*num)-L/30) # minumum edge width is L/30 

pbounds[f'b0'] = (L/100,L/2-L/30) # minumum edge width is L/30
pbounds[f'b{num-1}'] = (L/100,L/2-L/30) # minumum edge width is L/30 

for ii in range(1,num-1):
    pbounds[f'a{ii}'] = (L/100, L/(num))
    pbounds[f'b{ii}'] = (L/100, L/2-L/30)


acq = acquisition.ExpectedImprovement(xi=0.01)

optimizer = BayesianOptimization(
    f=eval_function,
    constraint = constraint,
    pbounds=pbounds,
    verbose=2,
    random_state=42,
)

# define Matern2.5 kernel with noise

optimizer.set_gp_params(alpha=1/500, n_restarts_optimizer=5)

# 5) run it
optimizer.maximize(init_points=10, n_iter=200)

# 6) look at results
for ii, res in enumerate(optimizer.res):
    print(f"Iteration {ii}: {res}")
print("Best found:", optimizer.max)

# remesh best solution
best = optimizer.max['params']

a = []
b = []
for ii in range(num):
    a.append(best[f'a{ii}'])
    b.append(best[f'b{ii}'])

generate_slit_ellipse(
        f_name=f'optimize/opt{int(sys.argv[1])}.xdmf',
        num_x=num,
        a=a,
        b=b
    )
# save optimizer state
optimizer.save_state(f'optimize/optimizer{sys.argv[1]}_state.json')