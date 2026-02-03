import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bayes_opt import BayesianOptimization
from scipy.optimize import NonlinearConstraint

plt.style.use('jeff_style.mplstyle')

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
pbounds[f'a0'] = (L/100, (L/(2*num)-L/30)) # minumum edge width is L/20
pbounds[f'a{num-1}'] = (L/100,L/(2*num)-L/30) # minumum edge width is L/20 

pbounds[f'b0'] = (L/100,L/2-L/30) # minumum edge width is L/30
pbounds[f'b{num-1}'] = (L/100,L/2-L/30) # minumum edge width is L/30 

for ii in range(1,num-1):
    pbounds[f'a{ii}'] = (L/100, L/(num))
    pbounds[f'b{ii}'] = (L/100, L/2-L/30)

optimizer_max = BayesianOptimization(f=eval_function, constraint = constraint,pbounds = pbounds, verbose=2)
optimizer_max.load_state('optimize/optimizer1_state.json')

optimizer_min = BayesianOptimization(f=eval_function, constraint = constraint,pbounds = pbounds, verbose=2)
optimizer_min.load_state('optimize/optimizer2_state.json')

# For appendix
fig, ax = plt.subplots(figsize=(6.5,5))

ax.plot(optimizer_max.space.target, label=r'Maximize: $I(X;Y)$', ls = ':', c = 'k')
ax.plot(np.maximum.accumulate(optimizer_max.space.target), c = 'k', lw = 2, label=r'Maximize: $ \max \,I(X;Y)$')
ax.plot(-optimizer_min.space.target, label=r'Minimize: $I(X;Y)$', ls = ':', c = '0.7')
ax.plot(np.minimum.accumulate(-optimizer_min.space.target), c = '0.7', lw = 2, label=r'Minimize: $\min \, I(X;Y)$')
ax.set_xlabel('Number of Iterations', fontsize=12)
ax.set_ylabel(r'$I(X;Y)$ (500 samples)', fontsize=12)
ax.set_title('Raw Data of Bayesian Optimization', fontsize=12)
plt.tight_layout()
plt.savefig('appendix_opt.pdf')

handles, labels = ax.get_legend_handles_labels()

plt.figure(figsize=(6.5,5))

plt.legend( handles, labels, ncol = 2, fontsize=12)
plt.savefig('appendix_opt_legend.pdf')

# For main paper
fig, ax = plt.subplots(figsize=(5.5,5))

ax.plot(np.maximum.accumulate(optimizer_max.space.target), c = 'k', lw = 2, label=r'Maximize: $I(X;Y)$')
ax.plot(np.minimum.accumulate(-optimizer_min.space.target), c = '0.7', lw = 2, label=r'Minimize: $I(X;Y)$')
ax.set_xlabel('Number of Iterations', fontsize=12)
ax.set_ylabel(r'$I(X;Y)$ (500 samples)', fontsize=12)
ax.set_title('Convergence of Bayesian Optimization', fontsize=12)


# Find indices where max updates:
running_max = np.maximum.accumulate(optimizer_max.space.target)
is_new_max = optimizer_max.space.target == running_max

running_idx = np.maximum.accumulate(
    np.where(is_new_max, np.arange(len(optimizer_max.space.target)), 0)
)

print(f'max index: {np.unique(running_idx)}')

# Find indices where min updates:
running_min = np.minimum.accumulate(-optimizer_min.space.target)
is_new_min = -optimizer_min.space.target == running_min


running_idx = np.maximum.accumulate(
    np.where(is_new_min, np.arange(len(optimizer_min.space.target)), 0)
)

print(f'min index: {np.unique(running_idx)}')

max_idx = [17,104,135]
min_idx = [8,22,60]

plt.plot(max_idx,running_max[max_idx], ls = 'none',marker = 's', fillstyle='none', mew = 1.0, mec = 'k', ms=5)
plt.plot(min_idx,running_min[min_idx], ls = 'none',marker = 's', fillstyle='none', mew = 1.0, mec = '0.7', ms=5)

plt.tight_layout()
plt.savefig('bayes_opt.pdf')
plt.show()