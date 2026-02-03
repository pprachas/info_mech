import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import Normalize
from custom_ee import entropy_r
import matplotlib.ticker as ticker
from postprocess import normalize_signal

plt.style.use('jeff_style.mplstyle')


# pore parameters for circle
num_arrays = np.linspace(1,9,9, dtype=int)
scaler = StandardScaler()

load_case = 'full'

num_coeff = 6
runs = 5

mi_all = []
Hx_all = []

coeffs = np.loadtxt(f'../../half_space/legendre/full/coeffs/legendre_coeffs{num_coeff}.txt')[:,:num_coeff] # legendre coefficients
coeff_scaled = scaler.fit_transform(coeffs)
for num_array in num_arrays:
    sigma_yy = []
    for run in range(runs):
        sigma_yy.append(np.load(f'pointwise_sigma/pore{num_array}/pore{run}.npz')['sigma_points'])

    sigma_yy = np.vstack(sigma_yy)

    scaler.fit(sigma_yy)
    sigma_yy_scaled = normalize_signal(sigma_yy, scaler)

    mi_sample,Hx_sample,_,_ = entropy_r(coeff_scaled,sigma_yy_scaled,base=np.e,k=5, vol=False)

    mi_all.append(mi_sample)
    Hx_all.append(Hx_sample)

mi_all = np.array(mi_all)
Hx_all = np.array(Hx_all)
norm_mi = mi_all/Hx_all

# get solid block
runs = 5
f_path = f'../rectangle/pointwise_sigma/H100'
sigma_yy_rec = []
for run in range(runs):
    sigma_yy_rec.append(np.load(f'{f_path}/rectangle{run}.npz')['sigma_points'])

sigma_yy_rec = np.vstack(sigma_yy_rec)

scaler.fit(sigma_yy_rec)
sigma_yy_rec_scaled = normalize_signal(sigma_yy_rec, scaler)
mi_sample_rec,Hx_sample_rec,_,_ = entropy_r(coeff_scaled,sigma_yy_rec_scaled,base=np.e,k=5, vol=False)

norm_mi_rec = mi_sample_rec/Hx_sample_rec


# get slits
runs = 5

mi_slit_all = []
Hx_slit_all = []

for num_array in num_arrays:
    sigma_yy = []
    for run in range(runs):
        sigma_yy.append(np.load(f'pointwise_sigma/slit{num_array}/slit{run}.npz')['sigma_points'])

    sigma_yy = np.vstack(sigma_yy)

    scaler.fit(sigma_yy)
    sigma_yy_scaled = normalize_signal(sigma_yy, scaler)
    mi_sample,Hx_sample,_,_ = entropy_r(coeff_scaled,sigma_yy_scaled,base=np.e,k=5, vol=False)

    mi_slit_all.append(mi_sample)
    Hx_slit_all.append(Hx_sample)

mi_slit_all = np.array(mi_slit_all)
Hx_slit_all = np.array(Hx_slit_all)
norm_mi_slit = mi_slit_all/Hx_slit_all

print(f'Norm mi rectangle: {norm_mi_rec}')
print(f'max slit: {np.max(norm_mi_slit)}, {np.argmax(norm_mi_slit)}')
print(f'min pore: {np.min(norm_mi)}, {np.argmin(norm_mi)}')
fig, ax = plt.subplots(figsize=(5.5,5))
plt.plot(num_arrays, norm_mi, marker='o', label = 'circle pores')
plt.axhline(y=norm_mi_rec, ls = ':', c = 'k',label = 'solid block')
plt.plot(num_arrays, norm_mi_slit, marker = 'd', label = 'slits')

plt.xlabel('Number of Units', fontsize=12)
plt.ylabel(r'$I(X;Y)/H(X)$', fontsize=12)


plt.tight_layout()
handles, labels = ax.get_legend_handles_labels()
plt.savefig('unit_mi.pdf')

fig_legend, ax_legend = plt.subplots(figsize=(6.5,5))

ax_legend.legend(handles, labels, loc='center', frameon=True, ncol = 3)

plt.tight_layout()
plt.savefig('unit_mi_legend.pdf')

plt.show()