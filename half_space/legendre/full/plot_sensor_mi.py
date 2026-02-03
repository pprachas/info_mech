import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from npeet import entropy_estimators as ee
from pathlib import Path

from postprocess import normalize_signal, get_grid_mi, greedy_sensor_selection

#-------------------Create files---------------------------------#
Path('./plots').mkdir(parents=True, exist_ok=True)
Path('./sensor_loc').mkdir(parents=True, exist_ok=True)

plt.style.use('../../jeff_style.mplstyle')

# ------------------------- Configuration -------------------------#
num_coeff = 3
num_samples = 5000
num_sensors = 4  # Number of sensors to select
a = 100  # Normalization factor for coordinates

# ------------------------- Load Data -------------------------#
# Load stress data from CSV and reshape
df = pd.read_csv(f'sigma_y/legendre_coeffs{num_coeff}.csv', index_col=[0, 1], header=[0])
sigma_y = df.to_numpy().reshape(num_samples, -1, df.shape[1]).transpose(2, 1, 0)

print(sigma_y.shape)

# Extract spatial coordinates
x = df.columns.to_numpy().astype(float)
y = df.index.get_level_values(1)[:sigma_y.shape[1]].to_numpy()
sensor_indices = np.loadtxt(f'sensor_loc/coeffs{num_coeff}.txt', dtype=int)
cmi_history = np.load(f'plots/coeffs{num_coeff}/cmi_history.npz')['arr_0']

print(sensor_indices)
# ------------------------- Plot CMI Maps -------------------------#
x_grid, y_grid = np.meshgrid(x / a, y / a)
fig, axes = plt.subplots(1, num_sensors, figsize=(5.3, 1.75), constrained_layout=True, sharex=True, sharey=True)
# fig.suptitle(fr'$d_x = {num_coeff}$')
fig.supylabel(r'$y/a$', fontsize=8)
fig.supxlabel(r'$x/a$', fontsize=8)
# if num_sensors == 1:
#     axes = [axes]  # Ensure iterable

for ii in range(num_sensors):
    ax = axes[ii]
    cmi = cmi_history[ii]
    levels = np.linspace(0, np.max(cmi_history), 15)

    contour = ax.contourf(x_grid, y_grid, cmi, levels=levels, cmap='Greys')
    ax.set_yscale('log')

    # Plot all candidate locations
    ax.plot(x_grid, y_grid, marker='.', ls='none', c=(0.75,0.5,0.5), markersize=1.75, mew = 0.0, label = 'Candidate sensor')

    # Plot selected sensors up to this step
    if ii != 0:
        for jj in range(ii):
            y_idx, x_idx = sensor_indices[jj]

            print(y_idx,x_idx)
            ax.plot(x_grid[y_idx, x_idx], y_grid[y_idx, x_idx], marker='*', ls='none',
                    c='r', markersize=5, label = 'Selected sensor')

    ax.set_title(f'Sensor {ii + 1}')
ax.invert_yaxis() # only ionvert one since that are all shared


# Shared colorbar
cbar = fig.colorbar(
    contour,
    ax=axes,                   # apply to all axes
    orientation='vertical',
    fraction=1.0,
    location = 'right',         
    # pad=0.05,                  # 10% of subplot height away from the axes
    aspect = 50,
    label='MI Gain'
)
# plt.tight_layout()
formatter = ticker.FormatStrFormatter('%.1f')
cbar.formatter = formatter

plt.savefig(f'plots/all_cmi_maps_coeffs{num_coeff}_full.pdf', bbox_inches='tight')
plt.close()

