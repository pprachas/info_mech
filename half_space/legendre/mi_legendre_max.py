import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from npeet import entropy_estimators as ee
from pathlib import Path

import sys 
sys.path.append('../..')

from utils.postprocess import normalize_signal, get_grid_mi, greedy_sensor_selection

#-------------------Create files---------------------------------#
Path('./plots').mkdir(parents=True, exist_ok=True)
Path('./sensor_loc').mkdir(parents=True, exist_ok=True)

plt.style.use('../jeff_style.mplstyle')

# ------------------------- Configuration -------------------------#
num_coeff = 3
num_samples = 5000
num_sensors = 4  # Number of sensors to select
a = 100  # Normalization factor for coordinates

# ------------------------- Load Data -------------------------#
# Load stress data from CSV and reshape
df = pd.read_csv(f'sigma_y/legendre_coeffs{num_coeff}.csv', index_col=[0, 1], header=[0])
sigma_y = df.to_numpy().reshape(num_samples, -1, df.shape[1]).transpose(2, 1, 0)

# Extract spatial coordinates
x = df.columns.to_numpy().astype(float)
y = df.index.get_level_values(1)[:sigma_y.shape[1]].to_numpy()

# Load and scale Legendre coefficients
coeffs = np.loadtxt(f'coeffs/legendre_coeffs{num_coeff}.txt')[:, :num_coeff]
scaler = StandardScaler()
coeff_scaled = scaler.fit_transform(coeffs)

# ------------------------- Run Selection -------------------------#
sensor_indices, sensor_signals, cmi_history = greedy_sensor_selection(num_sensors, sigma_y, coeff_scaled, scaler)

print(len(cmi_history), sensor_indices)

# Save sensor indices
np.savetxt(f'sensor_loc/coeffs{num_coeff}.txt', np.array(sensor_indices))
# ------------------------- Plot CMI Maps -------------------------#
x_grid, y_grid = np.meshgrid(x / a, y / a)
fig, axes = plt.subplots(1, num_sensors+1, figsize=(3 * (num_sensors+1), 3))
fig.suptitle(f'{num_coeff} Legendre Coefficients')
# if num_sensors == 1:
#     axes = [axes]  # Ensure iterable

for ii in range(num_sensors+1):
    ax = axes[ii]
    cmi = cmi_history[ii]
    levels = np.linspace(np.min(cmi), np.max(cmi), 15)

    contour = ax.contourf(x_grid, y_grid, cmi, levels=levels, cmap='YlOrRd')
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_xlabel(r'$\displaystyle{\sfrac{x}{a}}$')
    ax.set_ylabel(r'$\displaystyle{\sfrac{y}{a}}$')

    # Plot all candidate locations
    ax.plot(x_grid, y_grid, marker='.', ls='none', c='k', markersize=1.0)
    # Add individual colorbar for this axis
    cbar = fig.colorbar(contour, ax=ax, label='MI Gain',orientation='horizontal',pad=0.2, aspect=25)
    tick_locs = np.linspace(np.min(cmi), np.max(cmi), 5)  # Only 3 ticks
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([f"{tick:.2f}" for tick in tick_locs])

    # Plot selected sensors up to this step
    if ii != 0:
        for jj in range(ii):
            y_idx, x_idx = sensor_indices[jj]
            ax.plot(x_grid[y_idx, x_idx], y_grid[y_idx, x_idx], marker='X', ls='none',
                    c='blue', markersize=10)

    ax.set_title(f'Sensor {ii + 1}')

# Shared colorbar
plt.tight_layout()
plt.savefig(f'plots/all_cmi_maps_coeffs{num_coeff}.pdf')
plt.show()
