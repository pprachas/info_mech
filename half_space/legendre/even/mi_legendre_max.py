import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

# Load and scale Legendre coefficients
coeffs = np.loadtxt(f'coeffs/legendre_coeffs{num_coeff}.txt')[:, :num_coeff]
print(coeffs.shape)
scaler = StandardScaler()
coeff_scaled = scaler.fit_transform(coeffs)

# ------------------------- Run Selection -------------------------#
sensor_indices, sensor_signals, cmi_history = greedy_sensor_selection(num_sensors, sigma_y, coeff_scaled, scaler)

# Save sensor indices
np.savetxt(f'sensor_loc/coeffs{num_coeff}.txt', np.array(sensor_indices))
# ------------------------- Plot CMI Maps -------------------------#
x_grid, y_grid = np.meshgrid(x / a, y / a)
fig, axes = plt.subplots(1, num_sensors, figsize=(3 * (num_sensors), 3), constrained_layout=True)
fig.suptitle(fr'$d_x = {num_coeff}$')
# if num_sensors == 1:
#     axes = [axes]  # Ensure iterable

for ii in range(num_sensors):
    ax = axes[ii]
    cmi = cmi_history[ii]
    levels = np.linspace(np.min(cmi_history), np.max(cmi_history), 15)

    contour = ax.contourf(x_grid, y_grid, cmi, levels=levels, cmap='Greys')
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_xlabel(r'$\displaystyle{\sfrac{x}{a}}$')
    ax.set_ylabel(r'$\displaystyle{\sfrac{y}{a}}$')

    # Plot all candidate locations
    ax.plot(x_grid, y_grid, marker='.', ls='none', c=(0.7,0.5,0.5), markersize=1.0, label = 'Candidate sensor')

    # Plot selected sensors up to this step
    if ii != 0:
        for jj in range(ii):
            y_idx, x_idx = sensor_indices[jj]
            ax.plot(x_grid[y_idx, x_idx], y_grid[y_idx, x_idx], marker='*', ls='none',
                    c='r', markersize=10, label = 'Selected sensor')

    ax.set_title(f'Sensor {ii + 1}')

# Shared colorbar
cbar = fig.colorbar(
    contour,
    ax=axes,                   # apply to all axes
    orientation='horizontal',
    fraction=1.0,         
    pad=0.10,                  # 10% of subplot height away from the axes
    aspect = 100,
    label='MI Gain'
)
# plt.tight_layout()
plt.savefig(f'plots/all_cmi_maps_coeffs{num_coeff}.pdf', bbox_inches ='tight')
plt.close()
# 1) Create proxy artists matching your two marker styles
candidate_proxy = Line2D(
    [0], [0],
    marker='.', color=(0.7,0.5,0.5), linestyle='None',
    markersize=5,
    label='Candidate sensors'
)
selected_proxy = Line2D(
    [0], [0],
    marker='*', color='r', linestyle='None',
    markersize=10,
    label='Selected sensors'
)

# 2) Make a tiny figure with no axes
fig_leg = plt.figure(figsize=(4, 1))   # adjust width/height to taste

# 3) Place a legend in the center of this figure
fig_leg.legend(handles=[candidate_proxy, selected_proxy],loc='center',ncol=2)

# 4) Remove margins so the legend fills the PDF
fig_leg.subplots_adjust(left=0, right=1, top=1, bottom=0)

# 5) Save it
fig_leg.savefig('plots/legend_sensors.pdf', bbox_inches='tight')
plt.close(fig_leg)

