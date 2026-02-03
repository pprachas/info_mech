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
num_coeff = 5
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

plt.savefig(f'plots/all_cmi_maps_coeffs{num_coeff}_even.pdf', bbox_inches='tight')
plt.close()

# 1) Create proxy artists matching your two marker styles
candidate_proxy = Line2D(
    [0], [0],
    marker='.', color=(0.75,0.5,0.5), linestyle='None',
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

