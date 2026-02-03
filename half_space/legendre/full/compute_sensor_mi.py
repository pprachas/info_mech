import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from npeet import entropy_estimators as ee
from pathlib import Path
import sys
from postprocess import normalize_signal, get_grid_mi, greedy_sensor_selection


plt.style.use('../../jeff_style.mplstyle')

# ------------------------- Configuration -------------------------#
num_coeff = int(sys.argv[1])
num_samples = 5000
num_sensors = 4  # Number of sensors to select
a = 100  # Normalization factor for coordinates
#-------------------Create files---------------------------------#
Path(f'./plots/coeffs{num_coeff}').mkdir(parents=True, exist_ok=True)
Path('./sensor_loc').mkdir(parents=True, exist_ok=True)

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
sensor_indices, _, cmi_history = greedy_sensor_selection(num_sensors, sigma_y, coeff_scaled, scaler)

# Save results
np.savetxt(f'sensor_loc/coeffs{num_coeff}.txt', np.array(sensor_indices))
np.savez(f'plots/coeffs{num_coeff}/cmi_history.npz', np.array(cmi_history))



