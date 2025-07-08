import numpy as np
from npeet import entropy_estimators as ee

def normalize_signal(signal, scaler):
    """Standardize and zero-out numerically small values."""
    scaler.fit(signal)
    scaled = scaler.transform(signal)
    scaled[np.isclose(signal - scaler.mean_, 0, atol=1e-12)] = 0
    return scaled

def get_grid_mi(sensor_list, sigma_y, coeff_scaled, scaler):
    """Compute mutual information between coeffs and stacked signals."""
    mi_grid = np.zeros((sigma_y.shape[1], sigma_y.shape[0]))
    for ii in range(sigma_y.shape[0]):
        for jj in range(sigma_y.shape[1]):
            current_signal = normalize_signal(sigma_y[ii, jj, :, None], scaler)
            stacked = np.hstack(sensor_list + [current_signal])
            mi_sample = ee.mi(coeff_scaled, stacked, base=np.e, k=5)
            mi_grid[jj, ii] = mi_sample
    return mi_grid

def greedy_sensor_selection(num_sensors, sigma_y, coeff_scaled, scaler):
    """Greedy selection of sensor locations maximizing conditional MI."""
    sensor_list = []
    selected_indices = []
    cmi_history = []

    for s in range(num_sensors+1):
        current_mi_grid = get_grid_mi(sensor_list, sigma_y, coeff_scaled, scaler)

        if selected_indices:
            prev_idx = selected_indices[-1]
            base_mi = get_grid_mi(sensor_list[:-1], sigma_y, coeff_scaled, scaler)[prev_idx[0], prev_idx[1]]
            cmi_grid = current_mi_grid - base_mi
        else:
            cmi_grid = current_mi_grid

        best_idx = np.unravel_index(np.argmax(cmi_grid), cmi_grid.shape)
        new_signal = normalize_signal(sigma_y[best_idx[1], best_idx[0], :, None], scaler)

        sensor_list.append(new_signal)
        selected_indices.append(best_idx)
        cmi_history.append(cmi_grid.copy())

    return selected_indices, sensor_list, cmi_history