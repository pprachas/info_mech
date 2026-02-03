import sys
import os
import numpy as np

L = 100

num_arrays = np.linspace(1,9,9, dtype=int)
phi_all = [0.3] #[0.1,0.2,0.3]
c1_all = np.linspace(-0.2,0.2,5)
c2_all = np.linspace(-0.2,0.2,5)
runs = 5
num_coeffs = [6]
num_geom = 2

commands = []

for num_array in num_arrays:
    for run in range(runs):
        for geom in range(num_geom):
            commands.append(f'python3 run_fea.py {int(num_array)} {run} {geom}')

command = commands[int(sys.argv[1])-1]

print(len(commands))
print(command)

os.system(command)