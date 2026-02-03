import sys
import os
import numpy as np

L = 100

nums = 6

runs = 5
geoms = 2
num_coeffs = [6]

commands = []

for num in range(nums):
    for run in range(runs):
        for geom in range(geoms):
            commands.append(f'python3 fea_mesh_refine.py {num} {run} {geom}')

command = commands[int(sys.argv[1])-1]

print(len(commands))
print(command)


os.system(command)