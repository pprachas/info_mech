import sys
import os

H = 100
R_all = [H/2.5, H/5, H/10, H/20]
num_split = 100

commands = []

for R in R_all:
    for num in range(num_split):
        commands.append(f'python3 run_fea.py {int(R)} {num}')

command = commands[int(sys.argv[1])]
print(command)
os.system(command)