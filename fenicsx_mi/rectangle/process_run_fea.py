import sys
import os

L = 100
H_all = [L]#[4*L,2*L,L,L/2,L/4]
num_split = 5

commands = []

for H in H_all:
    for num in range(num_split):
        commands.append(f'python3 run_fea.py {int(H)} {num}')

command = commands[int(sys.argv[1])-1]
print(len(commands))
os.system(command)