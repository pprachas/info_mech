from mesh_slit import generate_slit_mesh
from mesh_pore import generate_pore
import numpy as np

# generate pore mesh

num_array = 9 # the "hardest case"
N = np.arange(-2,4)

char_lengths = 20*np.sqrt(2)**N

for count,char_length in enumerate(char_lengths):
    f_name = f'mesh_refinement/pore/pore{num_array}_{count}.xdmf'
    generate_pore(f_name = f_name,char_length_rat = char_length)

# Slit mesh
num_array = 9 
char_lengths = 20*np.sqrt(2)**N
for count,char_length in enumerate(char_lengths):

    f_name = f'mesh_refinement/slit/slit{num_array}_{count}.xdmf'
    generate_slit_mesh(f_name = f_name,num_x = num_array,char_length_rat = char_length)

print(len(char_lengths))