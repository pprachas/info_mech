import numpy as np
import matplotlib.pyplot as plt
from utils.analytical import sigma_y_uniform

P = 1.0
x = 0
y = 1e9

np.random.seed(0)
a = np.random.uniform(0,100,1000)
sigma_y = sigma_y_uniform(P,a,x,y)

# p(a)
plt.figure()
hist, bin_edges = np.histogram(a, bins='auto',density=True)

plt.bar(bin_edges[:-1],hist, width=np.diff(bin_edges), edgecolor='k', align='edge')
plt.title('distribution of load widths')

# p(Y)
plt.figure()
hist, bin_edges = np.histogram(sigma_y,bins='auto',density=True)

plt.bar(bin_edges[:-1],hist, width=np.diff(bin_edges), edgecolor='k', align='edge')
plt.title('distribution of reaction forces')
#p(X,Y)
hist, x_edges, y_edges, = np.histogram2d(a,sigma_y, bins = 50, density=True)
hist = hist.T # for visualization purposes

X, Y = np.meshgrid(x_edges, y_edges)
fig = plt.figure()
plt.pcolormesh(X,Y,hist, cmap = 'Greys')
plt.title('joint distribution')
plt.xlabel('Width of load')
plt.ylabel('Reaction Force')
plt.tight_layout()

plt.figure()
plt.scatter(a, sigma_y, marker = '.')
plt.title('scatter plot')
plt.xlabel('Width of load')
plt.ylabel('Reaction force')
plt.tight_layout()

plt.show()




