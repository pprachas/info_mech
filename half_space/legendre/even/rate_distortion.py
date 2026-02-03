import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy import *
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import seaborn as sns
from npeet import entropy_estimators as ee
from custom_ee import entropy_r
from postprocess import normalize_signal
# to make figure handles
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


plt.style.use('../../jeff_style.mplstyle')

num_coeff = 4
num_samples = 5000 # number of load samples

#-------------------------Load Data--------------------------------------------------------#
mse_greedy = np.loadtxt(f'RD/greedy/D{num_coeff}.txt')
mse_random = np.loadtxt(f'RD/random/D{num_coeff}.txt')

R_greedy = np.loadtxt(f'RD/greedy/R{num_coeff}.txt')
R_random = np.loadtxt(f'RD/random/R{num_coeff}.txt')

#setup dataframe for violin plot
df = pd.DataFrame(mse_random.T)
df.columns += 1 # make index star from 0

fig,ax = plt.subplots(figsize=(6.5/2,6.5/2))
#plt.plot(np.arange(0,len(score))+1,score_dummy, c='k', alpha = 0.5)
# ax = sns.violinplot(df,cut=0, inner = 'quartile', native_scale=True, color = '0.8', density_norm='width')

# only show median line
# for ii,line in enumerate(ax.lines):
#   line.set_linestyle('-')
#   if (ii-1)%3 != 0: 
#     line.set_linestyle('')

c = ['0.2', '0.6', (0.9,0.5,0.5), (0.7,0,0), (0.4,0,0)]

p = {}
for ii,color in enumerate(c):
       p[ii+1] = color
print(p)
# Violin plot
sns.stripplot(df, ax = ax, jitter = False, 
              native_scale=True, s=4,facecolor='none', 
              linewidth = 1.5, palette = p)

# Chnage stipplot color
for coll, col in zip(ax.collections[-len(df.columns):], df.columns):
    coll.set_facecolor('none')
    coll.set_edgecolor(p[col])

plt.scatter(np.arange(0,len(mse_greedy))+1,mse_greedy, marker = 's',s=64, edgecolors = c, linewidths = 1.5, zorder = 10, facecolors='none')
plt.xlabel('Number of Sensors', fontsize = 12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.tight_layout()
plt.savefig('mse.pdf')

# Legend
custom_handles = [
    Line2D([0], [0],
           marker='s',
           linestyle='none',
           mec=(0.5,0,0),
           markerfacecolor='none',
           markersize=8,
           mew=1.5,
           label='Greedy Selection'),
        Line2D([0], [0],
           marker='.',
           linestyle='none',
           color = (0.9,0.5,0.5),
           label='Random Selection'),
]

labels = [h.get_label() for h in custom_handles]

# Make a new tiny figure for the legend:
fig_leg = plt.figure()
fig_leg.legend(custom_handles,
               labels,
               fontsize=12, 
               ncol = 2)

fig_leg.tight_layout()
plt.savefig('mse_legend.pdf')

#----------------Rate-distortion-----------------#
fig, ax = plt.subplots(figsize=(6.5/2,6.5/2))

c = ['0.2', '0.6', (0.9,0.5,0.5), (0.7,0,0), (0.4,0,0)]

for ii in range(5):
  ax.plot(mse_greedy[ii], R_greedy[ii], marker = 's',markersize= 8, ls = 'none', mec = c[ii], mew = 1.0, zorder = 10, fillstyle='none', label = f'{ii+1} sensors')
  ax.plot(mse_random[ii], R_random[ii], marker = '.', ls = 'none', mec = c[ii],mew = 1.0, zorder = 10,  fillstyle='none',label = f'{ii+1} sensors')

D = np.linspace(np.min(mse_greedy), 0.3, 200)

d_D = 1
# shannon lower bound
R = 0.5* np.log(6/(np.pi*np.e*D))


R[R < 0] = 0
R = np.append(R,0)

D = np.append(D,1)

# Shade regions
ax.fill_between(np.array([-1,0.0001]),np.array([-1,10]),-1, color = 'lightgray')

ax.fill_between(D,R,-5, color = 'lightgray')
ax.plot(D,R, ls = ':', c = 'k', label = 'Shannon Lower Bound')

ax.set_xlabel('Distortion (MSE)', fontsize = 12)
ax.set_ylabel(r'Rate ($I(X;\hat{X})$)', fontsize = 12)
ax.set_ylim([-0.75, 7])
ax.set_xlim([-0.01, 1.0])

# Inset
x1, x2, y1, y2 = -0.005, 0.025, 4.95, 5.23 # subregion of the original image
axins = ax.inset_axes(
    [0.57, 0.57, 0.40, 0.40],
    xlim=(x1, x2), ylim=(y1, y2))
for ii in range(5):
    axins.plot(mse_greedy[ii], R_greedy[ii], marker = 's',markersize= 8, ls = 'none', mec = c[ii], mew = 1.0, zorder = 10, fillstyle='none', label = f'{ii+1} sensors')
    axins.plot(mse_random[ii], R_random[ii], marker = '.', ls = 'none', mec = c[ii],mew = 1.0, zorder = 10,  fillstyle='none',label = f'{ii+1} sensors')
axins.fill_between(np.array([-1,0.001]),np.array([-1,10]),-1, color = 'lightgray')
axins.fill_between(D,R,-1, color = 'lightgray')
axins.plot(D,R, ls = ':', c = 'k', label = 'Shannon Lower Bound')

ax.indicate_inset_zoom(axins, edgecolor="black")


plt.tight_layout()
plt.savefig('RD.pdf')

custom_handles = [
    Line2D([0], [0],
           marker='s',
           linestyle='none',
           markeredgecolor='k',
           markerfacecolor='none',
           markersize=8,
           mew=1.5,
           label='Greedy Selection'),
    Line2D([0], [0],
           marker='.',
           linestyle='none',
           markeredgecolor='k',
           markerfacecolor='none',
           markersize=8,
           mew=1.0,
           label='Random Selection'),
]

labels = [h.get_label() for h in custom_handles]

# Make a new tiny figure for the legend:
fig_leg = plt.figure()
fig_leg.legend(custom_handles,
               labels,
               fontsize=12, 
               ncol = 3)

fig_leg.tight_layout()
plt.savefig('RD_legend_marker.pdf')


custom_handles = [
    Patch(facecolor = '0.2', label = '1'), Patch(facecolor = '0.6', label = '2'), Patch(facecolor = (0.9,0.5,0.5), label = '3'),
    Patch(facecolor = (0.7,0.0,0.0), label = '4'), Patch(facecolor = (0.4,0.0,0.0), label = '5')
]

labels = [h.get_label() for h in custom_handles]


fig_leg = plt.figure(figsize=(8.5,6.5))
fig_leg.legend(custom_handles,
               labels,
               title = 'Number of Sensors',
               fontsize=12, 
               title_fontsize = 12,
               ncol = 5)


fig_leg.tight_layout()
plt.savefig('RD_legend_colors.pdf')

plt.show()