import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure(1)
fig.clf()
ax = Axes3D(fig)
datasets = random((8,100,3))*512
my_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

colors = ['k', "#B3C95A", 'b', '#63B8FF', 'g', "#FF3300",
          'r', 'k']
index = 0
for data, curr_color in zip(datasets, colors):
    ax.plot(np.log2(data[:, 0]), np.log2(data[:, 1]), 
                   np.log2(data[:, 2]), 'o', c=curr_color, label=my_labels[index])
    index += 1

ax.set_zlim3d([-1, 9])
ax.set_ylim3d([-1, 9])
ax.set_xlim3d([-1, 9])

ax.set_xticks(range(0,11))
ax.set_yticks([1,2,8])
ax.set_zticks(np.arange(0,9,.5))

ax.legend(loc = 'upper left')

plt.show()
