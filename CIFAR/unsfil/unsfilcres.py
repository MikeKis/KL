# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:24:47 2024

@author: Kiselev_Mi
"""

import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

nfilters = 22
filtersize = 3
nrows = 2

Wmin = -0.3
Wmax = 1

W = np.zeros((nfilters, filtersize, filtersize, 3))
with open("CIFAR10.filters.csv") as filres:
    filres.readline()
    for i in range(nfilters):
        for j in range(filtersize):
            s = filres.readline()
            lstr = s.split("),(")
            lstr[0] = lstr[0][1:]
            lstr[-1] = lstr[-1][:-2]
            for k in range(filtersize):
                ls = lstr[k].split(',')
                for l in range(3):
                    W[i][j][k][l] = float(ls[l])
        filres.readline()
            
norm = mpl.colors.TwoSlopeNorm(vmin = Wmin, vcenter = 0, vmax = Wmax)
fig, ax = plt.subplots(3, nfilters, sharex=True, sharey=True, layout="constrained", figsize=(42, 10))
for c in range(nfilters):
    a = np.transpose(W[c],(2, 0, 1))
    for r in range(3):
        ax[r, c].axis('off')
        ax[r, c].imshow(a[r], cmap = 'seismic', norm = norm)
plt.show()

fig, ax = plt.subplots(nrows, nfilters // nrows, sharex=True, sharey=True, layout="constrained", figsize=(42, 10))
i = 0
for r in range(nrows):
    for c in range(nfilters // nrows):
        a = W[i]
        dmax = 0
        with np.nditer(a, op_flags=['readwrite']) as it:
            for x in it:
                if x > 0:
                    dmax = max(dmax, x)
                else:
                    x[...] = 0        
            for x in it:
                x[...] = x / dmax        
        ax[r, c].axis('off')
        ax[r, c].imshow(a)
        i += 1
plt.show()
