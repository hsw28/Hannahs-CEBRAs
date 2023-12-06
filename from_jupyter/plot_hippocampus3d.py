import matplotlib.pyplot as plt
import cebra
from cebra import CEBRA
import numpy as np
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection


def normalize(array):
    return (array - array.min()) / (array.max() - array.min())

def plot_hippocampus3d(ax, embedding, label, label2, colormapping=True, binary=False, idx_order=(0, 1, 2), s=10):


    idx1, idx2, idx3 = idx_order

    if colormapping:
        normalized_labels = normalize(label)
        p = ax.scatter(embedding[:, idx1],
                       embedding[:, idx2],
                       embedding[:, idx3],
                       c=normalized_labels,
                       cmap='rainbow',
                       s=s,
                       alpha=1,
                       rasterized=True)
                       #s=0.5, zorder=1)

        cbar = plt.colorbar(p, ax=ax, shrink=0.5)

    '''
    r_ind = label2 == 10
    l_ind = label2 == 20


    if binary:
     #Using black color and size 5 for the points where label equals 10 or 20
        ax.scatter(embedding[r_ind, idx1],
                embedding[r_ind, idx2],
                embedding[r_ind, idx3],
                c='red')
                #s=20, zorder=2)  # zorder ensures these points are plotted on top

        ax.scatter(embedding[l_ind, idx1],
                embedding[l_ind, idx2],
                embedding[l_ind, idx3],
                c='blue')
                #s=20, zorder=2)  # zorder ensures these points are plotted on top
    '''
    return ax, p
