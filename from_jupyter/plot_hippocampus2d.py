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

def plot_hippocampus2d(ax, embedding, label, label2, colormapping=False, binary=True, idx_order=(0, 1)):
    idx1, idx2 = idx_order

    if colormapping:
        normalized_labels = normalize(label)
        p = ax.scatter(embedding[:, idx1],
                       embedding[:, idx2],
                       c=normalized_labels,
                       cmap='rainbow',
                       #s=0.5, zorder=1)
                       s=15, zorder=1,
                       rasterized=True)
        plt.colorbar(p, ax=ax, shrink=0.5)

    r_ind = label2 == 1
    l_ind = label2 == 2

    if binary:
     #Using black color and size 5 for the points where label equals 10 or 20
        ax.scatter(embedding[r_ind, idx1],
                embedding[r_ind, idx2],
                c='red',
                s=20, zorder=1, alpha=1,
                rasterized=True) # zorder ensures these points are plotted on top

        ax.scatter(embedding[l_ind, idx1],
                embedding[l_ind, idx2],
                c='blue',
                s=20, zorder=2, alpha=.25,
                rasterized=True)  # zorder ensures these points are plotted on top

    return ax
