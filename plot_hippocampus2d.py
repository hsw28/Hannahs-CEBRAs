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

def plot_hippocampus2d(ax, embedding, label, label2, colormapping=False, binary=True, idx_order=(0, 1), s=6):
    idx1, idx2 = idx_order

    s=6
    if colormapping:
        normalized_labels = normalize(label)
        p = ax.scatter(embedding[:, idx1],
                       embedding[:, idx2],
                       c=normalized_labels,
                       cmap='rainbow',
                       #s=0.5, zorder=1)
                       s=s,
                       alpha=1,
                       edgecolors='w',
                       linewidth=0.1,
                       rasterized=False)
        plt.colorbar(p, ax=ax, shrink=0.5)

    r_ind = label2 == 1
    l_ind = label2 == 2

    if binary:
        ax.scatter(embedding[r_ind, idx1],
                embedding[r_ind, idx2],
                c='blue',
                s=6, alpha=.85,
                edgecolors='w',
                linewidth=0.1,
                rasterized=False)

        ax.scatter(embedding[l_ind, idx1],
                embedding[l_ind, idx2],
                c='red',
                edgecolors='w',
                linewidth=0.1,
                s=6, alpha=.5,
                rasterized=False)

    return ax
