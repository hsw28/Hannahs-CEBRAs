import sys
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra')
import matplotlib.pyplot as plt
import cebra
from cebra import CEBRA
import numpy as np
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection






def plot_hippocampus3d(ax, embedding, label, label2, colormapping=True, binary=False, idx_order=(0, 1, 2), s=3):


    idx1, idx2, idx3 = idx_order

    if colormapping:
        normalized_labels = normalize(label)
        p = ax.scatter(embedding[:, idx1],
                       embedding[:, idx2],
                       embedding[:, idx3],
                       c=normalized_labels,
                       #cmap='rainbow',
                       cmap='rainbow',
                       s=s,
                       alpha=1,
                       depthshade=True,
                       #edgecolors='w',
                       #linewidth=0.1,
                       #rasterized=True)
                       #s=0.5, zorder=1)

        cbar = plt.colorbar(p, ax=ax, shrink=0.5)

    return ax, p


def normalize(array):
    return (array - array.min()) / (array.max() - array.min())
