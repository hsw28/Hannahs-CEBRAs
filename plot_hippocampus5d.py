import matplotlib.pyplot as plt
import cebra
from cebra import CEBRA
import numpy as np
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection



plt.rcParams['svg.fonttype'] = 'none'

def normalize(array):
    return (array - array.min()) / (array.max() - array.min())

def plot_hippocampus5d(ax, embedding, label, label2, colormapping=False, binary=True, idx_order=(0, 1, 2, 3, 4), s=6):
    idx1, idx2, idx3, idx4, idx5 = idx_order

    if colormapping:
        normalized_labels = normalize(label)
        p = ax.scatter(embedding[:, idx1],
                       embedding[:, idx2],
                       embedding[:, idx3],
                       embedding[:, idx4],
                       embedding[:, idx5],
                       c=normalized_labels,
                       cmap='rainbow',
                       #s=0.5, zorder=1)
                       s=s,
                       alpha=1,
                       edgecolors='w',
                       linewidth=0.1,
                       rasterized=False)
        plt.colorbar(p, ax=ax, shrink=0.5)


    one = label2 == 1
    two = label2 == 2
    three = label2 == 3
    four = label2 == 4
    five = label2 == 5



    if binary:
        ax.scatter(embedding[one, idx1],
                embedding[one, idx2],
                c='red',
                s=6, alpha=.85,
                edgecolors='w',
                linewidth=0.1,
                rasterized=False)

        ax.scatter(embedding[two, idx1],
                embedding[two, idx2],
                c='yellow',
                edgecolors='w',
                linewidth=0.1,
                s=6, alpha=.5,
                rasterized=False)

        ax.scatter(embedding[three, idx1],
                embedding[three, idx2],
                c='green',
                edgecolors='w',
                linewidth=0.1,
                s=6, alpha=.5,
                rasterized=False)

        ax.scatter(embedding[four, idx1],
                embedding[four, idx2],
                c='blue',
                edgecolors='w',
                linewidth=0.1,
                s=6, alpha=.5,
                rasterized=False)

        ax.scatter(embedding[five, idx1],
                embedding[five, idx2],
                c='purple',
                edgecolors='w',
                linewidth=0.1,
                s=6, alpha=.5,
                rasterized=False)

    return ax
