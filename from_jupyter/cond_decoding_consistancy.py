import matplotlib.pyplot as plt
import cebra
from cebra import CEBRA
import numpy as np
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection
print(cebra_loc_test21.shape)
print(cebra_loc_test22.shape)
print(cebra_loc_test24.shape)
print(cebra_loc_test25.shape)

cebra_loc_test21f = cebra_loc_test21[:432,:]
cebra_loc_test22f = cebra_loc_test22[:432,:]
cebra_loc_test24f = cebra_loc_test24[:432,:]
cebra_loc_test25f = cebra_loc_test25[:432,:]

embeddings_datasets, ids, modellabels = [], [], []


embeddings_datasets.append(cebra_loc_test21f)
embeddings_datasets.append(cebra_loc_test22f)
embeddings_datasets.append(cebra_loc_test24f)
embeddings_datasets.append(cebra_loc_test25f)

n_runs = 4

dataset_ids = ["An", "B1"]


scores_runs, pairs_runs, ids_runs = cebra.sklearn.metrics.consistency_score(embeddings=embeddings_datasets,
                                                                            between="runs")

assert scores_runs.shape == (n_runs**2 - n_runs, )
assert pairs_runs.shape == (n_runs**2 - n_runs, 2)
assert ids_runs.shape == (n_runs, )

fig = plt.figure(figsize=(10,4))

ax1 = fig.add_subplot(121)

ax1 = cebra.plot_consistency(scores_runs, pairs_runs, ids_runs, vmin=0, vmax=100, ax=ax1, title="Between-runs consistencies")
