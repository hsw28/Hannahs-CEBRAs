import cebra
from cebra import CEBRA
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection

#consistancy across pos decoding

cebra_loc_test21f = cebra_loc_train21[:4225,:]
cebra_loc_test22f = cebra_loc_train22[:4225,:]
cebra_loc_test24f = cebra_loc_train24[:4225,:]
cebra_loc_test25f = cebra_loc_train25[:4225,:]

import matplotlib.pyplot as plt


embeddings_datasets, ids, modellabels = [], [], []


embeddings_datasets.append(cebra_loc_test21f)
embeddings_datasets.append(cebra_loc_test22f)
embeddings_datasets.append(cebra_loc_test24f)
embeddings_datasets.append(cebra_loc_test25f)

n_runs = 4

dataset_ids = ["An-1","An", "B1", "B2"]
#dataset_ids = ["An", "B1"]



scores_runs, pairs_runs, ids_runs = cebra.sklearn.metrics.consistency_score(embeddings=embeddings_datasets,
                                                                            between="runs")

assert scores_runs.shape == (n_runs**2 - n_runs, )
assert pairs_runs.shape == (n_runs**2 - n_runs, 2)
assert ids_runs.shape == (n_runs, )

fig = plt.figure(figsize=(10,4))

ax1 = fig.add_subplot(121)

ax1 = cebra.plot_consistency(scores_runs, pairs_runs, ids_runs, vmin=0, vmax=100, ax=ax1, title="Between-runs consistencies")
