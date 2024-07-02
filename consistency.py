import sys
import argparse
import cebra
from cebra import CEBRA
import cebra.helper as cebra_helper
import numpy as np
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra')
import matplotlib.pyplot as plt
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection


def consistency(models):

    #    Calculate and plot the consistency scores for a list of models.

    #    Parameters:
    #    models (list): A list containing the model data arrays.

    # Assuming models are stored in a list of arrays named 'models'
    # models = [model1, model2, model3, ..., modelN]

    # Find the length of the shortest model
    min_length = min(len(model) for model in models)

    # Truncate all models to the length of the shortest model
    models = [model[:min_length] for model in models]

    embeddings_datasets, ids, modellabels = [], [], []

    # Add truncated models to the embeddings_datasets
    embeddings_datasets.extend(models)

    # Assuming 'n_runs' is the number of models
    n_runs = len(models)

    # Assuming dataset_ids are provided elsewhere, corresponding to each model
    dataset_ids = ["An", "B1", ..., "Zn"]


    # Calculate consistency scores
    scores_runs, pairs_runs, ids_runs = cebra.sklearn.metrics.consistency_score(
        embeddings=embeddings_datasets,
        between="runs"
    )

    assert scores_runs.shape == (n_runs**2 - n_runs, )
    assert pairs_runs.shape == (n_runs**2 - n_runs, 2)
    assert ids_runs.shape == (n_runs, )


    # Plotting

    fig = plt.figure(figsize=(10,4))

    ax1 = fig.add_subplot(121)

    ax1 = cebra.plot_consistency(
        scores_runs,
        pairs_runs,
        ids_runs,
        vmin=0,
        vmax=100,
        ax=ax1,
        title="Between-runs consistencies"
    )

    plt.show()


    return scores_runs, pairs_runs, ids_runs
