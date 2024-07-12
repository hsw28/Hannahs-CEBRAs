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
import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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

    embeddings_datasets, ids, labels = [], [], []


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

    fig = plt.figure(figsize=(30,12))

    ax1 = fig.add_subplot(121)

    ax1 = cebra.plot_consistency(
        scores_runs,
        pairs_runs,
        ids_runs,
        vmin=0,
        vmax=100,
        ax=ax1,
        title="Between-runs consistencies",
        cmap="Blues"
    )

    # Get the current figure and axis
    fig = ax1.get_figure()

    # Loop through all images in the axis and set their colormap
    for im in ax1.get_images():
        im.set_cmap(cm.Blues)

    # Update the plot
    plt.draw()

    # Get the current date and time
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")  # Formats the datetime as Year-Month-Day_Hour-Minute-Second

    # Construct the filename with the current date and time
    filename = f"consistency_plot_{formatted_time}.svg"

    # Save the figure with the dynamically generated filename
    fig.savefig(filename, format='svg')



    return scores_runs, pairs_runs, ids_runs
