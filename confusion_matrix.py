import matplotlib.pyplot as plt
import cebra
from cebra import CEBRA
import numpy as np
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection

def confmatrix(ax, actual, predicted):
    # Create confusion matrix
    confusionMat = np.zeros((5, 5), dtype=int)
    for i in range(len(actual)):
        confusionMat[actual[i]-1][predicted[i]-1] += 1

    # Normalize the confusion matrix to show percentages
    # Row-wise normalization (uncomment the line below if you want column-wise normalization)
    confusionMat_normalized = confusionMat.astype('float') / confusionMat.sum(axis=1)[:, np.newaxis]
    # confusionMat_normalized = confusionMat.astype('float') / confusionMat.sum(axis=0)[np.newaxis, :]  # For column-wise normalization

    # Multiply by 100 to convert to percentage and round off
    confusionMat_percentage = np.round(confusionMat_normalized * 100, 2)

    # Plot heatmap directly on the provided axes object
    sns.heatmap(confusionMat_percentage, annot=True, ax=ax, cmap='YlGnBu', fmt='.2f')
    ax.set_title('Heatmap of Actual vs. Predicted Values (%)')  # Use set_title instead of .title
    ax.set_xlabel('Predicted Values')  # Use set_xlabel instead of .xlabel
    ax.set_ylabel('Actual Values')  # Use set_ylabel instead of .ylabel

    return ax
