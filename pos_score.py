from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score
import sklearn
import numpy as np
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import cebra
from cebra import CEBRA
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection


#scores closeness of position decoding
import sklearn.metrics

# Define decoding function with kNN decoder. For a simple demo, we will use the fixed number of neighbors 36.

def pos_score(emb_train, emb_test, label_train, label_test, n_neighbors=36):
    pos_decoder = KNeighborsRegressor(n_neighbors, metric = 'cosine')
    pos_decoder.fit(emb_train, label_train)
    pos_pred = pos_decoder.predict(emb_test)
    prediction = pos_pred


    pos_test_score = sklearn.metrics.r2_score(label_test, prediction)
    pos_test_err = np.median(abs(prediction - label_test))

    # Compute the squared differences for each dimension
    squared_diffs = (prediction - label_test) ** 2
    # Sum the squared differences across columns (axis=1) and take the square root
    distances = np.sqrt(np.sum(squared_diffs, axis=1))

    dis_mean = (np.mean(distances))
    dis_median = (np.median(distances))


    return pos_test_score, pos_test_err, dis_mean, dis_median
