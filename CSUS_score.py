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

#scores closeness of CS/US decoding


def CSUS_score(emb_train, emb_test, label_train, label_test, n_neighbors=32):
    CSUS_decoder = KNeighborsClassifier(n_neighbors, metric='cosine')
    CSUS_decoder.fit(emb_train, label_train)
    predicted = CSUS_decoder.predict(emb_test)

    dif = (predicted.astype('int32') - label_test.astype('int32'))
    abs_dif = np.abs(dif)
    num_zeros = np.sum(abs_dif == 0)  # Count the number of zeros in abs_dif
    total_values = len(abs_dif)  # Get the total number of values in abs_dif
    fract = num_zeros / total_values  # Calculate the fraction of correct predictions
    abs_dif = np.mean(abs_dif)  # Calculate the mean absolute difference

    return fract
