# Define decoding function with kNN decoder. For a simple demo, we will use the fixed number of neighbors 36.

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cebra
from cebra import CEBRA
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score



def CSUS_prediction5(emb_train, emb_test, label_train, label_test, n_neighbors=32):
    CSUS_decoder = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')

    # Fit the model and predict
    CSUS_decoder.fit(emb_train, label_train)
    predicted = CSUS_decoder.predict(emb_test)
    actual = label_test

    return actual, predicted
