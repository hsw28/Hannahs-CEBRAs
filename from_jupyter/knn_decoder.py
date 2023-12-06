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
from matplotlib.collections import LineCollections

def decoding_CSUS5(emb_train, emb_test, label_train, label_test, n_neighbors=10):
    CSUS_decoder = KNeighborsClassifier(n_neighbors, metric = 'cosine')

    CSUS_decoder.fit(emb_train, label_train)

    predicted = CSUS_decoder.predict(emb_test)
    actual = label_test

    #print(prediction)


    CSUS_test_err = np.median(abs(predicted[:] - label_test[:]))
    CSUS_test_score = sklearn.metrics.r2_score(label_test[:], predicted[:])


    dif = (predicted.astype('int32') - label_test.astype('int32'))
    abs_dif = np.abs(dif)
    num_zeros = np.sum(abs_dif == 0)  # Count the number of zeros in abs_dif
    total_values = len(abs_dif)  # Get the total number of values in abs_dif

    fract = num_zeros / total_values
    abs_dif = np.mean(abs_dif)

    return fract, CSUS_test_err, CSUS_test_score, abs_dif, predicted, actual
