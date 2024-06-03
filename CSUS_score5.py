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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score




def CSUS_score5(emb_train, emb_test, label_train, label_test, n_neighbors=32):
    CSUS_decoder = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')

    # Fit the model and predict
    CSUS_decoder.fit(emb_train, label_train)
    predicted = CSUS_decoder.predict(emb_test)

    # Calculate metrics
    #measures the proportion of total correct predictions out of all predictions made.
    accuracy = accuracy_score(label_test, predicted)

    #Precision (Positive Predictive Value): The ratio of correct positive predictions to the total predicted positives.
    #Recall (Sensitivity or True Positive Rate): The ratio of correct positive predictions to the actual positives.
    #F1 Score: The harmonic mean of precision and recall, providing a single metric that balances both.
    precision = precision_score(label_test, predicted, average='macro', zero_division=1)
    recall = recall_score(label_test, predicted, average='macro', zero_division=1)
    f1 = f1_score(label_test, predicted, average='macro', zero_division=1)

    # Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    classes = np.unique(label_train)  # dynamically determining classes
    y_true_bin = label_binarize(label_test, classes=classes)
    y_pred_prob = CSUS_decoder.predict_proba(emb_test)
    roc_auc = roc_auc_score(y_true_bin, y_pred_prob, multi_class='ovr')

    return accuracy, precision, recall, f1, roc_auc
