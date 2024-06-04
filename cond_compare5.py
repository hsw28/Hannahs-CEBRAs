import sys
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra')
from torch.nn.modules import AdaptiveMaxPool2d
from cebra import CEBRA
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from plot_hippocampus2d import plot_hippocampus2d
import datetime
from hold_out import hold_out
import os
from CSUS_prediction5 import CSUS_prediction5
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'


#for making the shuffle position figure
#can optionally input parameters or hard code them
#python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_compare_script.py ./traceA1An_An.mat ./traceAnB1_An.mat ./traceA1An_A1.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkA1.mat ./eyeblinkB1.mat 2 0 --learning_rate 0.0035 --min_temperature 1.67 --max_iterations 20 --distance cosine



def cond_compare5(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, CSUSAn, CSUSA1, CSUSB1, dimensions, learning_rate=0.000775, min_temperature=0.001, max_iterations=6000, distance='cosine'):


    output_dimension = 5


#    output_dimension = 3 #here, we set as a variable for hypothesis testing below.
#    learning_rate = 0.000775
#    min_temperature = 0.1
#    max_iterations = 18000
#    distance = 'cosine'


    cebra_loc_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=learning_rate,
                        temperature_mode='constant',
                        #temperature=min_temperature,
                        min_temperature=min_temperature,
                        output_dimension=output_dimension,
                        max_iterations=max_iterations,
                        distance='euclidean',
                        conditional='time_delta',
                        device='cuda_if_available',
                        num_hidden_units=32,
                        time_offsets=1,
                        verbose=True)

    shuff_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=learning_rate,
                        temperature_mode='constant',
                        #temperature=min_temperature,
                        min_temperature=min_temperature,
                        output_dimension=output_dimension,
                        max_iterations=max_iterations,
                        distance='euclidean',
                        conditional='time_delta',
                        device='cuda_if_available',
                        num_hidden_units=32,
                        time_offsets=1,
                        verbose=True)



    # Create a figure and a 2x3 grid of subplots
    ###fig, axs = plt.subplots(2, 4, figsize=(15, 10))  # Adjust figsize as needed
    fig, axs = plt.subplots(2, 6, figsize=(15, 15))  # Adjust figsize as needed
    fig2, axs2 = plt.subplots(2, 6, figsize=(30, 10))


    # Convert each subplot to a 3D plot
    ###for i in range(2):
    for i in range(2):
        for j in range(6):
            ###axs[i, j] = fig.add_subplot(2, 4, i * 4 + j + 1, projection='3d')
            axs[i, j] = fig.add_subplot(2, 6, i * 6 + j + 1, projection='3d')



    traceA1An_An_train, traceA1An_An_test = hold_out(traceA1An_An, 0.75)
    CSUSAn_train, CSUSAn_test = hold_out(CSUSAn, 0.75)

    # Fitting the model on training data
    cebra_loc_model.fit(traceA1An_An_train, CSUSAn_train)

    # Transforming both the training and test data
    trainA_train = cebra_loc_model.transform(traceA1An_An_train)
    trainA1 = cebra_loc_model.transform(traceA1An_An_test)

    #cebra_loc_model.fit(traceA1An_An, CSUSAn) #this if want to fit on full data and not held out
    testA1 = cebra_loc_model.transform(traceA1An_A1)


    #plot day An not out (only cells also in day A1)(default model)
    pos = np.array(CSUSAn_train)  # Replace with your pos array
    plot_hippocampus2d(axs[0, 0], trainA_train, pos, pos, s=4, colormapping=False, binary=True) #<--------------------
    actual, predicted = CSUS_prediction5(trainA_train, trainA_train, CSUSAn_train, CSUSAn_train)
    plot_confusion_matrix(axs2[0, 0], actual, predicted)

    #plot day An held out (only cells also in day A1)(default model)
    pos = np.array(CSUSAn_test)  # Replace with your pos array
    plot_hippocampus2d(axs[0, 1], trainA1, pos, pos, s=4, colormapping=False, binary=True) #<--------------------
    actual, predicted = CSUS_prediction5(trainA_train, trainA1, CSUSAn_train, CSUSAn_test)
    plot_confusion_matrix(axs2[0, 1], actual, predicted)


    #plot day A1 after being trained on An
    pos = np.array(CSUSA1)  # Replace with your pos array
    plot_hippocampus2d(axs[0, 2], testA1, pos, pos, s=4, colormapping=False, binary=True)#<--------------------
    actual, predicted = CSUS_prediction5(trainA_train, testA1, CSUSAn_train, CSUSA1)
    plot_confusion_matrix(axs2[0, 2], actual, predicted)


    traceAnB1_An_train, traceAnB1_An_test = hold_out(traceAnB1_An, .75)
    CSUSAn_train, CSUSAn_test = hold_out(CSUSAn, .75)
    cebra_loc_model.fit(traceAnB1_An_train, CSUSAn_train)
    trainB1_train = cebra_loc_model.transform(traceAnB1_An_train)
    trainB1 = cebra_loc_model.transform(traceAnB1_An_test)


    #cebra_loc_model.fit(traceAnB1_An, CSUSAn) #this if want to fit on full data and not held out
    testB1 = cebra_loc_model.transform(traceAnB1_B1)



    #plot day An held out (only cells also in day B1)(default model)
    pos = np.array(CSUSAn_train)  # Replace with your pos array
    plot_hippocampus2d(axs[0, 3], trainB1_train, pos, pos, s=4, colormapping=False, binary=True) #<--------------------
    actual, predicted = CSUS_prediction5(trainB1_train, trainB1_train, CSUSAn_train, CSUSAn_train)
    plot_confusion_matrix(axs2[0, 3], actual, predicted)

    #plot day An held out (only cells also in day B1)(default model)
    pos = np.array(CSUSAn_test)  # Replace with your pos array
    plot_hippocampus2d(axs[0, 4], trainB1, pos, pos, s=4, colormapping=False, binary=True) #<--------------------
    actual, predicted = CSUS_prediction5(trainB1_train, trainB1, CSUSAn_train, CSUSAn_test)
    plot_confusion_matrix(axs2[0, 4], actual, predicted)



    #plot B1 after being trained on An
    pos = np.array(CSUSB1)  # Replace with your pos array
    plot_hippocampus2d(axs[0, 5], testB1, pos, pos, s=4, colormapping=False, binary=True)#<--------------------
    actual, predicted = CSUS_prediction5(trainB1_train, testB1, CSUSAn_train, CSUSB1)
    plot_confusion_matrix(axs2[0, 5], actual, predicted)







    # Convert to numpy array if not already
    pos = np.array(CSUSAn)
    # Create a new array to hold the shuffled data
    pos_shuff = pos.copy()
    # Shuffle each column independently
    np.random.shuffle(pos_shuff[:])





    # Fit the model with the shuffled data
    traceA1An_An_train, traceA1An_An_test = hold_out(traceA1An_An, .75)
    pos_shuff_train, pos_shuff_test = hold_out(pos_shuff, .75)

    cebra_loc_model.fit(traceA1An_An_train, pos_shuff_train)
    trainA_train = cebra_loc_model.transform(traceA1An_An_train)
    trainA1 = cebra_loc_model.transform(traceA1An_An_test)
    testA1 = cebra_loc_model.transform(traceA1An_A1)



    #plot day An not out (only cells also in day A1)(default model)
    pos = np.array(pos_shuff_train)  # Replace with your pos array
    plot_hippocampus2d(axs[1, 0], trainA_train, pos, pos, s=4, colormapping=False, binary=True) #<--------------------
    actual, predicted = CSUS_prediction5(trainA_train, trainA_train, pos_shuff_train, pos_shuff_train)
    plot_confusion_matrix(axs2[1, 0], actual, predicted)

    #plot day An held out (only cells also in day A1)(default model)
    pos = np.array(pos_shuff_test)  # Replace with your pos array
    plot_hippocampus2d(axs[1, 1], trainA1, pos, pos, s=4, colormapping=False, binary=True) #<--------------------
    actual, predicted = CSUS_prediction5(trainA_train, trainA1, pos_shuff_train, pos_shuff_test)
    plot_confusion_matrix(axs2[1, 1], actual, predicted)

    #plot day A1 (shuff)
    pos = np.array(CSUSA1)  # Replace with your pos array
    plot_hippocampus2d(axs[1, 2], testA1, pos, pos, s=4, colormapping=False, binary=True)#<--------------------
    actual, predicted = CSUS_prediction5(trainA_train, testA1, pos_shuff_train, CSUSA1)
    plot_confusion_matrix(axs2[1, 2], actual, predicted)


    traceAnB1_An_train, traceAnB1_An_test = hold_out(traceAnB1_An, .75)
    pos_shuff_train, pos_shuff_test = hold_out(pos_shuff, .75)

    cebra_loc_model.fit(traceAnB1_An_train, pos_shuff_train)

    trainB1_train = cebra_loc_model.transform(traceAnB1_An_train)
    trainB1 = cebra_loc_model.transform(traceAnB1_An_test)
    testB1 = cebra_loc_model.transform(traceAnB1_B1)



    #plot day An-B1 (shuff)
    pos = np.array(pos_shuff_train)  # Replace with your pos array
    plot_hippocampus2d(axs[1, 3], trainB1_train, pos, pos, s=4, colormapping=False, binary=True) #<--------------------
    actual, predicted = CSUS_prediction5(trainB1_train, trainB1_train, pos_shuff_train, pos_shuff_train)
    plot_confusion_matrix(axs2[1, 3], actual, predicted)

    #plot day B1 shuff
    pos = np.array(pos_shuff_test)  # Replace with your pos array
    plot_hippocampus2d(axs[1, 4], trainB1, pos, pos, s=4, colormapping=False, binary=True)#<--------------------
    actual, predicted = CSUS_prediction5(trainB1_train, trainB1, pos_shuff_train, pos_shuff_test)
    plot_confusion_matrix(axs2[1, 4], actual, predicted)

    #plot day B1 shuff
    pos = np.array(CSUSB1)  # Replace with your pos array
    plot_hippocampus2d(axs[1, 5], testB1, pos, pos, s=4, colormapping=False, binary=True)#<--------------------
    actual, predicted = CSUS_prediction5(trainB1_train, testB1, pos_shuff_train, CSUSB1)
    plot_confusion_matrix(axs2[1, 5], actual, predicted)


    # Save the first figure
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.getcwd()
    file_name = f"{folder_path}/cond_compare5_lr{learning_rate}_mt{min_temperature}_mi{max_iterations}_d{distance}_{current_time}.svg"
    plt.figure(fig.number)
    plt.savefig(file_name, format='svg')
    plt.close(fig)

    # Save the second figure with confusion matrices
    conf_matrix_file_name = f"{folder_path}/cond_compare5conf_lr{learning_rate}_mt{min_temperature}_mi{max_iterations}_d{distance}_{current_time}.svg"
    plt.figure(fig2.number)
    plt.savefig(conf_matrix_file_name, format='svg')
    plt.close(fig2)




def plot_confusion_matrix(ax, actual, predicted):
    matrix = confusion_matrix(actual, predicted)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')
    ax.set_aspect('equal')  # This line sets the aspect ratio to be equal, ensuring the plot is square

    # Depending on your version of matplotlib, you might need to adjust the layout
    plt.tight_layout()
