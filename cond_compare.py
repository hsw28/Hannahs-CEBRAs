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
from pos_score import pos_score
from plot_hippocampus2d import plot_hippocampus2d
import datetime
from hold_out import hold_out
import os



#for making the shuffle position figure
#can optionally input parameters or hard code them
#python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_compare_script.py ./traceA1An_An.mat ./traceAnB1_An.mat ./traceA1An_A1.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkA1.mat ./eyeblinkB1.mat 2 0 --learning_rate 0.0035 --min_temperature 1.67 --max_iterations 20 --distance cosine



def cond_compare(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, CSUSAn, CSUSA1, CSUSB1, dimensions, learning_rate=0.000775, min_temperature=0.001, max_iterations=6000, distance='cosine'):


    output_dimension = 2


#    output_dimension = 3 #here, we set as a variable for hypothesis testing below.
#    learning_rate = 0.000775
#    min_temperature = 0.1
#    max_iterations = 18000
#    distance = 'cosine'


    cebra_loc_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=learning_rate,
                        temperature_mode='constant',
                        temperature=min_temperature,
                        #min_temperature=min_temperature,
                        output_dimension=output_dimension,
                        max_iterations=max_iterations,
                        distance=distance,
                        conditional='time_delta',
                        device='cuda_if_available',
                        num_hidden_units=32,
                        time_offsets=1,
                        verbose=True)

    shuff_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=learning_rate,
                        temperature_mode='constant',
                        temperature=min_temperature,
                        #min_temperature=min_temperature,
                        output_dimension=output_dimension,
                        max_iterations=max_iterations,
                        distance=distance,
                        conditional='time_delta',
                        device='cuda_if_available',
                        num_hidden_units=32,
                        time_offsets=1,
                        verbose=True)



    # Create a figure and a 2x3 grid of subplots
    ###fig, axs = plt.subplots(2, 4, figsize=(15, 10))  # Adjust figsize as needed
    fig, axs = plt.subplots(2, 6, figsize=(15, 15))  # Adjust figsize as needed


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

    # for held out
    #Pos_test_score_train_A1An_An, Pos_test_err_train_A1An_An, dis_mean_train_A1An_An, dis_median_train_A1An_An = pos_score(trainA_train, trainA1, CSUSAn_train, CSUSAn_test)
    #for test
    #Pos_test_score_train_A1An_A1, Pos_test_err_train_A1An_A1, dis_mean_train_A1An_A1, dis_median_train_A1An_A1 = pos_score(trainA_train, testA1, CSUSAn_train, CSUSA1)

    #plot day An not out (only cells also in day A1)(default model)
    pos = np.array(CSUSAn_train)  # Replace with your pos array
    plot_hippocampus2d(axs[0, 0], trainA_train, pos, pos, s=4) #<--------------------

    #plot day An held out (only cells also in day A1)(default model)
    pos = np.array(CSUSAn_test)  # Replace with your pos array
    plot_hippocampus2d(axs[0, 1], trainA1, pos, pos, s=4) #<--------------------



    #plot day A1 after being trained on An
    pos = np.array(CSUSA1)  # Replace with your pos array
    plot_hippocampus2d(axs[0, 2], testA1, pos, pos, s=4)#<--------------------


    traceAnB1_An_train, traceAnB1_An_test = hold_out(traceAnB1_An, .75)
    CSUSAn_train, CSUSAn_test = hold_out(CSUSAn, .75)
    cebra_loc_model.fit(traceAnB1_An_train, CSUSAn_train)
    trainB1_train = cebra_loc_model.transform(traceAnB1_An_train)
    trainB1 = cebra_loc_model.transform(traceAnB1_An_test)


    #cebra_loc_model.fit(traceAnB1_An, CSUSAn) #this if want to fit on full data and not held out
    testB1 = cebra_loc_model.transform(traceAnB1_B1)


    # for held out
    #Pos_test_score_train_AnB1_An, Pos_test_err_train_AnB1_An, dis_mean_train_AnB1_An, dis_median_train_AnB1_An = pos_score(trainB1_train, trainB1, CSUSAn_train, CSUSAn_test)
    #for test
    #Pos_test_score_train_AnB1_B1, Pos_test_err_train_AnB1_B1, dis_mean_train_AnB1_B1, dis_median_train_AnB1_B1 = pos_score(trainB1_train, testB1, CSUSAn_train, CSUSB1)


    #plot day An held out (only cells also in day B1)(default model)
    pos = np.array(CSUSAn_train)  # Replace with your pos array
    plot_hippocampus2d(axs[0, 3], trainB1_train, pos, pos, s=4) #<--------------------

    #plot day An held out (only cells also in day B1)(default model)
    pos = np.array(CSUSAn_test)  # Replace with your pos array
    plot_hippocampus2d(axs[0, 4], trainB1, pos, pos, s=4) #<--------------------



    #plot B1 after being trained on An
    pos = np.array(CSUSB1)  # Replace with your pos array
    plot_hippocampus2d(axs[0, 5], testB1, pos, pos, s=4)#<--------------------







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


    # for held out
    #Pos_test_score_train_A1An_An_shuff, Pos_test_err_train_A1An_An_shuff, dis_mean_train_A1An_An_shuff, dis_median_train_A1An_An_shuff = pos_score(trainA_train, trainA1, pos_shuff_train, pos_shuff_test)
    #for test
    #Pos_test_score_train_A1An_A1_shuff, Pos_test_err_train_A1An_A1_shuff, dis_mean_train_A1An_A1_shuff, dis_median_train_A1An_A1_shuff = pos_score(trainA_train, testA1, pos_shuff_train, CSUSA1)


    #plot day An not out (only cells also in day A1)(default model)
    pos = np.array(pos_shuff_train)  # Replace with your pos array
    plot_hippocampus2d(axs[1, 0], trainA_train, pos, pos, s=4) #<--------------------

    #plot day An held out (only cells also in day A1)(default model)
    pos = np.array(pos_shuff_test)  # Replace with your pos array
    plot_hippocampus2d(axs[1, 1], trainA1, pos, pos, s=4) #<--------------------

    #plot day A1 (shuff)
    pos = np.array(CSUSA1)  # Replace with your pos array
    plot_hippocampus2d(axs[1, 2], testA1, pos, pos, s=4)#<--------------------


    traceAnB1_An_train, traceAnB1_An_test = hold_out(traceAnB1_An, .75)
    pos_shuff_train, pos_shuff_test = hold_out(pos_shuff, .75)

    cebra_loc_model.fit(traceAnB1_An_train, pos_shuff_train)

    trainB1_train = cebra_loc_model.transform(traceAnB1_An_train)
    trainB1 = cebra_loc_model.transform(traceAnB1_An_test)
    testB1 = cebra_loc_model.transform(traceAnB1_B1)


    #Pos_test_score_train_AnB1_An_shuff, Pos_test_err_train_AnB1_An_shuff, dis_mean_train_AnB1_An_shuff, dis_median_train_AnB1_An_shuff = pos_score(trainB1_train, trainB1, pos_shuff_train, pos_shuff_test)
    #for test
    #Pos_test_score_train_AnB1_B1_shuff, Pos_test_err_train_AnB1_B1_shuff, dis_mean_train_AnB1_B1_shuff, dis_median_train_AnB1_B1_shuff = pos_score(trainB1_train, testB1, pos_shuff_train, CSUSB1)


    #plot day An-B1 (shuff)
    pos = np.array(pos_shuff_train)  # Replace with your pos array
    plot_hippocampus2d(axs[1, 3], trainB1_train, pos, pos, s=4) #<--------------------

    #plot day B1 shuff
    pos = np.array(pos_shuff_test)  # Replace with your pos array
    plot_hippocampus2d(axs[1, 4], trainB1, pos, pos, s=4)#<--------------------

    #plot day B1 shuff
    pos = np.array(CSUSB1)  # Replace with your pos array
    plot_hippocampus2d(axs[1, 5], testB1, pos, pos, s=4)#<--------------------


    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time as a string
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Get the current working directory
    current_directory = os.getcwd()

    # Specify the folder path as the current directory
    folder_path = current_directory

    # Save the plot with the date and time in the file name, in the specified folder
    #file_name = f'{folder_path}/pos_compare_{date_time_str}.svg'
    file_name = f"{folder_path}/pos_compare_lr{learning_rate}_mt{min_temperature}_mi{max_iterations}_d{distance}_{current_time}.svg"

    plt.savefig(file_name, format='svg')

    # Close the figure to free up memory
    plt.close(fig)

    #plt.show()
