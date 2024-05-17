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
from plot_hippocampus3d import plot_hippocampus3d
import datetime
from hold_out import hold_out
import os



#for making the shuffle position figure
#can optionally input parameters or hard code them
#not inputed:
    # python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/pos_compare_script.py ./traceA1An_An.mat ./traceAnB1_An.mat ./traceA1An_A1.mat ./traceAnB1_B1.mat ./posAn.mat ./posA1.mat ./posB1.mat;
#inputed:
    # python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/pos_compare_script.py ./traceA1An_An.mat ./traceAnB1_An.mat ./traceA1An_A1.mat ./traceAnB1_B1.mat ./posAn.mat ./posA1.mat ./posB1.mat --learning_rate 0.1 --min_temperature 0.5 --max_iterations 20 --distance euclidean


def pos_compare(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, posAn, posA1, posB1, learning_rate=0.000775, min_temperature=0.001, max_iterations=6000, distance='cosine'):


    output_dimension = 3


#    output_dimension = 3 #here, we set as a variable for hypothesis testing below.
#    learning_rate = 0.000775
#    min_temperature = 0.1
#    max_iterations = 18000
#    distance = 'cosine'

    cebra_loc_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=learning_rate,
                        temperature_mode='auto',
                        #temperature=0.6,
                        min_temperature=min_temperature,
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
                        temperature_mode='auto',
                        #temperature=0.6,
                        min_temperature=min_temperature,
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
    fig, axs = plt.subplots(4, 6, figsize=(15, 15))  # Adjust figsize as needed


    # Convert each subplot to a 3D plot
    ###for i in range(2):
    for i in range(4):
        for j in range(6):
            ###axs[i, j] = fig.add_subplot(2, 4, i * 4 + j + 1, projection='3d')
            axs[i, j] = fig.add_subplot(4, 6, i * 4 + j + 1, projection='3d')



    traceA1An_An_train, traceA1An_An_test = hold_out(traceA1An_An, 0.75)
    posAn_train, posAn_test = hold_out(posAn, 0.75)

    # Fitting the model on training data
    cebra_loc_model.fit(traceA1An_An_train, posAn_train)

    # Transforming both the training and test data
    trainA_train = cebra_loc_model.transform(traceA1An_An_train)
    trainA1 = cebra_loc_model.transform(traceA1An_An_test)

    #cebra_loc_model.fit(traceA1An_An, posAn) #this if want to fit on full data and not held out
    testA1 = cebra_loc_model.transform(traceA1An_A1)

    # for held out
    Pos_test_score_train_A1An_An, Pos_test_err_train_A1An_An, dis_mean_train_A1An_An, dis_median_train_A1An_An = pos_score(trainA_train, trainA1, posAn_train, posAn_test)
    #for test
    Pos_test_score_train_A1An_A1, Pos_test_err_train_A1An_A1, dis_mean_train_A1An_A1, dis_median_train_A1An_A1 = pos_score(trainA_train, testA1, posAn_train, posA1)

    #plot day An not out (only cells also in day A1)(default model)
    pos = np.array(posAn_train)  # Replace with your pos array
    # Identify a corner, e.g., top-right corner
    corner_x = np.min(pos[:, 0])  # Maximum x-coordinate
    corner_y = np.max(pos[:, 1])  # Maximum y-coordinate
    corner = np.array([corner_x, corner_y])
    center_x = np.mean(pos[:, 0])  # Mean of x-coordinates
    center_y = np.mean(pos[:, 1])  # Mean of y-coordinates
    center = np.array([center_x, center_y])
    # Calculate distances from each point to the corner
    distances = np.sqrt(np.sum((pos - corner) ** 2, axis=1))
    #distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))
    plot_hippocampus3d(axs[0, 0], trainA_train, distances, distances, s=4) #<--------------------
    #plot_hippocampus3d(axs[0], trainA1, distances, distances, s=4) #<--------------------
    plot_hippocampus3d(axs[2, 0], trainA_train, pos[:, 0], pos[:, 0], s=4) #<--------------------new
    plot_hippocampus3d(axs[3, 0], trainA_train, pos[:, 1], pos[:, 1], s=4) #<--------------------new

    #plot day An held out (only cells also in day A1)(default model)
    pos = np.array(posAn_test)  # Replace with your pos array
    # Identify a corner, e.g., top-right corner
    corner_x = np.min(pos[:, 0])  # Maximum x-coordinate
    corner_y = np.max(pos[:, 1])  # Maximum y-coordinate
    corner = np.array([corner_x, corner_y])
    center_x = np.mean(pos[:, 0])  # Mean of x-coordinates
    center_y = np.mean(pos[:, 1])  # Mean of y-coordinates
    center = np.array([center_x, center_y])
    # Calculate distances from each point to the corner
    distances = np.sqrt(np.sum((pos - corner) ** 2, axis=1))
    #distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))
    plot_hippocampus3d(axs[0, 1], trainA1, distances, distances, s=4) #<--------------------
    #plot_hippocampus3d(axs[0], trainA1, distances, distances, s=4) #<--------------------
    plot_hippocampus3d(axs[2, 1], trainA1, pos[:, 0], pos[:, 0], s=4) #<--------------------new
    plot_hippocampus3d(axs[3, 1], trainA1, pos[:, 1], pos[:, 1], s=4) #<--------------------new


    #plot day A1 after being trained on An
    pos = np.array(posA1)  # Replace with your pos array
    # Identify a corner, e.g., top-right corner
    corner_x = np.min(pos[:, 0])  # Maximum x-coordinate
    corner_y = np.max(pos[:, 1])  # Maximum y-coordinate
    corner = np.array([corner_x, corner_y])
    center_x = np.mean(pos[:, 0])  # Mean of x-coordinates
    center_y = np.mean(pos[:, 1])  # Mean of y-coordinates
    center = np.array([center_x, center_y])
    # Calculate distances from each point to the corner
    distances = np.sqrt(np.sum((pos - corner) ** 2, axis=1))
    #distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))

    plot_hippocampus3d(axs[0, 2], testA1, distances, distances, s=4)#<--------------------
    #plot_hippocampus3d(axs2[0], testA1, distances, distances, s=4) #<--------------------
    plot_hippocampus3d(axs[2, 2], testA1, pos[:, 0], pos[:, 0], s=4) #<--------------------new
    plot_hippocampus3d(axs[3, 2], testA1, pos[:, 1], pos[:, 1], s=4) #<--------------------new



    traceAnB1_An_train, traceAnB1_An_test = hold_out(traceAnB1_An, .75)
    posAn_train, posAn_test = hold_out(posAn, .75)
    cebra_loc_model.fit(traceAnB1_An_train, posAn_train)
    trainB1_train = cebra_loc_model.transform(traceAnB1_An_train)
    trainB1 = cebra_loc_model.transform(traceAnB1_An_test)


    #cebra_loc_model.fit(traceAnB1_An, posAn) #this if want to fit on full data and not held out
    testB1 = cebra_loc_model.transform(traceAnB1_B1)


    # for held out
    Pos_test_score_train_AnB1_An, Pos_test_err_train_AnB1_An, dis_mean_train_AnB1_An, dis_median_train_AnB1_An = pos_score(trainB1_train, trainB1, posAn_train, posAn_test)
    #for test
    Pos_test_score_train_AnB1_B1, Pos_test_err_train_AnB1_B1, dis_mean_train_AnB1_B1, dis_median_train_AnB1_B1 = pos_score(trainB1_train, testB1, posAn_train, posB1)


    #plot day An held out (only cells also in day B1)(default model)
    pos = np.array(posAn_train)  # Replace with your pos array
    # Identify a corner, e.g., top-right corner
    corner_x = np.min(pos[:, 0])  # Maximum x-coordinate
    corner_y = np.max(pos[:, 1])  # Maximum y-coordinate
    corner = np.array([corner_x, corner_y])
    center_x = np.mean(pos[:, 0])  # Mean of x-coordinates
    center_y = np.mean(pos[:, 1])  # Mean of y-coordinates
    center = np.array([center_x, center_y])
    # Calculate distances from each point to the corner
    distances = np.sqrt(np.sum((pos - corner) ** 2, axis=1))
    #distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))
    plot_hippocampus3d(axs[0, 3], trainB1_train, distances, distances, s=4) #<--------------------
    #plot_hippocampus3d(axs[0], trainA1, distances, distances, s=4) #<--------------------
    plot_hippocampus3d(axs[2, 3], trainB1_train, pos[:, 0], pos[:, 0], s=4) #<--------------------new
    plot_hippocampus3d(axs[3, 3], trainB1_train, pos[:, 1], pos[:, 1], s=4) #<--------------------new


    #plot day An held out (only cells also in day B1)(default model)
    pos = np.array(posAn_test)  # Replace with your pos array
    # Identify a corner, e.g., top-right corner
    corner_x = np.min(pos[:, 0])  # Maximum x-coordinate
    corner_y = np.max(pos[:, 1])  # Maximum y-coordinate
    corner = np.array([corner_x, corner_y])
    center_x = np.mean(pos[:, 0])  # Mean of x-coordinates
    center_y = np.mean(pos[:, 1])  # Mean of y-coordinates
    center = np.array([center_x, center_y])
    # Calculate distances from each point to the corner
    distances = np.sqrt(np.sum((pos - corner) ** 2, axis=1))
    #distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))
    plot_hippocampus3d(axs[0, 4], trainB1, distances, distances, s=4) #<--------------------
    #plot_hippocampus3d(axs[0], trainA1, distances, distances, s=4) #<--------------------
    plot_hippocampus3d(axs[2, 4], trainB1, pos[:, 0], pos[:, 0], s=4) #<--------------------new
    plot_hippocampus3d(axs[3, 4], trainB1, pos[:, 1], pos[:, 1], s=4) #<--------------------new


    #plot B1 after being trained on An
    pos = np.array(posB1)  # Replace with your pos array
    # Identify a corner, e.g., top-right corner
    corner_x = np.min(pos[:, 0])  # Maximum x-coordinate
    corner_y = np.min(pos[:, 1])  # Maximum y-coordinate
    # Find the index of the minimum x-coordinate, # Use this index to find the corresponding y-coordinate
    min_x_index = np.argmin(pos[:, 0])
    corner_y = pos[min_x_index, 1]
    corner = np.array([corner_x, corner_y])
    center_x = np.mean(pos[:, 0])  # Mean of x-coordinates
    center_y = np.mean(pos[:, 1])  # Mean of y-coordinates
    center = np.array([center_x, center_y])
    # Calculate distances from each point to the corner
    distances = np.sqrt(np.sum((pos - corner) ** 2, axis=1))


    ##WHY DID I DO THIS?
    #data = distances  # Your data
    #mean = np.mean(data)
    #std_dev = np.std(data)
    #threshold = 2.5  # 3 standard deviations
    #wanted = np.abs(data - mean) <= threshold * std_dev
    #wanted = wanted.flatten()
    #distances = data[wanted]
    #distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))
    #testB1 = testB1[wanted,:]

    plot_hippocampus3d(axs[0, 5], testB1, distances, distances, s=4)#<--------------------
    plot_hippocampus3d(axs[2, 5], testB1, pos[:, 0], pos[:, 0], s=4) #<--------------------new
    plot_hippocampus3d(axs[3, 5], testB1, pos[:, 1], pos[:, 1], s=4) #<--------------------new
    ###p1.set_clim(0.05, 0.85)
    #plot_hippocampus3d(axs3[0], testB1, distances, distances, s=4) #<--------------------





    '''
    # Convert to numpy array if not already
    pos = np.array(posAn)
    # Create a new array to hold the shuffled data
    pos_shuff = pos.copy()
    # Shuffle each column independently
    for column in range(pos_shuff.shape[1]):
        np.random.shuffle(pos_shuff[:, column])



    # Fit the model with the shuffled data
    traceA1An_An_train, traceA1An_An_test = hold_out(traceA1An_An, .75)
    pos_shuff_train, pos_shuff_test = hold_out(pos_shuff, .75)

    cebra_loc_model.fit(traceA1An_An_train, pos_shuff_train)
    trainA_train = cebra_loc_model.transform(traceA1An_An_train)
    trainA1 = cebra_loc_model.transform(traceA1An_An_test)
    testA1 = cebra_loc_model.transform(traceA1An_A1)


    # for held out
    Pos_test_score_train_A1An_An_shuff, Pos_test_err_train_A1An_An_shuff, dis_mean_train_A1An_An_shuff, dis_median_train_A1An_An_shuff = pos_score(trainA_train, trainA1, pos_shuff_train, pos_shuff_test)
    #for test
    Pos_test_score_train_A1An_A1_shuff, Pos_test_err_train_A1An_A1_shuff, dis_mean_train_A1An_A1_shuff, dis_median_train_A1An_A1_shuff = pos_score(trainA_train, testA1, pos_shuff_train, posA1)


    #plot day An cells only in day A1 (shuff)
    pos = np.array(pos_shuff_test)  # Replace with your pos array
    # Identify a corner, e.g., top-right corner
    corner_x = np.min(pos[:, 0])  # Maximum x-coordinate
    corner_y = np.max(pos[:, 1])  # Maximum y-coordinate
    corner = np.array([corner_x, corner_y])
    center_x = np.mean(pos[:, 0])  # Mean of x-coordinates
    center_y = np.mean(pos[:, 1])  # Mean of y-coordinates
    center = np.array([center_x, center_y])
    # Calculate distances from each point to the corner
    distances = np.sqrt(np.sum((pos - corner) ** 2, axis=1))
    #distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))

    plot_hippocampus3d(axs[1, 0], trainA1, distances, distances, s=4) #<--------------------
    #plot_hippocampus3d(axs[1], trainA1, distances, distances, s=4) #<--------------------


    #plot day A1 (shuff)
    pos = np.array(posA1)  # Replace with your pos array
    # Identify a corner, e.g., top-right corner
    corner_x = np.min(pos[:, 0])  # Maximum x-coordinate
    corner_y = np.max(pos[:, 1])  # Maximum y-coordinate
    corner = np.array([corner_x, corner_y])
    center_x = np.mean(pos[:, 0])  # Mean of x-coordinates
    center_y = np.mean(pos[:, 1])  # Mean of y-coordinates
    center = np.array([center_x, center_y])
    # Calculate distances from each point to the corner
    distances = np.sqrt(np.sum((pos - corner) ** 2, axis=1))
    #distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))

    plot_hippocampus3d(axs[1, 1], testA1, distances, distances, s=4)#<--------------------
    #plot_hippocampus3d(axs2[1], testA1, distances, distances, s=4) #<--------------------


    traceAnB1_An_train, traceAnB1_An_test = hold_out(traceAnB1_An, .75)
    pos_shuff_train, pos_shuff_test = hold_out(pos_shuff, .75)

    cebra_loc_model.fit(traceAnB1_An_train, pos_shuff_train)

    trainB1_train = cebra_loc_model.transform(traceAnB1_An_train)
    trainB1 = cebra_loc_model.transform(traceAnB1_An_test)
    testB1 = cebra_loc_model.transform(traceAnB1_B1)


    Pos_test_score_train_AnB1_An_shuff, Pos_test_err_train_AnB1_An_shuff, dis_mean_train_AnB1_An_shuff, dis_median_train_AnB1_An_shuff = pos_score(trainB1_train, trainB1, pos_shuff_train, pos_shuff_test)
    #for test
    Pos_test_score_train_AnB1_B1_shuff, Pos_test_err_train_AnB1_B1_shuff, dis_mean_train_AnB1_B1_shuff, dis_median_train_AnB1_B1_shuff = pos_score(trainB1_train, testB1, pos_shuff_train, posB1)


    #plot day An-B1 (shuff)
    pos = np.array(pos_shuff_test)  # Replace with your pos array
    # Identify a corner, e.g., top-right corner
    corner_x = np.min(pos[:, 0])  # Maximum x-coordinate
    corner_y = np.max(pos[:, 1])  # Maximum y-coordinate
    corner = np.array([corner_x, corner_y])
    center_x = np.mean(pos[:, 0])  # Mean of x-coordinates
    center_y = np.mean(pos[:, 1])  # Mean of y-coordinates
    center = np.array([center_x, center_y])
    # Calculate distances from each point to the corner
    distances = np.sqrt(np.sum((pos - corner) ** 2, axis=1))
    #distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))

    plot_hippocampus3d(axs[1, 2], trainB1, distances, distances, s=4) #<--------------------
    #plot_hippocampus3d(axs[1], trainA1, distances, distances, s=4) #<--------------------

    #plot day B1 shuff
    pos = np.array(posB1)  # Replace with your pos array
    # Identify a corner, e.g., top-right corner
    corner_x = np.min(pos[:, 0])  # Maximum x-coordinate
    corner_y = np.min(pos[:, 1])  # Maximum y-coordinate
    # Find the index of the minimum x-coordinate, # Use this index to find the corresponding y-coordinate
    min_x_index = np.argmin(pos[:, 0])
    corner_y = pos[min_x_index, 1]
    corner = np.array([corner_x, corner_y])
    center_x = np.mean(pos[:, 0])  # Mean of x-coordinates
    center_y = np.mean(pos[:, 1])  # Mean of y-coordinates
    center = np.array([center_x, center_y])
    # Calculate distances from each point to the corner
    #distances = np.sqrt(np.sum((pos - corner) ** 2, axis=1))
    distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))

    data = distances  # Your data
    mean = np.mean(data)
    std_dev = np.std(data)
    threshold = 2.5  # 3 standard deviations
    wanted = np.abs(data - mean) <= threshold * std_dev
    wanted = wanted.flatten()
    distances = data[wanted]

    ax2, p2 = plot_hippocampus3d(axs[1, 3], testB1[wanted,:], distances, distances, s=4)#<--------------------
    #plot_hippocampus3d(axs3[1], testB1, distances, distances, s=4) #<--------------------
    p2.set_clim(0.1, 0.8)

    '''

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

    '''
    # Same thing, different point size
    fig2, axs2 = plt.subplots(4, 4, figsize=(15, 10))
    for i in range(4):
        for j in range(4):
            axs2[i, j] = fig2.add_subplot(4, 4, i * 4 + j + 1, projection='3d')

    # Plotting code with s=5
    plot_hippocampus3d(axs2[0, 0], trainA1, distances, distances, s=5)
    plot_hippocampus3d(axs2[2, 0], trainA1, pos[:, 0], pos[:, 0], s=5)
    plot_hippocampus3d(axs2[3, 0], trainA1, pos[:, 1], pos[:, 1], s=5)
    plot_hippocampus3d(axs2[0, 1], testA1, distances, distances, s=5)
    plot_hippocampus3d(axs2[2, 1], testA1, pos[:, 0], pos[:, 0], s=5)
    plot_hippocampus3d(axs2[3, 1], testA1, pos[:, 1], pos[:, 1], s=5)
    plot_hippocampus3d(axs2[0, 2], trainB1, distances, distances, s=5)
    plot_hippocampus3d(axs2[2, 2], trainB1, pos[:, 0], pos[:, 0], s=5)
    plot_hippocampus3d(axs2[3, 2], trainB1, pos[:, 1], pos[:, 1], s=5)
    plot_hippocampus3d(axs2[0, 3], testB1, distances, distances, s=5)
    plot_hippocampus3d(axs2[2, 3], testB1, pos[:, 0], pos[:, 0], s=5)
    plot_hippocampus3d(axs2[3, 3], testB1, pos[:, 1], pos[:, 1], s=5)

    # Save the second figure
    file_name2 = f"{current_directory}/pos_compare_lr{learning_rate}_mt{min_temperature}_mi{max_iterations}_d{distance}_{current_time}_size5.svg"
    plt.savefig(file_name2, format='svg')
    plt.close(fig2)


    # Same thing, different point size
    fig3, axs3 = plt.subplots(4, 4, figsize=(15, 10))
    for i in range(4):
        for j in range(4):
            axs3[i, j] = fig3.add_subplot(4, 4, i * 4 + j + 1, projection='3d')

    # Plotting code with s=5
    plot_hippocampus3d(axs3[0, 0], trainA1, distances, distances, s=3)
    plot_hippocampus3d(axs3[2, 0], trainA1, pos[:, 0], pos[:, 0], s=3)
    plot_hippocampus3d(axs3[3, 0], trainA1, pos[:, 1], pos[:, 1], s=3)
    plot_hippocampus3d(axs3[0, 1], testA1, distances, distances, s=3)
    plot_hippocampus3d(axs3[2, 1], testA1, pos[:, 0], pos[:, 0], s=3)
    plot_hippocampus3d(axs3[3, 1], testA1, pos[:, 1], pos[:, 1], s=3)
    plot_hippocampus3d(axs3[0, 2], trainB1, distances, distances, s=3)
    plot_hippocampus3d(axs3[2, 2], trainB1, pos[:, 0], pos[:, 0], s=3)
    plot_hippocampus3d(axs3[3, 2], trainB1, pos[:, 1], pos[:, 1], s=3)
    plot_hippocampus3d(axs3[0, 3], testB1, distances, distances, s=3)
    plot_hippocampus3d(axs3[2, 3], testB1, pos[:, 0], pos[:, 0], s=3)
    plot_hippocampus3d(axs3[3, 3], testB1, pos[:, 1], pos[:, 1], s=3)

    # Save the second figure
    file_name3 = f"{current_directory}/pos_compare_lr{learning_rate}_mt{min_temperature}_mi{max_iterations}_d{distance}_{current_time}_size3.svg"
    plt.savefig(file_name3, format='svg')
    plt.close(fig3)


    # Save the second figure
    file_name2 = f"{current_directory}/pos_compare_lr{learning_rate}_mt{min_temperature}_mi{max_iterations}_d{distance}_{current_time}_size5.svg"
    plt.savefig(file_name2, format='svg')
    plt.close(fig2)


    # Same thing, different point size
    fig4, axs4 = plt.subplots(4, 4, figsize=(15, 10))
    for i in range(4):
        for j in range(4):
            axs4[i, j] = fig3.add_subplot(4, 4, i * 4 + j + 1, projection='3d')

    # Plotting code with s=5
    plot_hippocampus3d(axs4[0, 0], trainA1, distances, distances, s=6)
    plot_hippocampus3d(axs4[2, 0], trainA1, pos[:, 0], pos[:, 0], s=6)
    plot_hippocampus3d(axs4[3, 0], trainA1, pos[:, 1], pos[:, 1], s=6)
    plot_hippocampus3d(axs4[0, 1], testA1, distances, distances, s=6)
    plot_hippocampus3d(axs4[2, 1], testA1, pos[:, 0], pos[:, 0], s=6)
    plot_hippocampus3d(axs4[3, 1], testA1, pos[:, 1], pos[:, 1], s=6)
    plot_hippocampus3d(axs4[0, 2], trainB1, distances, distances, s=6)
    plot_hippocampus3d(axs4[2, 2], trainB1, pos[:, 0], pos[:, 0], s=6)
    plot_hippocampus3d(axs4[3, 2], trainB1, pos[:, 1], pos[:, 1], s=6)
    plot_hippocampus3d(axs4[0, 3], testB1, distances, distances, s=6)
    plot_hippocampus3d(axs4[2, 3], testB1, pos[:, 0], pos[:, 0], s=6)
    plot_hippocampus3d(axs4[3, 3], testB1, pos[:, 1], pos[:, 1], s=6)

    # Save the second figure
    file_name4 = f"{current_directory}/pos_compare_lr{learning_rate}_mt{min_temperature}_mi{max_iterations}_d{distance}_{current_time}_size6.svg"
    plt.savefig(file_name4, format='svg')
    plt.close(fig4)
    '''
