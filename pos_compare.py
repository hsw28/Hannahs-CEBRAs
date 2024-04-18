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
import matplotlib.pyplot as plt

#for making the shuffle position figure


def pos_compare(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, posAn, posA1, posB1):


    output_dimension = 3 #here, we set as a variable for hypothesis testing below.
    cebra_loc_model = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            #learning_rate= .00026,
                            learning_rate= .001,
                            #temperature = 2,
                            temperature_mode = 'auto',
                            min_temperature = .3,
                            output_dimension=output_dimension,
                            max_iterations=8000, #was 10000 then 8000
                            distance='euclidean',
                            conditional='time_delta', #added, keep
                            device='cuda_if_available',
                            num_hidden_units = 32,
                            time_offsets = 1,
                            #hybrid=True, #added <-- if using time
                            verbose=True)


    shuff_model = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            #learning_rate= .00026,
                            learning_rate= .001,
                            #temperature = 2,
                            temperature_mode = 'auto',
                            min_temperature = .3,
                            output_dimension=output_dimension,
                            max_iterations=8000,
                            distance='euclidean',
                            conditional='time_delta', #added, keep
                            device='cuda_if_available',
                            num_hidden_units = 32,
                            time_offsets = 1,
                            #hybrid=True, #added <-- if using time
                            verbose=True)




    # Create a figure and a 2x3 grid of subplots
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))  # Adjust figsize as needed

    # Convert each subplot to a 3D plot
    for i in range(2):
        for j in range(4):
            axs[i, j] = fig.add_subplot(2, 4, i * 4 + j + 1, projection='3d')


    cebra_loc_model.fit(traceA1An_An, posAn)
    trainA1 = cebra_loc_model.transform(traceA1An_An)
    testA1 = cebra_loc_model.transform(traceA1An_A1)


    #plot day An (only cells also in day A1)(default model)
    pos = np.array(posAn)  # Replace with your pos array
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
    plot_hippocampus3d(axs[0, 0], trainA1, distances, distances, s=4) #<--------------------
    #plot_hippocampus3d(axs[0], trainA1, distances, distances, s=4) #<--------------------


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

    plot_hippocampus3d(axs[0, 1], testA1, distances, distances, s=4)#<--------------------
    #plot_hippocampus3d(axs2[0], testA1, distances, distances, s=4) #<--------------------



    cebra_loc_model.fit(traceAnB1_An, posAn)
    trainB1 = cebra_loc_model.transform(traceAnB1_An)
    testB1 = cebra_loc_model.transform(traceAnB1_B1)

    #plot day An (only cells also in day B1)(default model)
    pos = np.array(posAn)  # Replace with your pos array
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
    plot_hippocampus3d(axs[0, 2], trainB1, distances, distances, s=4) #<--------------------
    #plot_hippocampus3d(axs[0], trainA1, distances, distances, s=4) #<--------------------


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

    data = distances  # Your data
    mean = np.mean(data)
    std_dev = np.std(data)
    threshold = 2.5  # 3 standard deviations
    wanted = np.abs(data - mean) <= threshold * std_dev
    wanted = wanted.flatten()
    distances = data[wanted]


    #distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))


    testB1 = testB1[wanted,:]


    ax1, p1 = plot_hippocampus3d(axs[0, 3], testB1, distances, distances, s=4)#<--------------------
    p1.set_clim(0.05, 0.85)
    #plot_hippocampus3d(axs3[0], testB1, distances, distances, s=4) #<--------------------





    # Convert to numpy array if not already
    pos = np.array(posAn)
    # Create a new array to hold the shuffled data
    pos_shuff = pos.copy()
    # Shuffle each column independently
    for column in range(pos_shuff.shape[1]):
        np.random.shuffle(pos_shuff[:, column])

    # Fit the model with the shuffled data
    cebra_loc_model.fit(traceA1An_An, pos_shuff)
    trainA1 = cebra_loc_model.transform(traceA1An_An)
    testA1 = cebra_loc_model.transform(traceA1An_A1)


    #plot day An cells only in day A1 (shuff)
    pos = np.array(pos_shuff)  # Replace with your pos array
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



    cebra_loc_model.fit(traceAnB1_An, pos_shuff)
    trainB1 = cebra_loc_model.transform(traceAnB1_An)
    testB1 = cebra_loc_model.transform(traceAnB1_B1)

    #plot day An-B1 (shuff)
    pos = np.array(pos_shuff)  # Replace with your pos array
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



    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time as a string
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Specify the folder path
    folder_path = '/Users/Hannah/Programming/data_eyeblink/tempfigs/'

    # Save the plot with the date and time in the file name, in the specified folder
    file_name = f'{folder_path}pos_compare_{date_time_str}.svg'
    plt.savefig(file_name, format='svg')

    plt.show()
