from torch.nn.modules import AdaptiveMaxPool2d
#plotting versus shuffled
#SHUFFLING POS DECODING

import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from google.colab import drive
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import cebra
from cebra import CEBRA
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection






output_dimension = 3 #here, we set as a variable for hypothesis testing below.
cebra_loc_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        #learning_rate= 3e-4,
                        learning_rate= 5e-6,
                        #temperature = 2,
                        temperature_mode = 'auto',
                        min_temperature = .5, #no limit = 62
                        output_dimension=output_dimension,
                        max_iterations=9000, #9000
                        distance='euclidean',
                        conditional='time_delta', #added, keep
                        device='cuda_if_available',
                        num_hidden_units = 10,
                        time_offsets = 1,
                        #hybrid=True, #added <-- if using time
                        verbose=True)


shuff_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        #learning_rate= 3e-4,
                        learning_rate= 5e-6,
                        #temperature = 2,
                        temperature_mode = 'auto',
                        #min_temperature = .74,
                        output_dimension=output_dimension,
                        max_iterations=9000,
                        distance='euclidean',
                        conditional='time_delta', #added, keep
                        device='cuda_if_available',
                        num_hidden_units = 10,
                        time_offsets = 1,
                        #hybrid=True, #added <-- if using time
                        verbose=True)




def decoding_pos_dir(emb_train, emb_test, label_train, label_test, n_neighbors=32):
    pos_decoder = KNeighborsRegressor(n_neighbors, metric = 'cosine')
    pos_decoder.fit(emb_train, label_train)
    pos_pred = pos_decoder.predict(emb_test)
    prediction = pos_pred
    test_score = r2_score(label_test, prediction)
    pos_test_err = np.median(abs(prediction - label_test))
    pos_test_score = r2_score(label_test, prediction)
    # Compute the squared differences for each dimension
    squared_diffs = (prediction - label_test) ** 2
    # Sum the squared differences across columns (axis=1) and take the square root
    distances = np.sqrt(np.sum(squared_diffs, axis=1))
    dis_mean = (np.mean(distances))
    dis_median = (np.median(distances))

    return test_score, pos_test_err, pos_test_score, dis_mean, dis_median

'''
# Create a figure and a 2x3 grid of subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # Adjust figsize as needed
fig2, axs2= plt.subplots(2, 1, figsize=(10, 10))  # Adjust figsize as needed
fig3, axs3 = plt.subplots(2, 1, figsize=(10, 10))  # Adjust figsize as needed

# Convert each subplot to a 3D plot
for i in range(2):
        axs[i] = fig.add_subplot(2, 1, i * 1+1, projection='3d')
        axs2[i] = fig2.add_subplot(2, 1, i * 1+1, projection='3d')
        axs3[i] = fig3.add_subplot(2, 1, i * 1+1, projection='3d')
'''

# Create a figure and a 2x3 grid of subplots
fig, axs = plt.subplots(2, 4, figsize=(15, 10))  # Adjust figsize as needed

# Convert each subplot to a 3D plot
for i in range(2):
    for j in range(4):
        axs[i, j] = fig.add_subplot(2, 4, i * 4 + j + 1, projection='3d')


eyeblink_train = pos22
cell_train21 = traceA21A22_22
cell_train24 = traceA22B24_22 ##########


cebra_loc_model.fit(cell_train21, eyeblink_train)
train21 = cebra_loc_model.transform(cell_train21)
test21 = cebra_loc_model.transform(trace21)


#plot day 22-21 (training)
pos = np.array(eyeblink_train)  # Replace with your pos array
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
plot_hippocampus3d(axs[0, 0], train21, distances, distances, s=4) #<--------------------
#plot_hippocampus3d(axs[0], train21, distances, distances, s=2) #<--------------------


#plot day 21 (test)
pos = np.array(pos21)  # Replace with your pos array
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

plot_hippocampus3d(axs[0, 1], test21, distances, distances, s=4)#<--------------------
#plot_hippocampus3d(axs2[0], test21, distances, distances, s=2) #<--------------------



cebra_loc_model.fit(cell_train24, eyeblink_train)
train24 = cebra_loc_model.transform(cell_train24)
test24 = cebra_loc_model.transform(trace24)

#plot day 22 -24 (training)
pos = np.array(eyeblink_train)  # Replace with your pos array
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
plot_hippocampus3d(axs[0, 2], train24, distances, distances, s=4) #<--------------------
#plot_hippocampus3d(axs[0], train21, distances, distances, s=2) #<--------------------


#plot 24
pos = np.array(pos24)  # Replace with your pos array
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
print(distances.shape)
wanted = np.abs(data - mean) <= threshold * std_dev
wanted = wanted.flatten()
distances = data[wanted]


#distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))


test24 = test24[wanted,:]


ax1, p1 = plot_hippocampus3d(axs[0, 3], test24, distances, distances, s=4)#<--------------------
p1.set_clim(0.05, 0.85)
#plot_hippocampus3d(axs3[0], test24, distances, distances, s=2) #<--------------------





# Convert to numpy array if not already
eyeblink_train = np.array(eyeblink_train)
# Create a new array to hold the shuffled data
eyeblink_shuff = eyeblink_train.copy()
# Shuffle each column independently
for column in range(eyeblink_shuff.shape[1]):
    np.random.shuffle(eyeblink_shuff[:, column])

# Fit the model with the shuffled data
cebra_loc_model.fit(cell_train21, eyeblink_shuff)
train21 = cebra_loc_model.transform(cell_train21)
test21 = cebra_loc_model.transform(trace21)


#plot day 22-21 (shuff)
pos = np.array(eyeblink_shuff)  # Replace with your pos array
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

plot_hippocampus3d(axs[1, 0], train21, distances, distances, s=4) #<--------------------
#plot_hippocampus3d(axs[1], train21, distances, distances, s=2) #<--------------------


#plot day 21 (shuff)
pos = np.array(pos21)  # Replace with your pos array
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

plot_hippocampus3d(axs[1, 1], test21, distances, distances, s=4)#<--------------------
#plot_hippocampus3d(axs2[1], test21, distances, distances, s=2) #<--------------------



cebra_loc_model.fit(cell_train24, eyeblink_shuff)
train24 = cebra_loc_model.transform(cell_train24)
test24 = cebra_loc_model.transform(trace24)

#plot day 22-24 (shuff)
pos = np.array(eyeblink_shuff)  # Replace with your pos array
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

plot_hippocampus3d(axs[1, 2], train24, distances, distances, s=4) #<--------------------
#plot_hippocampus3d(axs[1], train21, distances, distances, s=2) #<--------------------

#plot day 24 shuff
pos = np.array(pos24)  # Replace with your pos array
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

ax2, p2 = plot_hippocampus3d(axs[1, 3], test24[wanted,:], distances, distances, s=4)#<--------------------
#plot_hippocampus3d(axs3[1], test24, distances, distances, s=2) #<--------------------
p2.set_clim(0.1, 0.8)


plt.savefig('my_figure.svg', format='svg')
