import numpy as np
from scipy.ndimage import gaussian_filter1d

def smoothpos(positions, sigma=2):
    """
    Apply a Gaussian filter to smooth the x and y coordinates of position data.

    Parameters:
    - positions: numpy array with columns representing time, x coordinate, and y coordinate
    - sigma: standard deviation of the Gaussian filter in cm (default is 2)

    Returns:
    - numpy array with the original time column and smoothed x and y coordinates
    """
    # Extract the time, x, and y columns
    time_positions = positions[:, 0]
    x_positions = positions[:, 1]
    y_positions = positions[:, 2]

    # Apply Gaussian filter to the x and y coordinates
    smoothed_x_positions = gaussian_filter1d(x_positions, sigma)
    smoothed_y_positions = gaussian_filter1d(y_positions, sigma)

    # Combine the smoothed coordinates with the original time column
    smoothed_positions = np.column_stack((time_positions, smoothed_x_positions, smoothed_y_positions))

    return smoothed_positions
