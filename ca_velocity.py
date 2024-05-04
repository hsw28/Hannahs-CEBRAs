import numpy as np

def ca_velocity(positions):
    """
    Calculate the velocity between consecutive points in the position array.

    Parameters:
    - positions: numpy array with columns representing x and y coordinates.

    Returns:
    - numpy array of velocities.
    """
    # Calculate differences between consecutive points
    delta_pos = np.diff(positions, axis=0)

    # Calculate Euclidean distance (speed) between points
    velocities = np.sqrt((delta_pos[:, 0] ** 2) + (delta_pos[:, 1] ** 2)) / 0.1333

    return velocities
