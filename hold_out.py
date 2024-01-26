import numpy as np

def hold_out(data, percent_to_hold):
    if percent_to_hold > 1:
        percent_to_hold = percent_to_hold / 100

    total_length = len(data)
    hold_length = int(total_length * percent_to_hold)

    # Calculate the start and end indices of the held-out portion
    start_hold = int((total_length - hold_length) / 2)
    end_hold = start_hold + hold_length

    # Split the data into three parts
    neural_train_first = data[:start_hold]
    neural_test = data[start_hold:end_hold]
    neural_train_second = data[end_hold:]

    # Concatenate the first and last parts to form the training set
    neural_train = np.concatenate((neural_train_first, neural_train_second), axis=0)

    # Convert to numpy arrays if not already
    neural_train = np.array(neural_train)
    neural_test = np.array(neural_test)

    return neural_train, neural_test
