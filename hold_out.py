import numpy as np

def hold_out(data, percent_to_train):

    if percent_to_train > 1:
        percent_to_train = percent_to_train/100
    split_idx = int(len(data)* (1-percent_to_train))
    neural_train = data[:split_idx]
    neural_test = data[split_idx:]

    return neural_train.numpy(), neural_test.numpy()
