# Sample dataset
data = [{'learn_rate': 0.002, 'min_temp': 1.67, 'max_it': 20000, 'mean_loss': 5.778, 'std_loss': 0.013, 'test_std': 0.005, 'mean_control': 0.805, 'mean_test': 0.682, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 1.67, 'max_it': 40000, 'mean_loss': 5.787, 'std_loss': 0.026, 'test_std': 0.029, 'mean_control': 0.805, 'mean_test': 0.639, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 1.67, 'max_it': 60000, 'mean_loss': 5.794, 'std_loss': 0.037, 'test_std': 0.031, 'mean_control': 0.76, 'mean_test': 0.634, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 1.67, 'max_it': 80000, 'mean_loss': 5.791, 'std_loss': 0.007, 'test_std': 0.025, 'mean_control': 0.811, 'mean_test': 0.696, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 2.33, 'max_it': 20000, 'mean_loss': 5.884, 'std_loss': 0.008, 'test_std': 0.049, 'mean_control': 0.795, 'mean_test': 0.678, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 2.33, 'max_it': 40000, 'mean_loss': 5.858, 'std_loss': 0.025, 'test_std': 0.048, 'mean_control': 0.803, 'mean_test': 0.684, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 2.33, 'max_it': 60000, 'mean_loss': 5.862, 'std_loss': 0.018, 'test_std': 0.062, 'mean_control': 0.76, 'mean_test': 0.65, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 2.33, 'max_it': 80000, 'mean_loss': 5.852, 'std_loss': 0.03, 'test_std': 0.017, 'mean_control': 0.827, 'mean_test': 0.645, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 2.66, 'max_it': 20000, 'mean_loss': 5.9, 'std_loss': 0.038, 'test_std': 0.005, 'mean_control': 0.784, 'mean_test': 0.661, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 2.66, 'max_it': 40000, 'mean_loss': 5.89, 'std_loss': 0.015, 'test_std': 0.026, 'mean_control': 0.797, 'mean_test': 0.643, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 2.66, 'max_it': 60000, 'mean_loss': 5.9, 'std_loss': 0.004, 'test_std': 0.041, 'mean_control': 0.771, 'mean_test': 0.662, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 2.66, 'max_it': 80000, 'mean_loss': 5.891, 'std_loss': 0.006, 'test_std': 0.05, 'mean_control': 0.792, 'mean_test': 0.666, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 3, 'max_it': 20000, 'mean_loss': 5.913, 'std_loss': 0.025, 'test_std': 0.011, 'mean_control': 0.712, 'mean_test': 0.647, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 3, 'max_it': 40000, 'mean_loss': 5.901, 'std_loss': 0.018, 'test_std': 0.039, 'mean_control': 0.784, 'mean_test': 0.662, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 3, 'max_it': 60000, 'mean_loss': 5.919, 'std_loss': 0.01, 'test_std': 0.029, 'mean_control': 0.757, 'mean_test': 0.684, 'n_runs': 3}, {'learn_rate': 0.002, 'min_temp': 3, 'max_it': 80000, 'mean_loss': 5.898, 'std_loss': 0.01, 'test_std': 0.01, 'mean_control': 0.717, 'mean_test': 0.673, 'n_runs': 3}, {'learn_rate': 0.004, 'min_temp': 1.67, 'max_it': 20000, 'mean_loss': 5.781, 'std_loss': 0.019, 'test_std': 0.024, 'mean_control': 0.709, 'mean_test': 0.634, 'n_runs': 3}, {'learn_rate': 0.004, 'min_temp': 1.67, 'max_it': 40000, 'mean_loss': 5.802, 'std_loss': 0.005, 'test_std': 0.031, 'mean_control': 0.845, 'mean_test': 0.66, 'n_runs': 3}]

#for cond decoding


def print_formatted_values(data):
    for entry in data:
        mean_control = round(entry['mean_control'], 2)
        mean_test = round(entry['mean_test'], 2)
        mean_loss = round(entry['mean_loss'], 2)
        std_loss = round(entry['std_loss'],2)
        test_std = round(entry['test_std'],2)
        print(f"{mean_control}, {mean_test}, {mean_loss}, {std_loss}, {test_std}")

    for entry in data:
        learn_rate = entry['learn_rate']
        min_temp = entry['min_temp']
        max_it = entry['max_it']
        n_runs = entry['n_runs']
        print(f"{learn_rate}, {min_temp}, {max_it}, {n_runs}")

'''
def print_formatted_values(data):
    for entry in data:
        mean_A = round(entry['mean_A'], 2)
        mean_B = round(entry['mean_B'], 2)
        mean_loss = round(entry['mean_loss'], 2)
        std_loss = round(entry['std_loss'],2)
        print(f"{mean_A}, {mean_B}, {mean_loss}, {std_loss}")

    for entry in data:
        learn_rate = entry['learn_rate']
        min_temp = entry['min_temp']
        max_it = entry['max_it']
        print(f"{learn_rate}, {min_temp}, {max_it}")

# Function to extract, round, and print specified values for pos decoding
def print_formatted_values(data):
    for entry in data:
        knn_err_test = round(entry['KNN_err_test'], 2)
        test_r2 = round(entry['test_r2'], 2)
        mead_test = round(entry['mead_test'], 2)
        shuff_dif = round(entry['shuff_minus_not'],2)
        shuff_median = round(entry['shuf_med'],2)
        print(f"{knn_err_test}, {test_r2}, {mead_test}, {shuff_dif}, {shuff_median}")
    for entry in data:
        learn_rate = entry['learn_rate']
        min_temp = entry['min_temp']
        max_it = entry['max_it']
        print(f"{learn_rate}, {min_temp}, {max_it}")
'''

# Calling the function to print values
print_formatted_values(data)
