# Sample dataset
data = [{'learn_rate': 0.0015, 'min_temp': 3.5, 'max_it': 50000, 'mean_loss': 5.942, 'std_loss': 0.015, 'test_std': 0.133, 'mean_control': 0.845, 'mean_test': 0.488, 'n_runs': 5}, {'learn_rate': 0.0015, 'min_temp': 3.5, 'max_it': 70000, 'mean_loss': 5.933, 'std_loss': 0.005, 'test_std': 0.099, 'mean_control': 0.782, 'mean_test': 0.656, 'n_runs': 5}, {'learn_rate': 0.0015, 'min_temp': 4, 'max_it': 50000, 'mean_loss': 5.971, 'std_loss': 0.01, 'test_std': 0.038, 'mean_control': 0.854, 'mean_test': 0.345, 'n_runs': 5}, {'learn_rate': 0.0015, 'min_temp': 4, 'max_it': 70000, 'mean_loss': 5.958, 'std_loss': 0.005, 'test_std': 0.04, 'mean_control': 0.856, 'mean_test': 0.444, 'n_runs': 5}, {'learn_rate': 0.0015, 'min_temp': 5, 'max_it': 50000, 'mean_loss': 6.005, 'std_loss': 0.015, 'test_std': 0.103, 'mean_control': 0.821, 'mean_test': 0.42, 'n_runs': 5}, {'learn_rate': 0.0015, 'min_temp': 5, 'max_it': 70000, 'mean_loss': 6.008, 'std_loss': 0.01, 'test_std': 0.081, 'mean_control': 0.784, 'mean_test': 0.619, 'n_runs': 5}, {'learn_rate': 0.0035, 'min_temp': 3.5, 'max_it': 50000, 'mean_loss': 5.942, 'std_loss': 0.011, 'test_std': 0.137, 'mean_control': 0.827, 'mean_test': 0.48, 'n_runs': 5}, {'learn_rate': 0.0035, 'min_temp': 3.5, 'max_it': 70000, 'mean_loss': 5.946, 'std_loss': 0.012, 'test_std': 0.044, 'mean_control': 0.803, 'mean_test': 0.616, 'n_runs': 5}, {'learn_rate': 0.0035, 'min_temp': 4, 'max_it': 50000, 'mean_loss': 6.01, 'std_loss': 0.072, 'test_std': 0.053, 'mean_control': 0.714, 'mean_test': 0.669, 'n_runs': 5}, {'learn_rate': 0.0035, 'min_temp': 4, 'max_it': 70000, 'mean_loss': 5.965, 'std_loss': 0.005, 'test_std': 0.12, 'mean_control': 0.837, 'mean_test': 0.618, 'n_runs': 5}, {'learn_rate': 0.0035, 'min_temp': 5, 'max_it': 50000, 'mean_loss': 6.032, 'std_loss': 0.064, 'test_std': 0.131, 'mean_control': 0.738, 'mean_test': 0.526, 'n_runs': 5}, {'learn_rate': 0.0035, 'min_temp': 5, 'max_it': 70000, 'mean_loss': 6.01, 'std_loss': 0.01, 'test_std': 0.096, 'mean_control': 0.792, 'mean_test': 0.398, 'n_runs': 5}, {'learn_rate': 0.0045, 'min_temp': 3.5, 'max_it': 50000, 'mean_loss': 5.979, 'std_loss': 0.073, 'test_std': 0.056, 'mean_control': 0.706, 'mean_test': 0.573, 'n_runs': 5}, {'learn_rate': 0.0045, 'min_temp': 3.5, 'max_it': 70000, 'mean_loss': 5.944, 'std_loss': 0.012, 'test_std': 0.14, 'mean_control': 0.808, 'mean_test': 0.602, 'n_runs': 5}, {'learn_rate': 0.0045, 'min_temp': 4, 'max_it': 50000, 'mean_loss': 5.963, 'std_loss': 0.009, 'test_std': 0.051, 'mean_control': 0.784, 'mean_test': 0.535, 'n_runs': 5}, {'learn_rate': 0.0045, 'min_temp': 4, 'max_it': 70000, 'mean_loss': 5.977, 'std_loss': 0.013, 'test_std': 0.101, 'mean_control': 0.749, 'mean_test': 0.536, 'n_runs': 5}, {'learn_rate': 0.0045, 'min_temp': 5, 'max_it': 50000, 'mean_loss': 6.037, 'std_loss': 0.061, 'test_std': 0.142, 'mean_control': 0.746, 'mean_test': 0.629, 'n_runs': 5}, {'learn_rate': 0.0045, 'min_temp': 5, 'max_it': 70000, 'mean_loss': 6.039, 'std_loss': 0.06, 'test_std': 0.139, 'mean_control': 0.725, 'mean_test': 0.613, 'n_runs': 5}]



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
