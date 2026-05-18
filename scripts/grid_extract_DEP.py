# Sample dataset
data = [{'learn_rate': 0.006, 'min_temp': 3.5, 'max_it': 75000, 'mean_loss': 5.937, 'std_loss': 0.007, 'test_std': 0.148, 'mean_control': 0.776, 'mean_test': 0.622, 'n_runs': 5}, {'learn_rate': 0.006, 'min_temp': 3.5, 'max_it': 90000, 'mean_loss': 5.948, 'std_loss': 0.011, 'test_std': 0.119, 'mean_control': 0.808, 'mean_test': 0.591, 'n_runs': 5}, {'learn_rate': 0.006, 'min_temp': 3.5, 'max_it': 105000, 'mean_loss': 6.045, 'std_loss': 0.086, 'test_std': 0.153, 'mean_control': 0.632, 'mean_test': 0.451, 'n_runs': 2}, {'learn_rate': 0.008, 'min_temp': 2.5, 'max_it': 60000, 'mean_loss': 5.872, 'std_loss': 0.012, 'test_std': 0.11, 'mean_control': 0.789, 'mean_test': 0.57, 'n_runs': 5}, {'learn_rate': 0.008, 'min_temp': 2.5, 'max_it': 75000, 'mean_loss': 5.872, 'std_loss': 0.013, 'test_std': 0.1, 'mean_control': 0.749, 'mean_test': 0.566, 'n_runs': 5}, {'learn_rate': 0.008, 'min_temp': 2.5, 'max_it': 90000, 'mean_loss': 5.867, 'std_loss': 0.011, 'test_std': 0.117, 'mean_control': 0.802, 'mean_test': 0.488, 'n_runs': 5}, {'learn_rate': 0.008, 'min_temp': 2.5, 'max_it': 105000, 'mean_loss': 5.879, 'std_loss': 0.013, 'test_std': 0.08, 'mean_control': 0.818, 'mean_test': 0.639, 'n_runs': 5}, {'learn_rate': 0.008, 'min_temp': 2.75, 'max_it': 60000, 'mean_loss': 5.887, 'std_loss': 0.009, 'test_std': 0.036, 'mean_control': 0.792, 'mean_test': 0.614, 'n_runs': 5}, {'learn_rate': 0.008, 'min_temp': 2.75, 'max_it': 75000, 'mean_loss': 5.896, 'std_loss': 0.025, 'test_std': 0.08, 'mean_control': 0.806, 'mean_test': 0.626, 'n_runs': 5}, {'learn_rate': 0.008, 'min_temp': 2.75, 'max_it': 90000, 'mean_loss': 5.892, 'std_loss': 0.017, 'test_std': 0.106, 'mean_control': 0.802, 'mean_test': 0.593, 'n_runs': 5}, {'learn_rate': 0.008, 'min_temp': 2.75, 'max_it': 105000, 'mean_loss': 5.897, 'std_loss': 0.018, 'test_std': 0.092, 'mean_control': 0.798, 'mean_test': 0.634, 'n_runs': 5}, {'learn_rate': 0.008, 'min_temp': 3, 'max_it': 60000, 'mean_loss': 5.911, 'std_loss': 0.014, 'test_std': 0.102, 'mean_control': 0.672, 'mean_test': 0.528, 'n_runs': 5}, {'learn_rate': 0.008, 'min_temp': 3, 'max_it': 75000, 'mean_loss': 5.911, 'std_loss': 0.018, 'test_std': 0.083, 'mean_control': 0.827, 'mean_test': 0.578, 'n_runs': 5}, {'learn_rate': 0.008, 'min_temp': 3, 'max_it': 90000, 'mean_loss': 5.972, 'std_loss': 0.134, 'test_std': 0.037, 'mean_control': 0.741, 'mean_test': 0.614, 'n_runs': 5}, {'learn_rate': 0.008, 'min_temp': 3, 'max_it': 105000, 'mean_loss': 5.909, 'std_loss': 0.012, 'test_std': 0.167, 'mean_control': 0.805, 'mean_test': 0.588, 'n_runs': 5}, {'learn_rate': 0.008, 'min_temp': 3.5, 'max_it': 60000, 'mean_loss': 5.95, 'std_loss': 0.01, 'test_std': 0.131, 'mean_control': 0.811, 'mean_test': 0.516, 'n_runs': 5}, {'learn_rate': 0.008, 'min_temp': 3.5, 'max_it': 75000, 'mean_loss': 5.938, 'std_loss': 0.01, 'test_std': 0.113, 'mean_control': 0.818, 'mean_test': 0.416, 'n_runs': 5}]



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
