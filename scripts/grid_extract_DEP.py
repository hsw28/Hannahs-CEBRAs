# Sample dataset
data = [{'learn_rate': 0.0005, 'min_temp': 3.5, 'max_it': 50000, 'mean_loss': 5.931, 'std_loss': 0.0, 'test_std': 0.0, 'mean_control': 0.728, 'mean_test': 0.29, 'n_runs': 1}, {'learn_rate': 0.0005, 'min_temp': 3.5, 'max_it': 70000, 'mean_loss': 5.941, 'std_loss': 0.0, 'test_std': 0.0, 'mean_control': 0.768, 'mean_test': 0.268, 'n_runs': 1}, {'learn_rate': 0.0005, 'min_temp': 4, 'max_it': 50000, 'mean_loss': 5.947, 'std_loss': 0.0, 'test_std': 0.0, 'mean_control': 0.784, 'mean_test': 0.282, 'n_runs': 1}, {'learn_rate': 0.0005, 'min_temp': 4, 'max_it': 70000, 'mean_loss': 5.971, 'std_loss': 0.0, 'test_std': 0.0, 'mean_control': 0.752, 'mean_test': 0.272, 'n_runs': 1}, {'learn_rate': 0.0005, 'min_temp': 5, 'max_it': 50000, 'mean_loss': 6.011, 'std_loss': 0.0, 'test_std': 0.0, 'mean_control': 0.808, 'mean_test': 0.274, 'n_runs': 1}, {'learn_rate': 0.0005, 'min_temp': 5, 'max_it': 70000, 'mean_loss': 6.004, 'std_loss': 0.0, 'test_std': 0.0, 'mean_control': 0.88, 'mean_test': 0.274, 'n_runs': 1}, {'learn_rate': 0.001, 'min_temp': 3.5, 'max_it': 50000, 'mean_loss': 5.943, 'std_loss': 0.008, 'test_std': 0.031, 'mean_control': 0.8, 'mean_test': 0.321, 'n_runs': 3}, {'learn_rate': 0.001, 'min_temp': 3.5, 'max_it': 70000, 'mean_loss': 5.943, 'std_loss': 0.015, 'test_std': 0.076, 'mean_control': 0.845, 'mean_test': 0.417, 'n_runs': 5}, {'learn_rate': 0.001, 'min_temp': 4, 'max_it': 50000, 'mean_loss': 5.965, 'std_loss': 0.003, 'test_std': 0.229, 'mean_control': 0.836, 'mean_test': 0.519, 'n_runs': 2}, {'learn_rate': 0.001, 'min_temp': 4, 'max_it': 70000, 'mean_loss': 5.957, 'std_loss': 0.015, 'test_std': 0.06, 'mean_control': 0.824, 'mean_test': 0.405, 'n_runs': 5}, {'learn_rate': 0.001, 'min_temp': 5, 'max_it': 50000, 'mean_loss': 6.009, 'std_loss': 0.009, 'test_std': 0.056, 'mean_control': 0.808, 'mean_test': 0.352, 'n_runs': 5}, {'learn_rate': 0.001, 'min_temp': 5, 'max_it': 70000, 'mean_loss': 6.017, 'std_loss': 0.007, 'test_std': 0.084, 'mean_control': 0.832, 'mean_test': 0.371, 'n_runs': 3}, {'learn_rate': 0.0025, 'min_temp': 3.5, 'max_it': 50000, 'mean_loss': 5.974, 'std_loss': 0.08, 'test_std': 0.071, 'mean_control': 0.746, 'mean_test': 0.634, 'n_runs': 5}, {'learn_rate': 0.0025, 'min_temp': 3.5, 'max_it': 70000, 'mean_loss': 5.945, 'std_loss': 0.018, 'test_std': 0.075, 'mean_control': 0.805, 'mean_test': 0.589, 'n_runs': 5}, {'learn_rate': 0.0025, 'min_temp': 4, 'max_it': 50000, 'mean_loss': 5.995, 'std_loss': 0.068, 'test_std': 0.035, 'mean_control': 0.685, 'mean_test': 0.576, 'n_runs': 5}, {'learn_rate': 0.0025, 'min_temp': 4, 'max_it': 70000, 'mean_loss': 5.961, 'std_loss': 0.009, 'test_std': 0.061, 'mean_control': 0.779, 'mean_test': 0.66, 'n_runs': 5}, {'learn_rate': 0.0025, 'min_temp': 5, 'max_it': 50000, 'mean_loss': 6.006, 'std_loss': 0.011, 'test_std': 0.048, 'mean_control': 0.746, 'mean_test': 0.586, 'n_runs': 5}, {'learn_rate': 0.0025, 'min_temp': 5, 'max_it': 70000, 'mean_loss': 6.011, 'std_loss': 0.014, 'test_std': 0.1, 'mean_control': 0.806, 'mean_test': 0.65, 'n_runs': 5}]



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
