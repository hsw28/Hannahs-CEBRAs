# Sample dataset
data = [{'learn_rate': 0.0005, 'min_temp': 3, 'max_it': 50000, 'mean_loss': 6.05, 'std_loss': 0.0, 'test_std': 0.0, 'mean_control': 0.72, 'mean_test': 0.286, 'n_runs': 1}, {'learn_rate': 0.0005, 'min_temp': 3, 'max_it': 70000, 'mean_loss': 6.035, 'std_loss': 0.0, 'test_std': 0.0, 'mean_control': 0.736, 'mean_test': 0.284, 'n_runs': 1}, {'learn_rate': 0.0005, 'min_temp': 4, 'max_it': 50000, 'mean_loss': 6.087, 'std_loss': 0.011, 'test_std': 0.073, 'mean_control': 0.749, 'mean_test': 0.335, 'n_runs': 3}, {'learn_rate': 0.0005, 'min_temp': 4, 'max_it': 70000, 'mean_loss': 6.098, 'std_loss': 0.014, 'test_std': 0.043, 'mean_control': 0.726, 'mean_test': 0.338, 'n_runs': 4}, {'learn_rate': 0.0005, 'min_temp': 5, 'max_it': 50000, 'mean_loss': 6.131, 'std_loss': 0.0, 'test_std': 0.0, 'mean_control': 0.696, 'mean_test': 0.296, 'n_runs': 1}, {'learn_rate': 0.0005, 'min_temp': 5, 'max_it': 70000, 'mean_loss': 6.123, 'std_loss': 0.0, 'test_std': 0.0, 'mean_control': 0.8, 'mean_test': 0.27, 'n_runs': 1}, {'learn_rate': 0.001, 'min_temp': 3, 'max_it': 50000, 'mean_loss': 6.049, 'std_loss': 0.01, 'test_std': 0.084, 'mean_control': 0.818, 'mean_test': 0.428, 'n_runs': 4}, {'learn_rate': 0.001, 'min_temp': 3, 'max_it': 70000, 'mean_loss': 6.04, 'std_loss': 0.006, 'test_std': 0.049, 'mean_control': 0.806, 'mean_test': 0.422, 'n_runs': 5}, {'learn_rate': 0.001, 'min_temp': 4, 'max_it': 50000, 'mean_loss': 6.088, 'std_loss': 0.0, 'test_std': 0.0, 'mean_control': 0.768, 'mean_test': 0.288, 'n_runs': 1}, {'learn_rate': 0.001, 'min_temp': 4, 'max_it': 70000, 'mean_loss': 6.077, 'std_loss': 0.0, 'test_std': 0.0, 'mean_control': 0.824, 'mean_test': 0.27, 'n_runs': 1}, {'learn_rate': 0.001, 'min_temp': 5, 'max_it': 50000, 'mean_loss': 6.131, 'std_loss': 0.0, 'test_std': 0.0, 'mean_control': 0.832, 'mean_test': 0.294, 'n_runs': 1}, {'learn_rate': 0.001, 'min_temp': 5, 'max_it': 70000, 'mean_loss': 6.109, 'std_loss': 0.009, 'test_std': 0.088, 'mean_control': 0.789, 'mean_test': 0.374, 'n_runs': 5}, {'learn_rate': 0.0015, 'min_temp': 3, 'max_it': 50000, 'mean_loss': 6.068, 'std_loss': 0.053, 'test_std': 0.057, 'mean_control': 0.688, 'mean_test': 0.624, 'n_runs': 5}, {'learn_rate': 0.0015, 'min_temp': 3, 'max_it': 70000, 'mean_loss': 6.038, 'std_loss': 0.01, 'test_std': 0.164, 'mean_control': 0.768, 'mean_test': 0.506, 'n_runs': 5}, {'learn_rate': 0.0015, 'min_temp': 4, 'max_it': 50000, 'mean_loss': 6.081, 'std_loss': 0.007, 'test_std': 0.169, 'mean_control': 0.78, 'mean_test': 0.471, 'n_runs': 4}, {'learn_rate': 0.0015, 'min_temp': 4, 'max_it': 70000, 'mean_loss': 6.084, 'std_loss': 0.008, 'test_std': 0.146, 'mean_control': 0.802, 'mean_test': 0.474, 'n_runs': 5}, {'learn_rate': 0.0015, 'min_temp': 5, 'max_it': 50000, 'mean_loss': 6.109, 'std_loss': 0.005, 'test_std': 0.076, 'mean_control': 0.758, 'mean_test': 0.564, 'n_runs': 5}, {'learn_rate': 0.0015, 'min_temp': 5, 'max_it': 70000, 'mean_loss': 6.107, 'std_loss': 0.001, 'test_std': 0.168, 'mean_control': 0.757, 'mean_test': 0.577, 'n_runs': 5}] 

~

#for cond decoding


def print_formatted_values(data):
    for entry in data:
        mean_control = round(entry['mean_control'], 2)
        mean_test = round(entry['mean_test'], 2)
        mean_loss = round(entry['mean_loss'], 2)
        std_loss = round(entry['std_loss'],2)
        print(f"{mean_control}, {mean_test}, {mean_loss}, {std_loss}")

    for entry in data:
        learn_rate = entry['learn_rate']
        min_temp = entry['min_temp']
        max_it = entry['max_it']
        print(f"{learn_rate}, {min_temp}, {max_it}")

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
