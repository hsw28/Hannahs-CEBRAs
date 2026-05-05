# Sample dataset
data = [{'learn_rate': 0.0025, 'min_temp': 1.5, 'max_it': 13000, 'mean_loss': 6.068, 'std_loss': 0.006, 'mean_control': 0.58, 'mean_test': 0.599}]
[{'learn_rate': 0.0025, 'min_temp': 1.5, 'max_it': 13000, 'mean_loss': 6.068, 'std_loss': 0.006, 'mean_control': 0.58, 'mean_test': 0.599}, {'learn_rate': 0.0025, 'min_temp': 1.5, 'max_it': 17000, 'mean_loss': 5.781, 'std_loss': 0.012, 'mean_control': 0.8, 'mean_test': 0.483}]
[{'learn_rate': 0.0025, 'min_temp': 1.5, 'max_it': 13000, 'mean_loss': 6.068, 'std_loss': 0.006, 'mean_control': 0.58, 'mean_test': 0.599}, {'learn_rate': 0.0025, 'min_temp': 1.5, 'max_it': 17000, 'mean_loss': 5.781, 'std_loss': 0.012, 'mean_control': 0.8, 'mean_test': 0.483}, {'learn_rate': 0.0025, 'min_temp': 1.5, 'max_it': 21000, 'mean_loss': 5.761, 'std_loss': 0.01, 'mean_control': 0.792, 'mean_test': 0.401}]
[{'learn_rate': 0.0025, 'min_temp': 1.5, 'max_it': 13000, 'mean_loss': 6.068, 'std_loss': 0.006, 'mean_control': 0.58, 'mean_test': 0.599}, {'learn_rate': 0.0025, 'min_temp': 1.5, 'max_it': 17000, 'mean_loss': 5.781, 'std_loss': 0.012, 'mean_control': 0.8, 'mean_test': 0.483}, {'learn_rate': 0.0025, 'min_temp': 1.5, 'max_it': 21000, 'mean_loss': 5.761, 'std_loss': 0.01, 'mean_control': 0.792, 'mean_test': 0.401}, {'learn_rate': 0.0025, 'min_temp': 1.5, 'max_it': 25000, 'mean_loss': 5.758, 'std_loss': 0.016, 'mean_control': 0.824, 'mean_test': 0.546}]


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
