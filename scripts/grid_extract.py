# Sample dataset
data = [{'learn_rate': 0.005, 'min_temp': 0.15, 'max_it': 16000, 'KNN_err_train': 17.632956722556152, 'KNN_err_test': 20.67542964223728, 'train_r2': -0.9755021430986118, 'test_r2': -1.345075394684389, 'med_control': 33.15861060538854, 'mead_test': 37.47687193661421, 'shuff_minus_not': -4.299093623106514, 'shuf_med': 7.582311754306503}, {'learn_rate': 0.005, 'min_temp': 0.15, 'max_it': 18000, 'KNN_err_train': 17.032342885920038, 'KNN_err_test': 19.3144907590786, 'train_r2': -1.1037804112192242, 'test_r2': -1.1751599312870737, 'med_control': 35.14518579710632, 'mead_test': 36.39161046999419, 'shuff_minus_not': -2.9369125035532426, 'shuf_med': 6.361412207960093}]
#for cond decoding

'''
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


# Calling the function to print values
print_formatted_values(data)
