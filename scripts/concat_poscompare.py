import pandas as pd
import re
import os

#run from hte folder with the files using  python ~/Programming/Hannahs-CEBRAs/scripts/concat_poscompare.py

# List of CSV files
csv_files = [
    'pos_compare_lr0.00055_mt0.25_mi24000_dcosine_2024-06-07_15-31-48.csv',
    'pos_compare_lr0.00055_mt0.25_mi24000_dcosine_2024-06-07_15-49-17.csv',
    'pos_compare_lr0.00055_mt0.25_mi24000_dcosine_2024-06-07_15-47-19.csv',
    'pos_compare_lr0.00055_mt0.25_mi24000_dcosine_2024-06-07_15-32-08.csv'
    #ex
    #'path/to/csv_files/pos_compare_lr0.00775_mt0.25_mi15000_deuclidean_2024-05-13_20-11-37.csv',
    #'path/to/csv_files/pos_compare_lr0.00775_mt0.25_mi15000_deuclidean_2024-05-15_15-37-10.csv'
    # Add more file paths as needed
]

# Extract the common prefix using regex
pattern = r'^(.*?_lr[\d\.e-]+_mt[\d\.e-]+_mi\d+_d\w+)_2024.*\.csv'
common_prefix = None
for file in csv_files:
    match = re.match(pattern, os.path.basename(file))
    if match:
        common_prefix = match.group(1)
        break

if common_prefix is None:
    raise ValueError("No files matched the expected pattern")

# Read and concatenate all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
concatenated_df = pd.concat(df_list, ignore_index=True)

# Construct the output file name
output_dir = os.path.dirname(csv_files[0])
output_file = os.path.join(output_dir, f'{common_prefix}_CONCAT.csv')
concatenated_df.to_csv(output_file, index=False)

print(f'Concatenated CSV file saved as: {output_file}')
