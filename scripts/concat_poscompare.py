import pandas as pd
import re
import os

#run from hte folder with the files using  python ~/Programming/Hannahs-CEBRAs/scripts/concat_poscompare.py

# List of CSV files
csv_files = [
    'cond_lr0.004_mt3_mi9000_dcosine_modeconstant_2024-07-25_11-34-12.csv',
    'cond_lr0.004_mt3_mi9000_dcosine_modeconstant_2024-07-25_15-19-37.csv',
    'cond_lr0.004_mt3_mi9000_dcosine_modeconstant_2024-07-25_15-00-07.csv',
    'cond_lr0.004_mt3_mi9000_dcosine_modeconstant_2024-07-25_14-55-18.csv'

    #'cond_lr0.008_mt3.33_mi6500_dcosine_modeconstant_2024-07-24_06-40-53.csv'
    #'cond_lr0.008_mt3_mi8000_dcosine_modeconstant_2024-07-24_07-07-09.csv'
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
