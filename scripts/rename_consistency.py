import os
import re
from datetime import datetime, timedelta

# Directory containing the files
directory = "/Users/Hannah/Programming/data_eyeblink/rat0222/cebra_variables"

# Function to extract datetime from filename using regex
def parse_datetime_from_filename(filename):
    # Regex pattern to find date and time in the format "YYYY-MM-DD_HH-MM-SS"
    pattern = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}')
    match = pattern.search(filename)
    if match:
        return datetime.strptime(match.group(), "%Y-%m-%d_%H-%M-%S")
    else:
        raise ValueError("No valid date found in filename")

# Dictionary to hold datetime objects and corresponding filenames
timestamps = {}

# First pass to collect all CSV filenames and their corresponding timestamps
for filename in os.listdir(directory):
    if "consistency" in filename and filename.endswith(".csv"):
        try:
            dt = parse_datetime_from_filename(filename)
            timestamps[dt] = filename.replace('.csv', '')
        except ValueError as e:
            print(f"Failed to parse date from {filename}: {e}")

# Second pass to rename SVG files based on close timestamp in CSV
for filename in os.listdir(directory):
    if "consistency" in filename and filename.endswith(".svg"):
        try:
            svg_dt = parse_datetime_from_filename(filename)
            # Check for nearby timestamps within one second
            for csv_dt in timestamps:
                if abs((csv_dt - svg_dt).total_seconds()) <= 1:
                    # New filename based on corresponding CSV but with SVG extension
                    new_filename = timestamps[csv_dt] + '.svg'
                    old_path = os.path.join(directory, filename)
                    new_path = os.path.join(directory, new_filename)
                    # Rename the file
                    os.rename(old_path, new_path)
                    print(f"Renamed {filename} to {new_filename}")
                    break
        except ValueError as e:
            print(f"Failed to parse date from {filename}: {e}")
