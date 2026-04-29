import argparse
import ast
import csv
import sys
from pathlib import Path


def grid_extract_POS(input_file):
    input_file = Path(input_file).expanduser()
    unique_rows = {}

    with input_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                records = ast.literal_eval(line)
            except Exception:
                continue

            if not isinstance(records, list):
                continue

            for record in records:
                key = (
                    record["learn_rate"],
                    record["min_temp"],
                    record["max_it"],
                )
                unique_rows[key] = record

    writer = csv.writer(sys.stdout)
    writer.writerow([
        "mean_control",
        "mean_test",
        "mean_loss",
        "mean_std_loss",
        "",
        "learning_rate",
        "min_temp",
        "max_iteration",
    ])

    for key in sorted(unique_rows):
        record = unique_rows[key]
        writer.writerow([
            record["mean_control"],
            record["mean_test"],
            record["mean_loss"],
            record["std_loss"],
            "",
            record["learn_rate"],
            record["min_temp"],
            record["max_it"],
        ])


def main():
    parser = argparse.ArgumentParser(
        description="Print deduplicated grid-search results as CSV."
    )
    parser.add_argument("input_file", help="SLURM output file to parse")
    args = parser.parse_args()
    grid_extract_POS(args.input_file)


if __name__ == "__main__":
    main()
