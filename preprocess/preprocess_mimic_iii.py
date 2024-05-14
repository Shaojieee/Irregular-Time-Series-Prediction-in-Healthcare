import argparse
import os
import importlib  


mimic3 = importlib.import_module("..mimic3-benchmarks.mimic3benchmark.scripts.create_in_hospital_mortality")

def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Change from get stay with at least 24 hours of data
    mimic3.process_partition(args, "test", n_hours=24.0)
    mimic3.process_partition(args, "train", n_hours=24.0)


if __name__ == '__main__':
    main()