from data import generate_mortality_dataset_mimic_iii_benchmark
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data_path", type=str, default='../mimic3-benchmarks/data/in-hospital-mortality'
    )
    parser.add_argument(
        "--text_data_path", type=str, default='../mimic3-benchmarks/data/root'
    )
    parser.add_argument(
        "--text_start_time_path", type=str, default='../mimic3-benchmarks/data'
    )
    parser.add_argument(
        "--output_dir", type=str, default='./mortality_mimic_3_benchmark'
    )
    parser.add_argument(
        "--train_listfile", type=str, default='../mimic3-benchmarks/data/in-hospital-mortality/train_listfile.csv'
    )
    parser.add_argument(
        "--val_listfile", type=str, default='../mimic3-benchmarks/data/in-hospital-mortality/val_listfile.csv'
    )
    parser.add_argument(
        "--test_listfile", type=str, default='../mimic3-benchmarks/data/in-hospital-mortality/test_listfile.csv'
    )
    parser.add_argument(
        "--channel_file", type=str, default='../MultimodalMIMIC/Data/irregular/channel_info.json'
    )
    parser.add_argument(
        "--dis_config_file", type=str, default='../MultimodalMIMIC/Data/irregular/discretizer_config.json'
    )
    parser.add_argument(
        "--max_len", type=int, default=500
    )
    parser.add_argument(
        "--period_length", type=int, default=48
    )

    args = parser.parse_args()

    print('Generating forecast dataset')
    generate_mortality_dataset_mimic_iii_benchmark(
        data_path=args.data_path,
        text_data_path=args.text_data_path,
        text_start_time_path=args.text_start_time_path,
        output_dir=args.output_dir, 
        train_listfile=args.train_listfile, 
        val_listfile=args.val_listfile, 
        test_listfile=args.test_listfile, 
        channel_file=args.channel_file,
        dis_config_file=args.dis_config_file,
        max_len=args.max_len,
        period_length=float(args.period_length)
    )