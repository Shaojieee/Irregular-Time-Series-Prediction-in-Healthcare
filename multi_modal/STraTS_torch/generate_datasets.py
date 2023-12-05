from data import generate_forecast_dataset, generate_mortality_dataset
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data_path", type=str, default='./mimic_iii_preprocessed.pkl'
    )
    parser.add_argument(
        "--forecast_output_dir", type=str,default='./forecast_datasets', help="forecasting_model or mortality_model"
    )
    parser.add_argument(
        "--mortality_output_dir", type=str,default='./mortality_datasets', help="forecasting_model or mortality_model"
    )

    args = parser.parse_args()

    print('Generating forecast dataset')
    generate_forecast_dataset(args.data_path, args.forecast_output_dir)
    print('Generating mortality dataset')
    generate_mortality_dataset(args.data_path, args.mortality_output_dir)