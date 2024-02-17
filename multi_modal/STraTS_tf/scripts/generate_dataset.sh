#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --job-name=STraTS_tf_data
#SBATCH --output=output/output_%x_%j.out 
#SBATCH --error=error/error_%x_%j.err

module load anaconda
source activate fyp_strats_37

cd /home/FYP/szhong005/fyp/multi_modal/STraTS_tf


# python -W ignore data.py  \
#                     --type "mortality" \
#                     --output_dir "./data_0_24_880" \
#                     --data_path "./mimic_iii_preprocessed.pkl" \
#                     --start_hour 0 \
#                     --end_hour 24 \
#                     --num_obs 880

# python -W ignore data.py  \
#                     --type "forecast" \
#                     --output_dir "./data_forecast" \
#                     --data_path "./mimic_iii_preprocessed.pkl" \
#                     --num_obs 880



python -W ignore data.py  \
                    --type "mortality" \
                    --output_dir "./new_data_physionet_mortality_0_24_470" \
                    --data_path "./physionet_2012_preprocessed.pkl" \
                    --start_hour 0 \
                    --end_hour 24 \
                    --num_obs 470 \
                    --dataset "physionet_2012" \

# python -W ignore data.py  \
#                     --type "forecast" \
#                     --output_dir "./data_physionet_forecast" \
#                     --data_path "./physionet_2012_preprocessed.pkl" \
#                     --num_obs 800 \
#                     --dataset "physionet_2012" \

python -W ignore data.py  \
                    --type "mtand" \
                    --output_dir "./new_data_physionet_mortality_0_24_470" \
                    --data_path "./new_data_physionet_mortality_0_48_1497" \
                    --start_hour 0 \
                    --end_hour 24 \
    


            