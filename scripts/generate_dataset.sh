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
#                         --type "forecast" \
#                         --output_dir "./data_physionet_forecast_0.99_1.0" \
#                         --data_path "./physionet_2012_preprocessed.pkl" \
#                         --num_obs 0.99 \
#                         --dataset "physionet_2012" 

# python -W ignore data.py  \
#                         --type "forecast" \
#                         --output_dir "./data_physionet_forecast_1.0" \
#                         --data_path "./physionet_2012_preprocessed.pkl" \
#                         --num_obs 1.0 \
#                         --dataset "physionet_2012" 

# python -W ignore data.py  \
#                         --type "mtand_forecast" \
#                         --output_dir "./data_physionet_forecast_0.99_1.0" \
#                         --data_path "./data_physionet_forecast_1.0" \
#                         --percentile_timestep 100 \
#                         --dataset "physionet_2012" 



# for t in 24
# do
#     python -W ignore data.py  \
#                         --type "mortality" \
#                         --output_dir "./data_mimic/data_mimic_0_${t}_0.99_0.99" \
#                         --data_path "./mimic_iii_48_preprocessed.pkl" \
#                         --start_hour 0 \
#                         --end_hour $t \
#                         --num_obs 0.99 \
#                         --dataset "mimic_iii" 

#     python -W ignore data.py  \
#                         --type "mortality" \
#                         --output_dir "./data_mimic/data_mimic_0_${t}_1.0" \
#                         --data_path "./mimic_iii_48_preprocessed.pkl" \
#                         --start_hour 0 \
#                         --end_hour $t \
#                         --num_obs 1.0 \
#                         --dataset "mimic_iii" 

#     python -W ignore data.py  \
#                         --type "mtand" \
#                         --output_dir "./data_mimic/data_mimic_0_${t}_0.99_0.99" \
#                         --data_path "./data_mimic/data_mimic_0_${t}_0.99_0.99" \
#                         --start_hour 0 \
#                         --end_hour $t \
#                         --percentile_timestep 99
# done


# python -W ignore data.py  \
#                     --type "mortality" \
#                     --output_dir "./data_physionet_12_24_&_36_48_0.99_1.0" \
#                     --data_path "./physionet_2012_preprocessed.pkl" \
#                     --start_hour 12,36 \
#                     --end_hour 24,48 \
#                     --num_obs 0.99 \
#                     --dataset "physionet_2012" 

# python -W ignore data.py  \
#                     --type "mortality" \
#                     --output_dir "./data_physionet_12_24_&_36_48_1.0" \
#                     --data_path "./physionet_2012_preprocessed.pkl" \
#                     --start_hour 12,36 \
#                     --end_hour 24,48 \
#                     --num_obs 1.0 \
#                     --dataset "physionet_2012" 

# python -W ignore data.py  \
#                     --type "mtand" \
#                     --output_dir "./data_physionet_12_24_&_36_48_0.99_1.0" \
#                     --data_path "./data_physionet_12_24_&_36_48_1.0" \
#                     --percentile_timestep 100


# python -W ignore data.py  \
#                     --type "mortality" \
#                     --output_dir "./data_physionet_0_48_0.99_srandom_0.2" \
#                     --data_path "./physionet_2012_preprocessed.pkl" \
#                     --start_hour 0 \
#                     --end_hour 48 \
#                     --num_obs 0.99 \
#                     --dataset "physionet_2012" 

# python -W ignore data.py  \
#                 --type "random" \
#                 --output_dir "./data_mimic/data_mimic_0_24_0.99_srandom_0.2" \
#                 --data_path "./data_mimic/data_mimic_0_24_0.99" \
#                 --random_drop 0.2 \

# python -W ignore data.py  \
#                 --type "mtand" \
#                 --output_dir "./data_mimic/data_mimic_0_24_0.99_srandom_0.2" \
#                 --data_path "./data_mimic/data_mimic_0_24_0.99_srandom_0.2" \
#                 --start_hour 0 \
#                 --end_hour 48 \
#                 --percentile_timestep 100 

# python -W ignore data.py  \
#                 --type "random" \
#                 --output_dir "./data_mimic/data_mimic_0_24_0.99_srandom_0.4" \
#                 --data_path "./data_mimic/data_mimic_0_24_0.99_srandom_0.2" \
#                 --random_drop 0.25 \

# python -W ignore data.py  \
#                 --type "mtand" \
#                 --output_dir "./data_mimic/data_mimic_0_24_0.99_srandom_0.4" \
#                 --data_path "./data_mimic/data_mimic_0_24_0.99_srandom_0.4" \
#                 --start_hour 0 \
#                 --end_hour 48 \
#                 --percentile_timestep 100 


# python -W ignore data.py  \
#                 --type "random" \
#                 --output_dir "./data_mimic/data_mimic_0_24_0.99_srandom_0.5" \
#                 --data_path "./data_mimic/data_mimic_0_24_0.99_srandom_0.4" \
#                 --random_drop 0.1666 \

# python -W ignore data.py  \
#                 --type "mtand" \
#                 --output_dir "./data_mimic/data_mimic_0_24_0.99_srandom_0.5" \
#                 --data_path "./data_mimic/data_mimic_0_24_0.99_srandom_0.5" \
#                 --start_hour 0 \
#                 --end_hour 48 \
#                 --percentile_timestep 100 

# python -W ignore data.py  \
#                 --type "random" \
#                 --output_dir "./data_mimic/data_mimic_0_24_0.99_srandom_0.6" \
#                 --data_path "./data_mimic/data_mimic_0_24_0.99_srandom_0.5" \
#                 --random_drop 0.2 \

# python -W ignore data.py  \
#                 --type "mtand" \
#                 --output_dir "./data_mimic/data_mimic_0_24_0.99_srandom_0.6" \
#                 --data_path "./data_mimic/data_mimic_0_24_0.99_srandom_0.6" \
#                 --start_hour 0 \
#                 --end_hour 48 \
#                 --percentile_timestep 100 