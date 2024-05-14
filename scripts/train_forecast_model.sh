#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --job-name=STraTS_tf_forecast
#SBATCH --output=output/output_%x_%j.out 
#SBATCH --error=error/error_%x_%j.err
#SBATCH --nodelist=SCSEGPU-TC1-06

module load anaconda
source activate fyp_strats_37

cd /home/FYP/szhong005/fyp/multi_modal/STraTS_tf


python -W ignore train_forecast.py  \
                    --output_dir "./logs_physionet/forecast_32_2_4_32_0_8_200_0.99_1.0" \
                    --data_dir "./data_physionet/data_physionet_forecast_0.99_1.0" \
                    --with_demo \
                    --len_time_query 200 \
                    --model_type "mtand_strats" \
                    --d_mtand 32 \
                    --N_mtand 0 \
                    --he_mtand 8 \
                    --dropout_mtand 0.2 \
                    --d_demo 32 \
                    --d_strats 32 \
                    --N_strats 2 \
                    --he_strats 4 \
                    --dropout_strats 0.2 \
                    --lr 0.0005 \
                    --patience 5 \
                    --batch_size 32 

# python -W ignore train.py  \
#                     --output_dir "./logs_physionet/forecast_32_2_4_32_0_8_200_0.99_1.0" \
#                     --data_dir "./data_physionet/data_physionet_0_48_0.99_1.0" \
#                     --model_weights "./logs_physionet/forecast_32_2_4_32_0_8_200_0.99_1.0/forecast_model.h5" \
#                     --with_demo \
#                     --len_time_query 200 \
#                     --max_time 48 \
#                     --model_type "mtand_strats" \
#                     --d_mtand 32 \
#                     --N_mtand 0 \
#                     --he_mtand 8 \
#                     --dropout_mtand 0.2 \
#                     --d_demo 32 \
#                     --d_strats 32 \
#                     --N_strats 2 \
#                     --he_strats 4 \
#                     --dropout_strats 0.2 \
#                     --lr 0.0005 \
#                     --patience 10 \
#                     --batch_size 32 \
#                     --lds 100 \
#                     --repeats 10


# python -W ignore train_forecast.py  \
#                     --output_dir "./logs_physionet/forecast_32_2_4_0.99_1.0" \
#                     --data_dir "./data_physionet_forecast_0.99_1.0" \
#                     --with_demo \
#                     --len_time_query 200 \
#                     --model_type "strats" \
#                     --d_mtand 32 \
#                     --N_mtand 0 \
#                     --he_mtand 8 \
#                     --dropout_mtand 0.2 \
#                     --d_demo 32 \
#                     --d_strats 32 \
#                     --N_strats 2 \
#                     --he_strats 4 \
#                     --dropout_strats 0.2 \
#                     --lr 0.0005 \
#                     --patience 5 \
#                     --batch_size 32 




            