#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=STraTS_tf_forecast
#SBATCH --output=output/output_%x_%j.out 
#SBATCH --error=error/error_%x_%j.err
#SBATCH --nodelist=SCSEGPU-TC1-06

module load anaconda
source activate fyp_strats_37

cd /home/FYP/szhong005/fyp/multi_modal/STraTS_tf


# python -W ignore train_forecast.py  \
#                     --output_dir "./logs/forecast_special_48_0_6" \
#                     --data_dir "./data_forecast" \
#                     --save_path "forecast_model.h5" \
#                     --custom_strats \
#                     --special_transformer \
#                     --with_demo \
#                     --max_len 880 \
#                     --d 48 \
#                     --N 0 \
#                     --he 6 \
#                     --dropout 0.2 \
#                     --lr 0.0005 \
#                     --patience 5 \
#                     --batch_size 32 \
#                     --samples_per_epoch 102400 


python -W ignore train.py  \
                    --output_dir "./logs/forecast_special_48_0_6" \
                    --model_weights "./logs/forecast_special_48_0_6/forecast_model.h5" \
                    --data_dir "./data" \
                    --max_len 880 \
                    --custom_strats \
                    --with_demo \
                    --special_transformer \
                    --d 48 \
                    --N 0 \
                    --he 6 \
                    --dropout 0.2 \
                    --lr 0.0005 \
                    --patience 10 \
                    --batch_size 32 \
                    --lds 30,40,60 \
                    --repeats 10


# python -W ignore train_forecast.py  \
#                     --output_dir "./logs/forecast_50_2_4" \
#                     --data_dir "./data_forecast" \
#                     --save_path "forecast_model.h5" \
#                     --max_len 880 \
#                     --with_demo \
#                     --d 50 \
#                     --N 2 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --lr 0.0005 \
#                     --patience 5 \
#                     --batch_size 32 \
#                     --samples_per_epoch 102400 

# python -W ignore train.py  \
#                     --output_dir "./logs/forecast_50_2_4" \
#                     --model_weights "./logs/forecast_50_2_4/forecast_model.h5" \
#                     --data_dir "./data" \
#                     --max_len 880 \
#                     --with_demo \
#                     --d 50 \
#                     --N 2 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --lr 0.0005 \
#                     --patience 10 \
#                     --batch_size 32 \
#                     --lds 50 \
#                     --repeats 10



            