#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name=fyp_STraTS_torch_mortality
#SBATCH --output=output/output_%x_%j.out 
#SBATCH --error=error/error_%x_%j.err
#SBATCH --nodelist=SCSEGPU-TC1-01

module load anaconda
source activate fyp_multi_modal

cd /home/FYP/szhong005/fyp/multi_modal/STraTS_torch


# python -W ignore train.py  \
#                     --fp16 \
#                     --train_job "mortality_model" \
#                     --output_dir "./logs/20231020_1009" \
#                     --data_dir "./mortality_datasets" \
#                     --forecast_model_weights "./logs/20231020_1009/best_forecasting_model.pth"\
#                     --d 50 \
#                     --N 2 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --learning_rate 0.0005 \
#                     --patience 5 \
#                     --early_stopper_min_delta 0 \
#                     --early_stopper_mode "min" \
#                     --early_stopper_restore_best_weights \
#                     --train_batch_size 32 \
#                     --eval_batch_size 32 \
#                     --lds 10,20,30,40 \
#                     --repeats 5 \
#                     --num_epochs 100 


python -W ignore train.py  \
                    --fp16 \
                    --train_job "mortality_model" \
                    --output_dir "./logs/202312111400" \
                    --data_dir "./mortality_datasets" \
                    --d 50 \
                    --N 2 \
                    --he 4 \
                    --dropout 0.2 \
                    --learning_rate 0.0005 \
                    --patience 5 \
                    --early_stopper_min_delta 0 \
                    --early_stopper_mode "min" \
                    --early_stopper_restore_best_weights \
                    --train_batch_size 32 \
                    --eval_batch_size 32 \
                    --lds 100 \
                    --repeats 1 \
                    --num_epochs 100 


            