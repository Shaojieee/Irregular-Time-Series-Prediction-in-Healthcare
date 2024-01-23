#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --job-name=fyp_STraTS_torch_forcast
#SBATCH --output=output/output_%x_%j.out 
#SBATCH --error=error/error_%x_%j.err
#SBATCH --nodelist=SCSEGPU-TC1-06

module load anaconda
source activate fyp_multi_modal

cd /home/FYP/szhong005/fyp/multi_modal/STraTS_torch


python -W ignore train.py  \
                    --train_job "forecasting_model" \
                    --output_dir "./logs_new/strats_forecast" \
                    --data_dir "./forecast_datasets" \
                    --d 50 \
                    --N 2 \
                    --he 4 \
                    --dropout 0.2 \
                    --ts_learning_rate 0.0005 \
                    --patience 5 \
                    --early_stopper_min_delta 0 \
                    --early_stopper_mode "min" \
                    --early_stopper_metric "LOSS" \
                    --early_stopper_restore_best_weights \
                    --train_batch_size 32 \
                    --eval_batch_size 32 \
                    --samples_per_epoch 256000 \
                    --num_epochs 100 \

            