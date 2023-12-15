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
#                     --output_dir "./logs/20231211_1200" \
#                     --data_dir "./mortality_mimic_3_benchmark" \
#                     --d 64 \
#                     --N 4 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --ts_learning_rate 0.0004 \
#                     --patience 10 \
#                     --early_stopper_min_delta 0 \
#                     --early_stopper_mode "min" \
#                     --early_stopper_restore_best_weights \
#                     --train_batch_size 4 \
#                     --eval_batch_size 4 \
#                     --lds 100 \
#                     --repeats 1 \
#                     --num_epochs 100 \
                

python -W ignore train.py  \
                    --fp16 \
                    --train_job "mortality_model" \
                    --output_dir "./logs/strats_with_text" \
                    --data_dir "./mortality_mimic_3_benchmark" \
                    --with_text \
                    --d 64 \
                    --N 4 \
                    --he 4 \
                    --dropout 0.2 \
                    --text_num_notes 5 \
                    --text_atten_embed_dim 128 \
                    --text_time_embed_dim 64 \
                    --period_length 48 \
                    --text_encoder_model "bioLongformer" \
                    --num_cross_layers 3 \
                    --num_cross_heads 8 \
                    --cross_dropout 0.10 \
                    --ts_learning_rate 0.0004 \
                    --text_learning_rate 0.00002 \
                    --patience 10 \
                    --early_stopper_min_delta 0 \
                    --early_stopper_mode "min" \
                    --early_stopper_restore_best_weights \
                    --train_batch_size 4 \
                    --eval_batch_size 4 \
                    --lds 100 \
                    --repeats 1 \
                    --num_epochs 100 \


            