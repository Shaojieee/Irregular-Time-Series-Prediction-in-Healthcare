#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --job-name=fyp_STraTS_torch_mortality
#SBATCH --output=output/output_%x_%j.out 
#SBATCH --error=error/error_%x_%j.err
#SBATCH --nodelist=SCSEGPU-TC1-06

module load anaconda
source activate fyp_multi_modal

cd /home/FYP/szhong005/fyp/multi_modal/STraTS_torch

# STraTS tuning
# python -W ignore train.py  \
#                     --train_job "mortality_model_tuning" \
#                     --study_name "strats_tuned_orig_dataset" \
#                     --output_dir "./logs/strats_tuning_orig_dataset" \
#                     --optuna_sampler "./strats_tuned_orig_dataset_sampler.pkl" \
#                     --data_dir "./mortality_datasets" \
#                     --d 64 \
#                     --N 4 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --ts_learning_rate 0.0004 \
#                     --patience 10 \
#                     --early_stopper_min_delta 0 \
#                     --early_stopper_mode "min" \
#                     --early_stopper_restore_best_weights \
#                     --lds 100 \
#                     --repeats 1 \
#                     --num_epochs 50 \

# Original STraTS
# python -W ignore train.py  \
#                     --fp16 \
#                     --train_job "mortality_model" \
#                     --output_dir "./logs/strats_weighted_orig_dataset" \
#                     --data_dir "./mortality_datasets" \
#                     --weighted_class_weights \
#                     --d 64 \
#                     --N 4 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --ts_learning_rate 0.0002 \
#                     --patience 10 \
#                     --early_stopper_min_delta 0 \
#                     --early_stopper_mode "min" \
#                     --early_stopper_restore_best_weights \
#                     --train_batch_size 128 \
#                     --eval_batch_size 128 \
#                     --lds 100 \
#                     --repeats 5 \
#                     --num_epochs 100 \


# # Numerical with new value encoding
# python -W ignore train.py  \
#                     --fp16 \
#                     --train_job "mortality_model" \
#                     --output_dir "./logs/strats_tuned_new_encoding_orig_dataset" \
#                     --data_dir "./mortality_datasets" \
#                     --new_value_encoding \
#                     --d 64 \
#                     --N 4 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --ts_learning_rate 0.0002 \
#                     --patience 10 \
#                     --early_stopper_min_delta 0 \
#                     --early_stopper_mode "min" \
#                     --early_stopper_restore_best_weights \
#                     --train_batch_size 128 \
#                     --eval_batch_size 128 \
#                     --lds 100 \
#                     --repeats 1 \
#                     --num_epochs 100 \



# Numerical with new value encoding & normalise varis
# python -W ignore train.py  \
#                     --fp16 \
#                     --train_job "mortality_model" \
#                     --output_dir "./logs/strats_new_encoding_normalise_varis_orig_dataset" \
#                     --data_dir "./mortality_datasets" \
#                     --new_value_encoding \
#                     --normalise_varis \
#                     --d 64 \
#                     --N 4 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --ts_learning_rate 0.0002 \
#                     --patience 10 \
#                     --early_stopper_min_delta 0 \
#                     --early_stopper_mode "min" \
#                     --early_stopper_restore_best_weights \
#                     --train_batch_size 128 \
#                     --eval_batch_size 128 \
#                     --lds 100 \
#                     --repeats 5 \
#                     --num_epochs 100 \


# mTAND STraTS
# python -W ignore train.py  \
#                     --fp16 \
#                     --train_job "mortality_model" \
#                     --output_dir "./logs/strats_mtand_weighted_32_2_4_orig_dataset" \
#                     --data_dir "./mortality_datasets" \
#                     --normalise_time \
#                     --custom_strats \
#                     --d 32 \
#                     --N 2 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --ts_learning_rate 0.0002 \
#                     --patience 10 \
#                     --early_stopper_min_delta 0 \
#                     --early_stopper_mode "min" \
#                     --early_stopper_restore_best_weights \
#                     --train_batch_size 16 \
#                     --eval_batch_size 16 \
#                     --lds 100 \
#                     --repeats 3 \
#                     --num_epochs 100 \

python -W ignore train.py  \
                    --fp16 \
                    --train_job "mortality_model" \
                    --output_dir "./logs/strats_mtand_time2vec_32_1_8" \
                    --data_dir "./mortality_mimic_3_benchmark" \
                    --time_2_vec \
                    --custom_strats \
                    --d 32 \
                    --N 1 \
                    --he 8 \
                    --dropout 0.2 \
                    --ts_learning_rate 0.0002 \
                    --patience 10 \
                    --early_stopper_min_delta 0 \
                    --early_stopper_mode "min" \
                    --early_stopper_restore_best_weights \
                    --train_batch_size 16 \
                    --eval_batch_size 16 \
                    --lds 100 \
                    --repeats 3 \
                    --num_epochs 100 \



# STraTS with text
# python -W ignore train.py  \
#                     --fp16 \
#                     --train_job "mortality_model" \
#                     --output_dir "./logs/strats_with_text_1" \
#                     --data_dir "./mortality_mimic_3_benchmark" \
#                     --model_weights "./logs/strats_with_text/best_mortality_model_100_repeat_0.pth" \
#                     --with_text \
#                     --d 64 \
#                     --N 4 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --text_num_notes 3 \
#                     --period_length 48 \
#                     --text_encoder_model "bioLongformer" \
#                     --ts_learning_rate 0.0004 \
#                     --text_learning_rate 0.00002 \
#                     --patience 10 \
#                     --early_stopper_min_delta 0 \
#                     --early_stopper_mode "min" \
#                     --early_stopper_restore_best_weights \
#                     --train_batch_size 4 \
#                     --eval_batch_size 4 \
#                     --lds 100 \
#                     --repeats 1 \
#                     --num_epochs 100 \


            