#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=2
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
python -W ignore train.py  \
                    --train_job "mortality_model" \
                    --output_dir "./logs_new/strats" \
                    --data_dir "./mortality_datasets" \
                    --with_demographics \
                    --weighted_class_weights \
                    --d 32 \
                    --N 2 \
                    --he 4 \
                    --dropout 0.2 \
                    --ts_learning_rate 0.0005 \
                    --patience 10 \
                    --early_stopper_min_delta 0 \
                    --early_stopper_mode "min" \
                    --early_stopper_metric "LOSS" \
                    --early_stopper_restore_best_weights \
                    --train_batch_size 32 \
                    --eval_batch_size 32 \
                    --lds 50 \
                    --repeats 5 \
                    --num_epochs 100 \


# mTAND STraTS
# python -W ignore train.py  \
#                     --train_job "mortality_model" \
#                     --output_dir "./logs_new/mtand_ft_ft_ftv_32_2_4" \
#                     --data_dir "./mortality_datasets" \
#                     --custom_strats \
#                     --with_demographics \
#                     --d 32 \
#                     --N 2 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --ts_learning_rate 0.0005 \
#                     --gradient_accumulation_steps 1 \
#                     --patience 10 \
#                     --early_stopper_min_delta 0 \
#                     --early_stopper_mode "max" \
#                     --early_stopper_metric "SUM_PR_AUC_ROC_AUC" \
#                     --early_stopper_restore_best_weights \
#                     --train_batch_size 32 \
#                     --eval_batch_size 32 \
#                     --lds 50 \
#                     --repeats 5 \
#                     --num_epochs 100 \





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


            