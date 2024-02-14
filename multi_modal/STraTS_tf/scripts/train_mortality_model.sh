#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=STraTS_tf_mortality
#SBATCH --output=output/output_%x_%j.out 
#SBATCH --error=error/error_%x_%j.err

module load anaconda
source activate fyp_strats_37

cd /home/FYP/szhong005/fyp/multi_modal/STraTS_tf


# python -W ignore train.py  \
#                     --output_dir "./logs/forecast_mtand_ffn_tf_fv_tfv_50_2_4" \
#                     --model_weights "./logs/forecast_mtand_ffn_tf_fv_tfv_50_2_4/forecast_model.h5" \
#                     --data_dir "./data" \
#                     --max_len 880 \
#                     --with_demo \
#                     --model_type "custom" \
#                     --d 50 \
#                     --N 1 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --lr 0.0005 \
#                     --patience 10 \
#                     --batch_size 32 \
#                     --lds 100\
#                     --repeats 10

# python -W ignore train.py  \
#                     --output_dir "./logs_physionet/32_2_4_36_48_220" \
#                     --data_dir "./data_physionet_mortality_36_48_220" \
#                     --with_demo \
#                     --max_len 220 \
#                     --d 32 \
#                     --N 2 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --lr 0.001 \
#                     --patience 10 \
#                     --batch_size 32 \
#                     --lds 10,20,40 \
#                     --repeats 10


# python -W ignore train.py  \
#                     --output_dir "./logs_physionet/32_2_4_imputed_800_all_2" \
#                     --data_dir "./data_all_imputed_physionet_mortality" \
#                     --with_demo \
#                     --max_len 800 \
#                     --model_type "imputed" \
#                     --d 32 \
#                     --N 2 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --lr 0.001 \
#                     --patience 10 \
#                     --batch_size 32 \
#                     --lds 50 \
#                     --repeats 10



python -W ignore train.py  \
                    --output_dir "./logs_physionet/32_2_4_32_0_8_mtand_strats_200_mask_negative_2" \
                    --data_dir "./data_physionet_mortality_0_48_800" \
                    --with_demo \
                    --max_len 800 \
                    --len_time_query 200 \
                    --model_type "mtand_strats" \
                    --d_mtand 32 \
                    --d_demo 32 \
                    --d 32 \
                    --N 2 \
                    --he 4 \
                    --dropout 0.2 \
                    --lr 0.0005 \
                    --patience 10 \
                    --batch_size 32 \
                    --lds 50 \
                    --repeats 10



# python -W ignore train.py  \
#                     --output_dir "./logs_physionet/32_2_4" \
#                     --data_dir "./data_physionet_mortality_0_48_800" \
#                     --with_demo \
#                     --max_len 800 \
#                     --d 32 \
#                     --N 2 \
#                     --he 4 \
#                     --dropout 0.2 \
#                     --lr 0.001 \
#                     --patience 10 \
#                     --batch_size 32 \
#                     --lds 100 \
#                     --repeats 10












            