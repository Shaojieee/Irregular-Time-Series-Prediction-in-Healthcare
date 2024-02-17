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


python -W ignore optuna_tune.py  \
                    --output_dir "./mtand_strats_tune" \
                    --data_dir "./new_data_physionet_mortality_0_48_800" \
                    --study_name "mtand_strats_tuning" \
                    --with_demo \
                    --max_len 800 \
                    --max_time 48 \
                    --d_demo 32 \
                    --d_strats 32 \
                    --N_strats 2 \
                    --he_strats 4 \
                    --dropout_strats 0.2 \
                    --patience 10 \
                    --batch_size 32 \












            