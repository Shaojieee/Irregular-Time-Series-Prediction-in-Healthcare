#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=fyp_strats_generate_dataset
#SBATCH --output=./output/output_%x_%j.out 
#SBATCH --error=./error/error_%x_%j.err

module load anaconda
source activate fyp_multi_modal

cd /home/FYP/szhong005/fyp/multi_modal/STraTS_torch

python generate_datasets_mimic_iii_benchmark.py