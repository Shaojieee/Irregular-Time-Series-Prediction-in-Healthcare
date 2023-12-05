#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=fyp_strats_preprocess 
#SBATCH --output=./output/output_%x_%j.out 
#SBATCH --error=./error/error_%x_%j.err

module load anaconda
source activate fyp_strats

cd /home/FYP/szhong005/fyp/strats

python ./STraTS/preprocess/preprocess_mimic_iii.py