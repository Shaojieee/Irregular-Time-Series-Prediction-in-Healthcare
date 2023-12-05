#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --job-name=fyp_mimic3_benchmark_clinic
#SBATCH --output=./output/output_%x_%j.out 
#SBATCH --error=./error/error_%x_%j.err

module load anaconda

source activate fyp_mimic3_benchmark

cd /home/FYP/szhong005/fyp/multi_modal/ClinicalNotesICU


python ./scripts/extract_T0.py