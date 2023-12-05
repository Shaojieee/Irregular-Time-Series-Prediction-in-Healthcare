#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=fyp_mimic3_benchmark 
#SBATCH --output=./output/output_%x_%j.out 
#SBATCH --error=./error/error_%x_%j.err

module load anaconda
# module load python/3.7.13
source activate fyp_mimic3_benchmark

cd /home/FYP/szhong005/fyp/multi_modal/mimic3-benchmarks
pwd

echo 'Extracting Subjects'
python -m mimic3benchmark.scripts.extract_subjects ../physionet.org/files/mimiciii/1.4/ data/root/

echo 'Validating Events'
python -m mimic3benchmark.scripts.validate_events data/root/

echo 'Extracting Episodes from subjects'
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/

echo 'Train Test Split'
python -m mimic3benchmark.scripts.split_train_and_test data/root/

echo 'Creating different task'
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
python -m mimic3benchmark.scripts.create_decompensation data/root/ data/decompensation/
python -m mimic3benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/
python -m mimic3benchmark.scripts.create_phenotyping data/root/ data/phenotyping/
python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/

echo 'Train Val Split'
python -m mimic3models.split_train_val data/in-hospital-mortality/
python -m mimic3models.split_train_val data/decompensation/
python -m mimic3models.split_train_val data/length-of-stay/
python -m mimic3models.split_train_val data/phenotyping/
python -m mimic3models.split_train_val data/multitask/