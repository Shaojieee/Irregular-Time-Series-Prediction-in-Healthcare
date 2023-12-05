#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --job-name=mimiciii_download
#SBATCH --output=output/output_%x_%j.out 
#SBATCH --error=error/error_%x_%j.err

cd /home/FYP/szhong005/fyp/multi_modal

wget -r -N -c -np --user szhong005 --password "Sj67430921^" https://physionet.org/files/mimiciii/1.4/


