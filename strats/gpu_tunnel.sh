#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --job-name=ssh_tunnel
#SBATCH --output=output/output_%x_%j.out 
#SBATCH --error=error/error_%x_%j.err


module load anaconda
source activate fyp_strats_37

jupyter lab --ip=$(hostname -i) --port=8886