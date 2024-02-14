#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --job-name=ssh_tunnel
#SBATCH --output=output/output_%x_%j.out 
#SBATCH --error=error/error_%x_%j.err


module load anaconda
source activate fyp_strats_37

cd /home/FYP/szhong005/fyp/multi_modal/STraTS_tf

jupyter lab --ip=$(hostname -i) --port=8886