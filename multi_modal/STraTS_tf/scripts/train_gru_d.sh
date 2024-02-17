#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal

#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=gru_d
#SBATCH --output=output/output_%x_%j.out 
#SBATCH --error=error/error_%x_%j.err
#SBATCH --nodelist=SCSEGPU-TC1-06

module load anaconda
source activate fyp_strats_37

cd /home/FYP/szhong005/fyp/multi_modal/STraTS_tf



python gru_d.py  \
        --dataset "physionet_2012" \
        --data_dir "./data_grud" \
        --output_dir "./logs_physionet/grud" \
        --model "GRUD" \
        --lr 0.0001 --batch_size 32 --patience 10 \
        --lds 50 --repeats 10

python gru_d.py  \
        --dataset "physionet_2012" \
        --data_dir "./data_seft" \
        --output_dir "./logs_physionet/seft" \
        --model "SeFT" \
        --lr 0.00081 --batch_size 32 --patience 10 \
        --lds 50 --repeats 10