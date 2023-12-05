#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name=fyp_multimodal 
#SBATCH --output=output/output_%x_%j.out 
#SBATCH --error=error/error_%x_%j.err

module load anaconda
source activate fyp_multi_modal

cd /home/FYP/szhong005/fyp/multi_modal/MultimodalMIMIC

# for mixup_level in 'batch' 'batch_seq' 'batch_seq_feature'
for mixup_level in 'batch_seq_feature' 
do
python -W ignore main.py  --num_train_epochs 6  --modeltype 'TS_Text' \
                --kernel_size 1 --train_batch_size 4 --eval_batch_size 8   --seed 42 \
                --gradient_accumulation_steps 16  --num_update_bert_epochs 2 --bertcount 3 \
                --ts_learning_rate  0.0004 --txt_learning_rate 0.00002 \
                --notes_order 'Last' --num_of_notes 5 --max_length 1024 --layers 3\
                --output_dir "run/TS_Text" --embed_dim 128 \
                --model_name "bioLongformer"\
                --task 'ihm'\
                --file_path 'Data/ihm' \
                --num_labels 2 \
                --num_heads 8\
                --irregular_learn_emb_text\
                --embed_time 64\
                --tt_max 48\
                --TS_mixup\
                --mixup_level $mixup_level\
                --fp16 \
                --irregular_learn_emb_text \
                --irregular_learn_emb_ts \
                --reg_ts \
                # --debug
                # --tensorboard_dir "./tensorboard_dir" \
                
done

            