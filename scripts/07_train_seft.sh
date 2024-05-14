python train_grud_seft.py  \
        --dataset "physionet" \
        --data_dir "./data_physionet/seft" \
        --output_dir "./logs_physionet/seft" \
        --model "SeFT" \
        --lr 0.0001 --batch_size 32 --patience 10 \
        --lds 100 --repeats 10



python train_grud_seft.py  \
        --dataset "mimic" \
        --data_dir "./data_mimic/seft" \
        --output_dir "./logs_mimic/seft" \
        --model "SeFT" \
        --lr 0.0001 --batch_size 32 --patience 10 \
        --lds 100 --repeats 10
