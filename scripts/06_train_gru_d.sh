python train_grud_seft.py  \
        --dataset "physionet" \
        --data_dir "./data_physionet/grud" \
        --output_dir "./logs_physionet/grud" \
        --model "GRUD" \
        --lr 0.0001 --batch_size 32 --patience 10 \
        --lds 100 --repeats 10



python train_grud_seft.py  \
        --dataset "mimic" \
        --data_dir "./data_mimic/grud" \
        --output_dir "./logs_mimic/grud" \
        --model "GRUD" \
        --lr 0.0001 --batch_size 32 --patience 10 \
        --lds 100 --repeats 10
