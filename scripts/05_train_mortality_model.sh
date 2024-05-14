# Change the data_dir to the appropriate dataset


# Train strats
python -W ignore train.py  \
                    --output_dir "./logs_mimic/strats" \
                    --data_dir "./data_physionet/strats_mtand" \
                    --model_type "strats" \
                    --with_demo \
                    --d_demo 32 \
                    --d_strats 32 \
                    --N_strats 2 \
                    --he_strats 4 \
                    --dropout_strats 0.2 \
                    --lr 0.0005 \
                    --patience 10 \
                    --batch_size 32 \
                    --lds 100 \
                    --repeats 10

# Train mtand
python -W ignore train.py  \
                    --output_dir "./logs_mimic/mtand" \
                    --data_dir "./data_physionet/strats_mtand" \
                    --len_time_query 100 \
                    --model_type "mtand" \
                    --max_time 24 \
                    --with_demo \
                    --d_mtand 32 \
                    --N_mtand 0 \
                    --he_mtand 8 \
                    --dropout_mtand 0.2 \
                    --lr 0.0005 \
                    --patience 10 \
                    --batch_size 32 \
                    --lds 100 \
                    --repeats 10

# Train strats_mtand
python -W ignore train.py  \
                    --output_dir "./logs_physionet/strats_mtand" \
                    --data_dir "./data_physionet/strats_mtand" \
                    --with_demo \
                    --len_time_query 200 \
                    --max_time 48 \
                    --model_type "mtand_strats" \
                    --d_mtand 32 \
                    --N_mtand 0 \
                    --he_mtand 8 \
                    --dropout_mtand 0.2 \
                    --d_demo 32 \
                    --d_strats 32 \
                    --N_strats 2 \
                    --he_strats 4 \
                    --dropout_strats 0.2 \
                    --lr 0.0005 \
                    --patience 10 \
                    --batch_size 32 \
                    --lds 100 \
                    --repeats 10











            