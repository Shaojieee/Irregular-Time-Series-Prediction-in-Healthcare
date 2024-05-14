python -W ignore data.py  \
                        --dataset "mimic" \
                        --type "mortality" \
                        --output_dir "./data_mimic/strats_mtand" \
                        --data_path "./data_mimic" 



python -W ignore data.py  \
                        --dataset "mimic" \
                        --type "mtand" \
                        --output_dir "./data_mimic/strats_mtand" \
                        --data_path "./data_mimic/strats_mtand" \
                        --percentile_timestep 100



python -W ignore data.py  \
                        --dataset "physionet_2012" \
                        --type "mortality" \
                        --output_dir "./data_physionet/strats_mtand" \
                        --data_path "./data_physionet/physionet_2012_preprocessed.pkl" \
                        --start_hour 0 \
                        --end_hour 48 \
                        --num_obs 0.99 \
                    

python -W ignore data.py  \
                        --dataset "physionet_2012" \
                        --type "mtand" \
                        --output_dir "./data_physionet/strats_mtand" \
                        --data_path "./data_physionet/strats_mtand" \
                        --percentile_timestep 100








    


            