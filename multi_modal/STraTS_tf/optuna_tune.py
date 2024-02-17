from model import build_mtand_strats
from utils import mortality_loss, CustomCallback, parse_args, build_forecast_loss
from data import load_mortality_dataset, shorten_dataset, load_mtand_mortality_dataset

import pickle
import numpy as np
import optuna
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import os
import argparse




def tune_model(args):
    
    batch_size, patience = args.batch_size, args.patience
    # STraTS args
    d_strats, N_strats, he_strats, dropout_strats = args.d_strats, args.N_strats, args.he_strats, args.dropout_strats
    # mTAND args
    max_time = args.max_time
    max_len = args.max_len


    def objective(trial):

        learning_rate = trial.suggest_float(
            'learning_rate',
            0.0001,
            0.001,
            log=False
        )

        he_mtand = trial.suggest_int(
            'he_mtand',
            1,
            16,
            step=2
        )
        N_mtand = trial.suggest_int(
            'N_mtand',
            0,
            2,
            step=1
        )
        d_mtand = trial.suggest_int(
            'd_mtand',
            4,
            128,
            step=4
        )
        dropout_mtand = trial.suggest_float(
            'dropout_mtand',
            0,
            0.6
        )
        len_time_query = trial.suggest_int(
            'len_time_query',
            20,
            500,
            step=10
        )
        train_ip, train_op, valid_ip, valid_op, test_ip, test_op, D, V, len_time_key = load_mtand_mortality_dataset(args.data_dir, max_time=max_time, with_demo=args.with_demo, len_time_query=len_time_query)
    
        print(f"V:{V}, D:{D}")

        model,fore_model = build_mtand_strats(
            D, V, max_len, 
            d_strats, N_strats, he_strats, dropout_strats,
            len_time_query, len_time_key, d_mtand, N_mtand, he_mtand, dropout_mtand, 
            forecast=True, with_demo=args.with_demo
        )
        

        model.compile(loss=mortality_loss, optimizer=Adam(learning_rate))
        fore_model.compile(loss=build_forecast_loss(V), optimizer=Adam(learning_rate))

        es = EarlyStopping(monitor='custom_metric', patience=patience, mode='max', restore_best_weights=True)

        cus = CustomCallback(validation_data=(valid_ip, valid_op), batch_size=batch_size)

        print(model.summary())
        
        his = model.fit(train_ip, train_op, batch_size=batch_size, epochs=1000, verbose=1, callbacks=[cus, es]).history

        return np.max(his['custom_metric'])

    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.isfile(f"{args.output_dir}/{args.study_name}_sampler.pkl"):
        print('Loading sampler')
        sampler = pickle.load(open(f"{args.output_dir}/{args.study_name}_sampler.pkl", "rb"))
    else: 
        sampler = optuna.samplers.TPESampler()
    if os.path.isfile(f"{args.output_dir}/{args.study_name}_pruner.pkl"):
        print('Loading pruner')
        pruner = pickle.load(open(f"{args.output_dir}/{args.study_name}_pruner.pkl", "rb"))
    else: 
        pruner = optuna.pruners.HyperbandPruner()

    study = optuna.create_study(
        direction='maximize', 
        study_name=args.study_name, 
        storage=f"sqlite:///{os.path.dirname(os.path.realpath(__file__))}/{args.output_dir}/{args.study_name}.db", 
        load_if_exists=True, 
        sampler=sampler, 
        pruner=pruner
    )
    study.enqueue_trial(
        {'learning_rate': 0.0005, 'he_mtand': 8, 'N_mtand': 0, 'd_mtand': 32, 'dropout_mtand': 0.2, 'len_time_query': 200},
        skip_if_exists=True
    )
    study.optimize(objective, n_trials=100, timeout=5*60*60)

    with open(f"{args.output_dir}/{args.study_name}_sampler.pkl", "wb") as fout:
        pickle.dump(study.sampler, fout)
    with open(f"{args.output_dir}/{args.study_name}_pruner.pkl", "wb") as fout:
        pickle.dump(study.pruner, fout)


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./optuna_tuning'
    )
    parser.add_argument(
        "--data_dir", type=str, help="A path to dataset folder"
    )
    parser.add_argument(
        "--study_name", type=str, help="Name of study", default='mtand_strats_tuning'
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=880,
        help="No. of obs in the dataset for each sequence",
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=48,
        help="Max Timing for time query",
    )
    parser.add_argument('--with_demo', action='store_true')
    parser.add_argument(
        "--d_strats",
        type=int,
        default=32,
        help="",
    )
    parser.add_argument(
        "--N_strats",
        type=int,
        default=2,
        help="",
    )
    parser.add_argument(
        "--he_strats",
        type=int,
        default=4,
        help="",
    )
    parser.add_argument(
        "--dropout_strats",
        type=float,
        default=0.2,
        help="",
    )
    parser.add_argument(
        "--d_demo",
        type=int,
        default=32,
        help="",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopper"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )

    args = parser.parse_args()

    return args

if __name__=='__main__':
    args = parse_args()

    tune_model(args)