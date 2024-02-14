from model import build_strats, build_modified_strats, build_special_strats
from utils import CustomCallback, get_res, forecast_parse_args, build_forecast_loss
from data import load_forecast_dataset
from tqdm import tqdm

import pickle
import numpy as np

from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import os



def main():
    args = forecast_parse_args()
    train(args)


def train(args):
    
    batch_size, lr, patience, samples_per_epoch = args.batch_size, args.lr, args.patience, args.samples_per_epoch
    d, N, he, dropout = args.d, args.N, args.he, args.dropout
    max_len = args.max_len
    fore_file_name = args.save_path

    D,V = 2, 129

    fore_train_ip, fore_train_op, fore_valid_ip, fore_valid_op = load_forecast_dataset(args.data_dir, with_demo=args.with_demo)


    gen_res = {}
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(2021)
    
    
    if args.custom_strats:
        if args.special_transformer:
            model, fore_model =  build_special_strats(D, max_len, V, d, N, he, dropout, forecast=True, with_demo=args.with_demo)
        else:
            model, fore_model =  build_modified_strats(D, max_len, V, d, N, he, dropout, forecast=True, with_demo=args.with_demo)
    else:
        model, fore_model =  build_strats(D, max_len, V, d, N, he, dropout, forecast=True, with_demo=args.with_demo)
        
    print (fore_model.summary())
    fore_model.compile(loss=build_forecast_loss(V), optimizer=Adam(lr))

    # Pretrain fore_model.
    best_val_loss = np.inf
    N_fore = len(fore_train_op)
    for e in range(1000):
        e_indices = np.random.choice(range(N_fore), size=samples_per_epoch, replace=False)
        e_loss = 0
        pbar = tqdm(range(0, len(e_indices), batch_size))
        for start in pbar:
            ind = e_indices[start:start+batch_size]
            e_loss += fore_model.train_on_batch([ip[ind] for ip in fore_train_ip], fore_train_op[ind])
            pbar.set_description('%f'%(e_loss/(start+1)))
        val_loss = fore_model.evaluate(fore_valid_ip, fore_valid_op, batch_size=batch_size, verbose=0)
        print ('Epoch', e, 'loss', e_loss*batch_size/samples_per_epoch, 'val loss', val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            fore_model.save_weights(args.output_dir+'/'+fore_file_name)
            best_epoch = e
        if (e-best_epoch)>patience:
            break


if __name__=='__main__':
    main()