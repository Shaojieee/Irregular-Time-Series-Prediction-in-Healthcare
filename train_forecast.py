from model import build_strats, build_modified_strats, build_special_strats, build_imputed_strats, build_mtand, build_mtand_strats
from utils import forecast_parse_args, build_forecast_loss, parse_args
from data import load_forecast_dataset, load_mtand_forecast_dataset
from tqdm import tqdm

import pickle
import numpy as np
import pandas as pd

from tensorflow.keras.optimizers import Adam
import os



def main():
    args = parse_args()
    train(args)


def train(args):
    
    batch_size, lr, patience = args.batch_size, args.lr, args.patience
    # STraTS args
    d_strats, N_strats, he_strats, dropout_strats = args.d_strats, args.N_strats, args.he_strats, args.dropout_strats
    # mTAND args
    d_mtand, N_mtand, he_mtand, len_time_query, dropout_mtand = args.d_mtand, args.N_mtand, args.he_mtand, args.len_time_query, args.dropout_mtand

    args.with_imputation = True if args.model_type=='imputed' else False
    if 'mtand' in args.model_type:
        train_ip, train_op, valid_ip, valid_op, D, V, max_len, len_time_key = load_mtand_forecast_dataset(args.data_dir, with_demo=args.with_demo, len_time_query=len_time_query)
        print(f"len_time_query: {len_time_query}, len_time_key: {len_time_key}")
    else:
        train_ip, train_op, valid_ip, valid_op, D, V, max_len = load_forecast_dataset(args.data_dir, with_demo=args.with_demo)
    
    if args.model_type=='mtand':
        train_ip = [train_ip[0]] + train_ip[-4:]
        valid_ip = [valid_ip[0]] + valid_ip[-4:]
    
    print(f"V:{V}, D:{D}")
    

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(2021)
    
    
    if args.model_type=='imputed':
        model, fore_model =  build_imputed_strats(D, max_len, V, d_strats, N_strats, he_strats, dropout_strats, forecast=True, with_demo=args.with_demo)
    elif args.model_type=='custom':
        model, fore_model =  build_modified_strats(D, max_len, V, d_strats, N_strats, he_strats, dropout_strats, forecast=True, with_demo=args.with_demo)
    elif args.model_type=='special': 
        model, fore_model =  build_special_strats(D, max_len, V, d_strats, N_strats, he_strats, dropout_strats, forecast=True, with_demo=args.with_demo)
    elif args.model_type=='mtand_strats':
        model,fore_model = build_mtand_strats(D, V, max_len, d_strats, N_strats, he_strats, dropout_strats, len_time_query, len_time_key, d_mtand, N_mtand, he_mtand, dropout_mtand, forecast=True, with_demo=args.with_demo)
    elif args.model_type=='mtand':
        model, fore_model = build_mtand(D, args.len_time_query, len_time_key, V, d_mtand, args.d_demo, N_mtand, he_mtand, dropout_mtand, forecast=True, with_demo=args.with_demo)
    elif args.model_type=='strats':
        model, fore_model =  build_strats(D, max_len, V, d_strats, N_strats, he_strats, dropout_strats, forecast=True, with_demo=args.with_demo)
    else:
        print('Model not found')
        return
    
    print (fore_model.summary())
    fore_model.compile(loss=build_forecast_loss(V), optimizer=Adam(lr))
    logs = {'val_loss':[], 'train_loss':[],}
    # Pretrain fore_model.
    best_val_loss = np.inf
    N_fore = len(train_op)
    for e in range(1000):
        e_indices = np.random.choice(range(N_fore), size=N_fore, replace=False)
        e_loss = 0
        pbar = tqdm(range(0, len(e_indices), batch_size))
        for start in pbar:
            ind = e_indices[start:start+batch_size]
            e_loss += fore_model.train_on_batch([ip[ind] for ip in train_ip], train_op[ind])
            pbar.set_description('%f'%(e_loss/(start+1)))
        val_loss = fore_model.evaluate(valid_ip, valid_op, batch_size=batch_size, verbose=0)
        print ('Epoch', e, 'loss', e_loss*batch_size/N_fore, 'val loss', val_loss)
        logs['val_loss'].append(val_loss); logs['train_loss'].append(e_loss*batch_size/N_fore)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            fore_model.save_weights(args.output_dir+'/forecast_model.h5')
            best_epoch = e
        if (e-best_epoch)>patience:
            break
        pd.DataFrame(logs).to_csv(f'{args.output_dir}/forecast_loss.csv')
        

if __name__=='__main__':
    main()