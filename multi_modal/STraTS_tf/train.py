from model import build_imputed_strats, build_strats, build_modified_strats, build_special_strats, build_mtand_strats, build_mtand
from utils import mortality_loss, CustomCallback, get_res, parse_args, build_forecast_loss
from data import load_mortality_dataset, shorten_dataset, load_mtand_mortality_dataset

import pickle
import numpy as np

from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import os



def main():
    args = parse_args()
    train(args)


def train(args):
    lds = args.lds
    repeats = {k:args.repeats for k in lds}
    
    batch_size, lr, patience = args.batch_size, args.lr, args.patience
    # STraTS args
    d_strats, N_strats, he_strats, dropout_strats = args.d_strats, args.N_strats, args.he_strats, args.dropout_strats
    # mTAND args
    d_mtand, N_mtand, he_mtand, len_time_query, max_time, dropout_mtand = args.d_mtand, args.N_mtand, args.he_mtand, args.len_time_query, args.max_time, args.dropout_mtand

    max_len = args.max_len

    args.with_imputation = True if args.model_type=='imputed' else False

    if 'mtand' in args.model_type:
        train_ip, train_op, valid_ip, valid_op, test_ip, test_op, D, V, len_time_key = load_mtand_mortality_dataset(args.data_dir, max_time=max_time, with_demo=args.with_demo, len_time_query=len_time_query)
    else:
        train_ip, train_op, valid_ip, valid_op, test_ip, test_op, D, V = load_mortality_dataset(args.data_dir, with_demo=args.with_demo, with_imputation=args.with_imputation)
    
    if args.with_imputation:
        train_ip = shorten_dataset(train_ip, max_len=args.max_len)
        valid_ip = shorten_dataset(valid_ip, max_len=args.max_len)
        test_ip = shorten_dataset(test_ip, max_len=args.max_len)
    if args.model_type=='mtand':
        train_ip = [train_ip[0]] + train_ip[-4:]
        valid_ip = [valid_ip[0]] + valid_ip[-4:]
        test_ip = [test_ip[0]] + test_ip[-4:]

    print(f"V:{V}, D:{D}")

    train_inds = np.arange(len(train_op))
    valid_inds = np.arange(len(valid_op))

    gen_res = {}
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(2021)
    for ld in lds:
        logs = {'val_metric':[], 'roc_auc':[], 'pr_auc':[], 'min_rp':[], 'loss':[], 'save_path':[]}
        
        # train_starts = [int(i) for i in np.linspace(0, len(train_inds)-int(ld*len(train_inds)/100), repeats[ld])]
        # valid_starts = [int(i) for i in np.linspace(0, len(valid_inds)-int(ld*len(valid_inds)/100), repeats[ld])]
        # f.write('Training on '+str(ld)+' % of labaled data+\n'+'val_metric,roc_auc,pr_auc,min_rp,savepath\n')
        all_test_res = []
        for i in range(repeats[ld]):
            print ('Repeat', i, 'ld', ld)
            # Get train and validation data.
            # curr_train_ind = train_inds[np.arange(train_starts[i], train_starts[i]+int(ld*len(train_inds)/100))]
            # curr_valid_ind = valid_inds[np.arange(valid_starts[i], valid_starts[i]+int(ld*len(valid_inds)/100))]
            np.random.shuffle(train_inds)
            np.random.shuffle(valid_inds)
            curr_train_ind = np.random.choice(train_inds, size=int(ld*len(train_inds)), replace=False)
            curr_valid_ind = np.random.choice(valid_inds, int(ld*len(valid_inds)), replace=False)
            print(f'Train : {len(curr_train_ind)} Val: {len(curr_valid_ind)}')
            curr_train_ip = [ip[curr_train_ind] for ip in train_ip]
            curr_valid_ip = [ip[curr_valid_ind] for ip in valid_ip]
            curr_train_op = train_op[curr_train_ind]
            curr_valid_op = valid_op[curr_valid_ind]
            print ('Num train:',len(curr_train_op),'Num valid:',len(curr_valid_op))
            # Construct save_path.
            savepath = args.output_dir + '/repeat'+str(i)+'_'+str(ld)+'ld'+'.h5'

            print (savepath)
            # Build and compile model.
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

            model.compile(loss=mortality_loss, optimizer=Adam(lr))
            fore_model.compile(loss=build_forecast_loss(V), optimizer=Adam(lr))

            if i==0:
                print(model.summary())
                plot_model(model, show_layer_names=True, show_shapes=True, to_file=args.output_dir+f'/model_{str(lds)}.png')


            # Load pretrained weights here.
            if args.model_weights is not None:
                print('Loading self-supervised weights')
                fore_model.load_weights(args.model_weights)

            # Train model.
            es = EarlyStopping(monitor='custom_metric', patience=patience, mode='max', restore_best_weights=True)

            cus = CustomCallback(validation_data=(curr_valid_ip, curr_valid_op), batch_size=batch_size)
            his = model.fit(curr_train_ip, curr_train_op, batch_size=batch_size, epochs=1000,
                            verbose=1, callbacks=[cus, es]).history
            model.save_weights(savepath)
            # Test and write to log.
            rocauc, prauc, minrp, test_loss = get_res(test_op, model.predict(test_ip, verbose=0, batch_size=batch_size))
            # f.write(str(np.min(his['custom_metric']))+str(rocauc)+str(prauc)+str(minrp)+savepath+'\n')
            
            logs['val_metric'].append(np.max(his['custom_metric']));logs['roc_auc'].append(rocauc);logs['pr_auc'].append(prauc);
            logs['min_rp'].append(minrp);logs['loss'].append(test_loss);logs['save_path'].append(savepath);
            
            print ('Test results: ', rocauc, prauc, minrp, test_loss)
            all_test_res.append([rocauc, prauc, minrp, test_loss])


            rocauc, prauc, minrp, test_loss = get_res(curr_valid_op, model.predict(curr_valid_ip, verbose=0, batch_size=batch_size))
            print ('Val results: ', rocauc, prauc, minrp, test_loss)

            pd.DataFrame(logs).to_csv(f'{args.output_dir}/ld_{str(ld)}.csv')

            del model
            
        gen_res[ld] = []
        for i in range(len(all_test_res[0])):
            nums = [test_res[i] for test_res in all_test_res]
            gen_res[ld].append((np.mean(nums), np.std(nums)))

        print ('gen_res', gen_res)


if __name__=='__main__':
    main()