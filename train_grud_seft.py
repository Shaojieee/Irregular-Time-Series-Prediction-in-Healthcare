from grud_model import load_grud_mortality_dataset, GRUDModel
from seft_model import load_seft_mortality_dataset, DeepSetAttentionModel
from utils import CustomCallback, get_res

import numpy as np
import os
import pandas as pd
import argparse

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


def train(args):
    lds = args.lds
    repeats = {k:args.repeats for k in lds}
    
    batch_size, lr, patience = args.batch_size, args.lr, args.patience
    
    assert args.model in ['GRUD', 'SeFT']
    if args.model=='GRUD':
        train_ip, train_op, valid_ip, valid_op, test_ip, test_op = load_grud_mortality_dataset(args.data_dir)
    elif args.model=='SeFT':
        train_ip, train_op, valid_ip, valid_op, test_ip, test_op = load_seft_mortality_dataset(args.data_dir)

    train_inds = np.arange(len(train_op))
    valid_inds = np.arange(len(valid_op))

    gen_res = {}
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(2021)
    for ld in lds:
        logs = {'val_metric':[], 'roc_auc':[], 'pr_auc':[], 'min_rp':[], 'loss':[], 'save_path':[]}
        np.random.shuffle(train_inds)
        np.random.shuffle(valid_inds)
        train_starts = [int(i) for i in np.linspace(0, len(train_inds)-int(ld*len(train_inds)/100), repeats[ld])]
        valid_starts = [int(i) for i in np.linspace(0, len(valid_inds)-int(ld*len(valid_inds)/100), repeats[ld])]
        # f.write('Training on '+str(ld)+' % of labaled data+\n'+'val_metric,roc_auc,pr_auc,min_rp,savepath\n')
        all_test_res = []
        for i in range(repeats[ld]):
            print ('Repeat', i, 'ld', ld)
            # Get train and validation data.
            curr_train_ind = train_inds[np.arange(train_starts[i], train_starts[i]+int(ld*len(train_inds)/100))]
            curr_valid_ind = valid_inds[np.arange(valid_starts[i], valid_starts[i]+int(ld*len(valid_inds)/100))]

            curr_train_ip = [ip[curr_train_ind] for ip in train_ip]
            curr_valid_ip = [ip[curr_valid_ind] for ip in valid_ip]
            curr_train_op = train_op[curr_train_ind]
            curr_valid_op = valid_op[curr_valid_ind]
            print ('Num train:',len(curr_train_op),'Num valid:',len(curr_valid_op))
            
            
            # Construct save_path.
            savepath = args.output_dir + '/repeat'+str(i)+'_'+str(ld)+'ld'+'.h5'

            print (savepath)
            # Build and compile model.

            if args.model=='GRUD':
                if args.dataset=='mimic':
                    model = GRUDModel.from_hyperparameter_dict(
                        output_activation='sigmoid',
                        n_outputs=1,
                        hparams={'n_units': 60, 'dropout': 0.2, 'recurrent_dropout': 0.2}
                    )
                elif args.dataset=='physionet':
                    model = GRUDModel.from_hyperparameter_dict(
                        output_activation='sigmoid',
                        n_outputs=1,
                        hparams={'n_units': 49, 'dropout': 0.2, 'recurrent_dropout': 0.2}
                    )
            elif args.model=='SeFT':
                if args.dataset=='mimic':
                    model = DeepSetAttentionModel.from_hyperparameter_dict(
                        output_activation='sigmoid',
                        n_outputs=1,
                        hparams={
                            'n_phi_layers': 4, 
                            'phi_width': 128, 
                            'n_psi_layers': 2,
                            'psi_width': 64,
                            'psi_latent_width': 128,
                            'dot_prod_dim': 128,
                            'n_heads': 4,
                            'attn_dropout': 0.5,
                            'latent_width': 32,
                            'phi_dropout': 0.0,
                            'n_rho_layers': 2,
                            'rho_width': 512,
                            'rho_dropout': 0.0,
                            'max_timescale': 100.0,
                            'n_positional_dims': 4,
                        }
                    )
                    model._n_modalities = 17
                elif args.dataset=='physionet':
                    model = DeepSetAttentionModel.from_hyperparameter_dict(
                        output_activation='sigmoid',
                        n_outputs=1,
                        hparams={
                            'n_phi_layers': 4, 
                            'phi_width': 128, 
                            'n_psi_layers': 2,
                            'psi_width': 64,
                            'psi_latent_width': 128,
                            'dot_prod_dim': 128,
                            'n_heads': 4,
                            'attn_dropout': 0.5,
                            'latent_width': 32,
                            'phi_dropout': 0.0,
                            'n_rho_layers': 2,
                            'rho_width': 512,
                            'rho_dropout': 0.0,
                            'max_timescale': 100.0,
                            'n_positional_dims': 4,
                        }
                    )
                    model._n_modalities = 37
            else:
                print('Please specify valid model')
                return

            model.compile(loss='binary_crossentropy', optimizer=Adam(args.lr))
            # Train model.
            es = EarlyStopping(monitor='custom_metric', patience=patience, mode='max', 
                            restore_best_weights=True)

            cus = CustomCallback(validation_data=(curr_valid_ip, curr_valid_op), batch_size=batch_size)
            his = model.fit(
                curr_train_ip,
                curr_train_op,
                epochs=1000,
                verbose=1, 
                callbacks=[cus, es]
            ).history
            
            model.save_weights(savepath)

            # Test and write to log.
            rocauc, prauc, minrp, test_loss = get_res(test_op, model.predict(test_ip, verbose=0, batch_size=batch_size))
            # f.write(str(np.min(his['custom_metric']))+str(rocauc)+str(prauc)+str(minrp)+savepath+'\n')
            
            logs['val_metric'].append(np.max(his['custom_metric']));logs['roc_auc'].append(rocauc);logs['pr_auc'].append(prauc);
            logs['min_rp'].append(minrp);logs['loss'].append(test_loss);logs['save_path'].append(savepath);
            
            print ('Test results: ', rocauc, prauc, minrp, test_loss)
            all_test_res.append([rocauc, prauc, minrp, test_loss])

            pd.DataFrame(logs).to_csv(f'{args.output_dir}/ld_{str(ld)}.csv')
            
        gen_res[ld] = []
        for i in range(len(all_test_res[0])):
            nums = [test_res[i] for test_res in all_test_res]
            gen_res[ld].append((np.mean(nums), np.std(nums)))

        print ('gen_res', gen_res)
    

def parse_args():

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset", type=str, default='mimic'
    )
    parser.add_argument(
        "--data_dir", type=str, default=None
    )
    parser.add_argument(
        "--output_dir", type=str, default=None
    )
    parser.add_argument(
        '--model', type=str
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001
    )
    parser.add_argument(
        "--batch_size", type=int, default=32
    )
    parser.add_argument(
        "--patience", type=int, default=10
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeats",
    )
    parser.add_argument(
        "--lds",
        type=list_of_ints,
        default=[50],
        help="Percentage of training and validation data",
    )

    args = parser.parse_args()

    return args


if __name__=='__main__':
    args = parse_args()

    train(args)