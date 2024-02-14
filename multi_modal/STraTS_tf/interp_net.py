from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
import keras
from keras import activations
from keras.layers import Input, Dense, GRU, Lambda, Permute
from keras.models import Model
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd
from tensorflow.keras.utils import plot_model

import gzip

from utils import get_res
from tqdm import tqdm

import argparse
import os
import pickle


class single_channel_interp(Layer):

    def __init__(self, ref_points, hours_look_ahead, **kwargs):
        self.ref_points = ref_points
        self.hours_look_ahead = hours_look_ahead  # in hours
        super(single_channel_interp, self).__init__(**kwargs)

    def build(self, input_shape):
        #input_shape [batch, features, time_stamp]
        self.time_stamp = input_shape[2]
        self.d_dim = input_shape[1] // 4
        self.activation = activations.get('sigmoid')
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.d_dim, ),
            initializer=keras.initializers.Constant(value=0.0),
            trainable=True)
        super(single_channel_interp, self).build(input_shape)

    def call(self, x, reconstruction=False):
        self.reconstruction = reconstruction
        x_t = x[:, :self.d_dim, :]
        d = x[:, 2*self.d_dim:3*self.d_dim, :]
        if reconstruction:
            output_dim = self.time_stamp
            m = x[:, 3*self.d_dim:, :]
            ref_t = K.tile(d[:, :, None, :], (1, 1, output_dim, 1))
        else:
            m = x[:, self.d_dim: 2*self.d_dim, :]
            ref_t = np.linspace(0, self.hours_look_ahead, self.ref_points)
            output_dim = self.ref_points
            ref_t.shape = (1, ref_t.shape[0])
        #x_t = x_t*m
        d = K.tile(d[:, :, :, None], (1, 1, 1, output_dim))
        mask = K.tile(m[:, :, :, None], (1, 1, 1, output_dim))
        x_t = K.tile(x_t[:, :, :, None], (1, 1, 1, output_dim))
        norm = (d - ref_t)*(d - ref_t)
        a = K.ones((self.d_dim, self.time_stamp, output_dim))
        pos_kernel = K.log(1 + K.exp(self.kernel))
        alpha = a*pos_kernel[:, np.newaxis, np.newaxis]
        w = K.logsumexp(-alpha*norm + K.log(mask), axis=2)
        w1 = K.tile(w[:, :, None, :], (1, 1, self.time_stamp, 1))
        w1 = K.exp(-alpha*norm + K.log(mask) - w1)
        y = K.sum(w1*x_t, axis=2)
        if reconstruction:
            rep1 = tf.concat([y, w], 1)
        else:
            w_t = K.logsumexp(-10.0*alpha*norm + K.log(mask),
                              axis=2)  # kappa = 10
            w_t = K.tile(w_t[:, :, None, :], (1, 1, self.time_stamp, 1))
            w_t = K.exp(-10.0*alpha*norm + K.log(mask) - w_t)
            y_trans = K.sum(w_t*x_t, axis=2)
            rep1 = tf.concat([y, w, y_trans], 1)
        return rep1

    def compute_output_shape(self, input_shape):
        if self.reconstruction:
            return (input_shape[0], 2*self.d_dim, self.time_stamp)
        return (input_shape[0], 3*self.d_dim, self.ref_points)


class cross_channel_interp(Layer):

    def __init__(self, **kwargs):
        super(cross_channel_interp, self).__init__(**kwargs)

    def build(self, input_shape):
        self.d_dim = input_shape[1] // 3
        self.activation = activations.get('sigmoid')
        self.cross_channel_interp = self.add_weight(
            name='cross_channel_interp',
            shape=(self.d_dim, self.d_dim),
            initializer=keras.initializers.Identity(gain=1.0),
            trainable=True)

        super(cross_channel_interp, self).build(input_shape)

    def call(self, x, reconstruction=False):
        self.reconstruction = reconstruction
        self.output_dim = K.int_shape(x)[-1]
        cross_channel_interp = self.cross_channel_interp
        y = x[:, :self.d_dim, :]
        w = x[:, self.d_dim:2*self.d_dim, :]
        intensity = K.exp(w)
        y = tf.transpose(y, perm=[0, 2, 1])
        w = tf.transpose(w, perm=[0, 2, 1])
        w2 = w
        w = K.tile(w[:, :, :, None], (1, 1, 1, self.d_dim))
        den = K.logsumexp(w, axis=2)
        w = K.exp(w2 - den)
        mean = K.mean(y, axis=1)
        mean = K.tile(mean[:, None, :], (1, self.output_dim, 1))
        w2 = K.dot(w*(y - mean), cross_channel_interp) + mean
        rep1 = tf.transpose(w2, perm=[0, 2, 1])
        if reconstruction is False:
            y_trans = x[:, 2*self.d_dim:3*self.d_dim, :]
            y_trans = y_trans - rep1  # subtracting smooth from transient part
            rep1 = tf.concat([rep1, intensity, y_trans], 1)
        return rep1

    def compute_output_shape(self, input_shape):
        if self.reconstruction:
            return (input_shape[0], self.d_dim, self.output_dim)
        return (input_shape[0], 3*self.d_dim, self.output_dim)


def interp_net(num_features, timestamp, ref_points, hours_look_ahead, units, recurrent_dropout):
    main_input = Input(shape=(4*num_features, timestamp), name='input')
    sci = single_channel_interp(ref_points, hours_look_ahead)
    cci = cross_channel_interp()
    interp = cci(sci(main_input))
    reconst = cci(sci(main_input, reconstruction=True),
                  reconstruction=True)
    aux_output = Lambda(lambda x: x, name='aux_output')(reconst)
    z = Permute((2, 1))(interp)
    z = GRU(units, activation='tanh', recurrent_dropout=recurrent_dropout, dropout=recurrent_dropout)(z)
    main_output = Dense(1, activation='sigmoid', name='main_output')(z)
    model = Model([main_input], [main_output, aux_output])
    
    print(model.summary())
    return model


def generate_data(data_path, output_dir, start_hour=0, end_hour=24, input_dropout=0.2):
    data, oc, train_ind, valid_ind, test_ind = pickle.load(open(data_path, 'rb'))
    # Filter labeled data in first 24h.
    data = data.loc[data.ts_ind.isin(np.concatenate((train_ind, valid_ind, test_ind), axis=-1))]
    data = data.loc[(data.hour>=start_hour)&(data.hour<=end_hour)]

    oc = oc.loc[oc.ts_ind.isin(np.concatenate((train_ind, valid_ind, test_ind), axis=-1))]
    # Fix age.
    data.loc[(data.variable=='Age')&(data.value>200), 'value'] = 91.4
    # Get y and N.
    y = np.array(oc.sort_values(by='ts_ind')['in_hospital_mortality']).astype('float32')
    N = data.ts_ind.max() + 1
    # Get static data with mean fill and missingness indicator.
    static_varis = ['Age', 'Gender']
    ii = data.variable.isin(static_varis)
    static_data = data.loc[ii]
    data = data.loc[~ii]
    def inv_list(l, start=0):
        d = {}
        for i in range(len(l)):
            d[l[i]] = i+start
        return d
    static_var_to_ind = inv_list(static_varis)
    D = len(static_varis)
    demo = np.zeros((N, D))
    for row in tqdm(static_data.itertuples()):
        demo[row.ts_ind, static_var_to_ind[row.variable]] = row.value
    # Normalize static data.
    means = demo.mean(axis=0, keepdims=True)
    stds = demo.std(axis=0, keepdims=True)
    stds = (stds==0)*1 + (stds!=0)*stds
    demo = (demo-means)/stds
    # Trim to max len.
    data = data.sample(frac=1)
    print(data.groupby('ts_ind')['hour'].nunique().quantile([0.25, 0.5, 0.75, 0.9, 0.99]))

    max_timestep = int(data.groupby('ts_ind')['hour'].nunique().quantile(0.99))

    # Get N, V, var_to_ind.
    N = data.ts_ind.max() + 1
    varis = sorted(list(set(data.variable)))
    V = len(varis)
    def inv_list(l, start=0):
        d = {}
        for i in range(len(l)):
            d[l[i]] = i+start
        return d

    var_to_ind = inv_list(varis, start=1)
    data['vind'] = data.variable.map(var_to_ind)
    data = data[['ts_ind', 'vind', 'hour', 'value']]
    # Add obs index.
    data = data.sort_values(by=['ts_ind', 'hour', 'vind']).reset_index(drop=True)
    data = data.reset_index().rename(columns={'index':'obs_ind'})
    data = data.merge(data.groupby('ts_ind').agg({'obs_ind':'min'}).reset_index().rename(columns={ \
                                                                'obs_ind':'first_obs_ind'}), on='ts_ind')
    data['obs_ind'] = data['obs_ind'] - data['first_obs_ind']
    # Find max_timestep.
    print ('max_timestep', max_timestep)

    times_inp = np.zeros((N, V, max_timestep), dtype='float32')
    values_inp = np.zeros((N, V, max_timestep), dtype='float32')
    mask_inp = np.zeros((N, V, max_timestep), dtype='int32')

    cur_time = None
    time_index = 0
    prev_ts_ind = 0
    
    for row in tqdm(data.itertuples()):
        # Check if to iterate to next patient
        if time_index==max_timestep-1 and prev_ts_ind==row.ts_ind:
            continue
        # For first patient
        if cur_time==None:
            cur_time = row.hour
            time_index = 0
        # if different patient
        elif prev_ts_ind!=row.ts_ind:
            prev_ts_ind = row.ts_ind
            time_index = 0
            cur_time = row.hour
        # If same patient but different time
        elif cur_time!=row.hour:
            time_index += 1
            cur_time = row.hour
        
        v = row.vind-1 #variable
        times_inp[row.ts_ind, :, time_index] = row.hour
        values_inp[row.ts_ind, v, time_index] = row.value
        mask_inp[row.ts_ind, v, time_index] = 1

        
    def mean_imputation(vitals, mask):
        """For the time series missing entirely, our interpolation network 
        assigns the starting point (time t=0) value of the time series to 
        the global mean before applying the two-layer interpolation network.
        In such cases, the first interpolation layer just outputs the global
        mean for that channel, but the second interpolation layer performs 
        a more meaningful interpolation using the learned correlations from
        other channels."""
        counts = np.sum(np.sum(mask, axis=2), axis=0)
        mean_values = np.sum(np.sum(vitals*mask, axis=2), axis=0)/counts
        for i in tqdm(range(mask.shape[0])):
            for j in range(mask.shape[1]):
                if np.sum(mask[i, j]) == 0:
                    mask[i, j, 0] = 1
                    vitals[i, j, 0] = mean_values[j]
        return

    def hold_out(mask, perc=0.2):
        """To implement the autoencoder component of the loss, we introduce a set
        of masking variables mr (and mr1) for each data point. If drop_mask = 0,
        then we removecthe data point as an input to the interpolation network,
        and includecthe predicted value at this time point when assessing
        the autoencoder loss. In practice, we randomly select 20% of the
        observed data points to hold out from
        every input time series."""
        drop_mask = np.ones_like(mask)
        drop_mask *= mask
        for i in tqdm(range(mask.shape[0])):
            for j in range(mask.shape[1]):
                count = np.sum(mask[i, j], dtype='int')
                if int(0.20*count) > 1:
                    index = 0
                    r = np.ones((count, 1))
                    b = np.random.choice(count, int(0.20*count), replace=False)
                    r[b] = 0
                    for k in range(mask.shape[2]):
                        if mask[i, j, k] > 0:
                            drop_mask[i, j, k] = r[index]
                            index += 1
        return drop_mask


    mean_imputation(values_inp, mask_inp)

    x = np.concatenate((values_inp, mask_inp, times_inp, hold_out(mask_inp, input_dropout)), axis=1)

    feature_std = np.array(data.groupby(by=['vind'])['value'].std().sort_index())
    # Replace Famotidine with std 1.0. This vairable has 0.0 std. All the values are 1.0.
    feature_std[feature_std==0] = 1.0

    train_ip, train_op = x[train_ind], y[train_ind]
    valid_ip, valid_op = x[valid_ind], y[valid_ind]
    test_ip, test_op = x[test_ind], y[test_ind]


    os.makedirs(output_dir, exist_ok=True)
    with gzip.GzipFile(f'{output_dir}/train_ip.npy.gz', 'w') as f:
        np.save(f, train_ip)
    with gzip.GzipFile(f'{output_dir}/train_op.npy.gz', 'w') as f:
        np.save(f, train_op)

    with gzip.GzipFile(f'{output_dir}/val_ip.npy.gz', 'w') as f:
        np.save(f, valid_ip)
    with gzip.GzipFile(f'{output_dir}/val_op.npy.gz', 'w') as f:
        np.save(f, valid_op)

    with gzip.GzipFile(f'{output_dir}/test_ip.npy.gz', 'w') as f:
        np.save(f, test_ip)
    with gzip.GzipFile(f'{output_dir}/test_op.npy.gz', 'w') as f:
        np.save(f, test_op) 
    
    with gzip.GzipFile(f'{output_dir}/feature_std.npy.gz', 'w') as f:
        np.save(f, feature_std) 


def load_data(data_dir):
    with gzip.GzipFile(f'{data_dir}/test_ip.npy.gz', 'r') as f:
        test_ip = np.load(f)
    with gzip.GzipFile(f'{data_dir}/test_op.npy.gz', 'r') as f:
        test_op = np.load(f)

    with gzip.GzipFile(f'{data_dir}/train_ip.npy.gz', 'r') as f:
        train_ip = np.load(f)
    with gzip.GzipFile(f'{data_dir}/train_op.npy.gz', 'r') as f:
        train_op = np.load(f)

    with gzip.GzipFile(f'{data_dir}/val_ip.npy.gz', 'r') as f:
        valid_ip = np.load(f)
    with gzip.GzipFile(f'{data_dir}/val_op.npy.gz', 'r') as f:
        valid_op = np.load(f)

    with gzip.GzipFile(f'{data_dir}/feature_std.npy.gz', 'r') as f:
        feature_std = np.load(f)

    return train_ip, train_op, valid_ip, valid_op, test_ip, test_op, feature_std


def create_customloss(feature_std, num_features):
    def customloss(ytrue, ypred):
        """ Autoencoder loss
        """
        # standard deviation of each feature mentioned in paper for MIMIC_III data
        wc = feature_std
        wc.shape = (1, num_features)
        y = ytrue[:, :num_features, :]
        m2 = ytrue[:, 3*num_features:4*num_features, :]
        m2 = 1 - m2
        m1 = ytrue[:, num_features:2*num_features, :]
        m = m1*m2
        ypred = ypred[:, :num_features, :]
        x = (y - ypred)*(y - ypred)
        x = x*m
        count = tf.reduce_sum(m, axis=2)
        count = tf.where(count > 0, count, tf.ones_like(count))
        x = tf.reduce_sum(x, axis=2)/count
        x = x/(wc**2)  # dividing by standard deviation
        x = tf.reduce_sum(x, axis=1)/num_features
        return tf.reduce_mean(x)
    return customloss


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, validation_data, batch_size):
        self.val_x, self.val_y = validation_data
        self.batch_size = batch_size
        super(keras.callbacks.Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.val_x, verbose=0, batch_size=self.batch_size)[0]
        if type(y_pred)==type([]):
            y_pred = y_pred[0]
        precision, recall, thresholds = precision_recall_curve(self.val_y, y_pred)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(self.val_y, y_pred)
        logs['custom_metric'] = pr_auc + roc_auc
        print ('val_aucs:', pr_auc, roc_auc, pr_auc+roc_auc)


class CustomEarlyStopping(keras.callbacks.Callback):
    def __init__(self, patience, mode, min_delta, metric):
        self.patience = patience
        self.mode = mode 
        self.min_delta = min_delta
        self.metric = metric
        self.best_weights = None 
        self.best_score = np.inf if mode=='min' else -np.inf
        self.cur_patience = 0
        if mode=='min':
            self.monitor_op = np.less
        elif mode=='max':
            self.monitor_op = np.greater
        super(keras.callbacks.Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        cur_metric = logs[self.metric]
        if self.monitor_op(cur_metric - self.min_delta, self.best_score):
            print(f"new best score of {cur_metric}")
            self.best_score = cur_metric 
            self.best_weights = self.model.get_weights()
            self.cur_patience = 0
        else:
            self.cur_patience += 1

        if self.cur_patience>=self.patience:
            print(f'Early stopping at epoch {epoch}')
            self.model.set_weights(self.best_weights)
            self.stopped_epoch = epoch
            self.model.stop_training = True



def train(args):

    os.makedirs(args.output_dir, exist_ok=True)

    train_ip, train_op, valid_ip, valid_op, test_ip, test_op, feature_std = load_data(
        data_dir=args.data_dir
    )

    timestamp = train_ip.shape[2]
    num_features = train_ip.shape[1] // 4

    customloss = create_customloss(feature_std, num_features)

    
    gen_res = {}
    train_inds = np.arange(len(train_ip))
    valid_inds = np.arange(len(valid_ip))
    np.random.seed(2021)
    for ld in args.lds:
        logs = {'val_metric':[], 'roc_auc':[], 'pr_auc':[], 'min_rp':[], 'loss':[]}
        np.random.shuffle(train_inds)
        np.random.shuffle(valid_inds)
        train_starts = [int(i) for i in np.linspace(0, len(train_inds)-int(ld*len(train_inds)/100), args.repeats)]
        valid_starts = [int(i) for i in np.linspace(0, len(valid_inds)-int(ld*len(valid_inds)/100), args.repeats)]
        all_test_res = []
        for i in range(args.repeats):

            cur_train_ind = train_inds[np.arange(train_starts[i], train_starts[i]+int(ld*len(train_inds)/100))]
            cur_valid_ind = valid_inds[np.arange(valid_starts[i], valid_starts[i]+int(ld*len(valid_inds)/100))]
            cur_train_ip = train_ip[cur_train_ind]
            cur_valid_ip = valid_ip[cur_valid_ind]
            cur_train_op = train_op[cur_train_ind]
            cur_valid_op = valid_op[cur_valid_ind]

            input_dropout = 0.2
            model = interp_net(
                num_features=num_features,
                timestamp=timestamp,
                ref_points=96,
                hours_look_ahead=24,
                units=100,
                recurrent_dropout=0.2
            )

            if i==0:
                print(model.summary())
                plot_model(model, show_layer_names=True, show_shapes=True, to_file=args.output_dir+f'/model_{str(lds)}.png')

            model.compile(
                optimizer=keras.optimizers.Adam(lr=0.001),
                loss={'main_output': 'binary_crossentropy', 'aux_output': customloss},
                loss_weights={'main_output': 1., 'aux_output': 1.}
            )

            cus = CustomCallback(validation_data=(cur_valid_ip, cur_valid_op), batch_size=32)
            earlystop = CustomEarlyStopping(
                patience=20, 
                mode='max', 
                min_delta=0, 
                metric='custom_metric'
            )
            # earlystop = keras.callbacks.EarlyStopping(
            #     monitor='custom_metric', 
            #     min_delta=0.0000, 
            #     mode='max',
            #     patience=10, 
            #     verbose=1,
            # )

            his = model.fit(
                {'input': cur_train_ip}, 
                {'main_output': cur_train_op, 'aux_output': cur_train_ip},
                validation_data=({'input': cur_valid_ip}, {'main_output': cur_valid_op, 'aux_output': cur_valid_ip}),
                batch_size=32,
                callbacks=[cus, earlystop],
                epochs=1000,
                verbose=1
            ).history
            
            rocauc, prauc, minrp, test_loss = get_res(test_op, model.predict(test_ip, verbose=0, batch_size=32)[0])

            logs['val_metric'].append(np.max(his['custom_metric']));logs['roc_auc'].append(rocauc);logs['pr_auc'].append(prauc);
            logs['min_rp'].append(minrp);logs['loss'].append(test_loss);
            # logs['save_path'].append(savepath);
            print ('Test results: ', rocauc, prauc, minrp, test_loss)
            
            rocauc, prauc, minrp, valid_loss = get_res(cur_valid_op, model.predict(cur_valid_ip, verbose=0, batch_size=32)[0])

            print('Val results:', rocauc, prauc, minrp, valid_loss)

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
        "--data_dir", type=str, default=None
    )
    parser.add_argument(
        "--output_dir", type=str, default=None
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Number of repeats",
    )
    parser.add_argument(
        "--lds",
        type=list_of_ints,
        default=None,
        help="Percentage of training and validation data",
    )

    args = parser.parse_args()

    return args
    

if __name__=='__main__':

    # generate_data(
    #     data_path='./interp_net_mimic_iii_preprocessed.pkl', 
    #     output_dir='./data_interp_net', 
    #     input_dropout=0.2
    # )

    args = parse_args()
    train(args)