import gzip
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os
import json
import math


def load_mortality_dataset(path, with_demo=True, with_imputation=False):
    X = {'train': [], 'val': [], 'test': []}
    Y = {'train': [], 'val': [], 'test': []}

    for mode in ['train', 'val', 'test']:
        if with_demo:
            with gzip.GzipFile(f'{path}/{mode}_demos.npy.gz', 'r') as f:
                X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_times.npy.gz', 'r') as f:
            X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_values.npy.gz', 'r') as f:
            X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_varis.npy.gz', 'r') as f:
            X[mode].append(np.load(f))
        if with_imputation:
            with gzip.GzipFile(f'{path}/{mode}_imputed_mask.npy.gz', 'r') as f:
                X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_y.npy.gz', 'r') as f:
            Y[mode] = np.load(f)

    with open(f'{path}/extra_params.json') as f:
        params = json.load(f)
    
    V = int(params['V'])
    D = int(params['D'])
    max_len = int(params['max_len'])
    

    return X['train'], Y['train'], X['val'], Y['val'], X['test'], Y['test'], D, V, max_len


def generate_physionet_mortality_dataset(data_path, output_dir, start_hour=[0], end_hour=[48], num_obs=0.99):
    # Read data.
    data, oc, train_ind, valid_ind, test_ind = pickle.load(open(data_path, 'rb'))
    # Filter labeled data in first 24h.
    data = data.loc[data.ts_ind.isin(np.concatenate((train_ind, valid_ind, test_ind), axis=-1))]

    filtering_cond = (data.hour>=start_hour[0])&(data.hour<=end_hour[0])
    for i in range(1,len(start_hour)):
        filtering_cond = (filtering_cond) | ((data.hour>=start_hour[i])&(data.hour<=end_hour[i]))
    data = data.loc[filtering_cond]
    print(f'Total: {data.ts_ind.nunique()}')

    # Getting patient_id due to filtering of time
    all_ind = data.ts_ind.to_numpy(copy=True)
    all_ind = np.unique(all_ind)
    train_ind = np.unique(np.intersect1d(all_ind, train_ind, assume_unique=False))
    valid_ind = np.unique(np.intersect1d(all_ind, valid_ind, assume_unique=False))
    test_ind = np.unique(np.intersect1d(all_ind, test_ind, assume_unique=False))

    oc = oc.loc[oc.ts_ind.isin(np.concatenate((train_ind, valid_ind, test_ind), axis=-1))]

    print(f'Train:{len(train_ind)} Val:{len(valid_ind)} Test:{len(test_ind)}')
    if np.max(all_ind)!=len(all_ind)-1 or np.min(all_ind)!=0: # Checking if we removed any samples due to time filtering
        
        new_ind_map = {old:new for old, new in zip(all_ind, [x for x in range(len(all_ind))])}
        train_ind = np.array([new_ind_map[train_ind[x]] for x in range(len(train_ind))])
        valid_ind = np.array([new_ind_map[valid_ind[x]] for x in range(len(valid_ind))])
        test_ind = np.array([new_ind_map[test_ind[x]] for x in range(len(test_ind))])
        
        data['ts_ind'] = data['ts_ind'].replace(new_ind_map)
        oc['ts_ind'] = oc['ts_ind'].replace(new_ind_map)

        all_ind = data.ts_ind.to_numpy(copy=True)
        all_ind = np.unique(all_ind)

    assert oc['ts_ind'].nunique() == len(all_ind)
    assert np.max(all_ind)==len(all_ind)-1 and np.min(all_ind)==0

    # Fix age.
    data.loc[(data.variable=='Age')&(data.value>200), 'value'] = 91.4
    # Get y and N.
    y = np.array(oc.sort_values(by='ts_ind')['in_hospital_mortality']).astype('float32')
    N = data.ts_ind.max() + 1
    # Get static data with mean fill and missingness indicator.
    static_varis = ['Age', 'Gender', 'Height', 'ICUType_1', 'ICUType_2', 'ICUType_3', 'ICUType_4']

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
    print(data.groupby('ts_ind')['hour'].count().quantile([0.25, 0.5, 0.75, 0.9, 0.99, 1]))

    if num_obs<=1:
        num_obs = int(data.groupby('ts_ind')['hour'].count().quantile([num_obs]).iloc[0])
        data = data.groupby('ts_ind').head(num_obs)
    else:
        data = data.groupby('ts_ind').head(num_obs)

    # Get N, V, var_to_ind.
    N = data.ts_ind.max() + 1
    varis = sorted(list(set(data.variable)))
    print(varis)
    V = len(varis)
    def inv_list(l, start=0):
        d = {}
        for i in range(len(l)):
            d[l[i]] = i+start
        return d
    var_to_ind = inv_list(varis, start=1)
    print(var_to_ind)
    data = data[data['variable'].isin(varis)]
    data['vind'] = data.variable.map(var_to_ind)
    print(data['vind'].nunique())

    data = data[['ts_ind', 'vind', 'hour', 'value']].sort_values(by=['ts_ind', 'vind', 'hour'])
    # Add obs index.
    data = data.sort_values(by=['ts_ind']).reset_index(drop=True)
    data = data.reset_index().rename(columns={'index':'obs_ind'})
    data = data.merge(data.groupby('ts_ind').agg({'obs_ind':'min'}).reset_index().rename(columns={ \
                                                                'obs_ind':'first_obs_ind'}), on='ts_ind')
    data['obs_ind'] = data['obs_ind'] - data['first_obs_ind']
    # Find max_len.
    max_len = data.obs_ind.max()+1
    print ('max_len', max_len)
    # Generate times_ip and values_ip matrices.
    times_inp = np.zeros((N, max_len), dtype='float32')
    values_inp = np.zeros((N, max_len), dtype='float32')
    varis_inp = np.zeros((N, max_len), dtype='int32')
    for row in tqdm(data.itertuples()):
        ts_ind = row.ts_ind
        l = row.obs_ind
        times_inp[ts_ind, l] = row.hour
        values_inp[ts_ind, l] = row.value
        varis_inp[ts_ind, l] = row.vind
    data.drop(columns=['obs_ind', 'first_obs_ind'], inplace=True)
    # Generate 3 sets of inputs and outputs.
    train_ip = [ip[train_ind] for ip in [demo, times_inp, values_inp, varis_inp]]
    valid_ip = [ip[valid_ind] for ip in [demo, times_inp, values_inp, varis_inp]]
    test_ip = [ip[test_ind] for ip in [demo, times_inp, values_inp, varis_inp]]
    del times_inp, values_inp, varis_inp
    train_op = y[train_ind]
    valid_op = y[valid_ind]
    test_op = y[test_ind]
    del y
    print('TS Hour Max, Min')
    print(np.max(train_ip[1][train_ip[3]!=0]), np.min(train_ip[1][train_ip[3]!=0]))
    print(np.max(valid_ip[1][valid_ip[3]!=0]), np.min(valid_ip[1][valid_ip[3]!=0]))
    print(np.max(test_ip[1][test_ip[3]!=0]), np.min(test_ip[1][test_ip[3]!=0]))
    assert np.max(train_ip[1][train_ip[3]!=0])<=max(end_hour) and np.min(train_ip[1][train_ip[3]!=0])>=min(start_hour)
    assert np.max(valid_ip[1][valid_ip[3]!=0])<=max(end_hour) and np.min(valid_ip[1][valid_ip[3]!=0])>=min(start_hour)
    assert np.max(test_ip[1][test_ip[3]!=0])<=max(end_hour) and np.min(test_ip[1][test_ip[3]!=0])>=min(start_hour)

    os.makedirs(output_dir, exist_ok=True)

    with gzip.GzipFile(f'{output_dir}/train_demos.npy.gz', 'w') as f:
        np.save(f, train_ip[0])
    with gzip.GzipFile(f'{output_dir}/train_times.npy.gz', 'w') as f:
        np.save(f, train_ip[1])
    with gzip.GzipFile(f'{output_dir}/train_values.npy.gz', 'w') as f:
        np.save(f, train_ip[2])
    with gzip.GzipFile(f'{output_dir}/train_varis.npy.gz', 'w') as f:
        np.save(f, train_ip[3])
    with gzip.GzipFile(f'{output_dir}/train_y.npy.gz', 'w') as f:
        np.save(f, train_op)

    with gzip.GzipFile(f'{output_dir}/val_demos.npy.gz', 'w') as f:
        np.save(f, valid_ip[0])
    with gzip.GzipFile(f'{output_dir}/val_times.npy.gz', 'w') as f:
        np.save(f, valid_ip[1])
    with gzip.GzipFile(f'{output_dir}/val_values.npy.gz', 'w') as f:
        np.save(f, valid_ip[2])
    with gzip.GzipFile(f'{output_dir}/val_varis.npy.gz', 'w') as f:
        np.save(f, valid_ip[3])
    with gzip.GzipFile(f'{output_dir}/val_y.npy.gz', 'w') as f:
        np.save(f, valid_op)

    with gzip.GzipFile(f'{output_dir}/test_demos.npy.gz', 'w') as f:
        np.save(f, test_ip[0])
    with gzip.GzipFile(f'{output_dir}/test_times.npy.gz', 'w') as f:
        np.save(f, test_ip[1])
    with gzip.GzipFile(f'{output_dir}/test_values.npy.gz', 'w') as f:
        np.save(f, test_ip[2])
    with gzip.GzipFile(f'{output_dir}/test_varis.npy.gz', 'w') as f:
        np.save(f, test_ip[3])
    with gzip.GzipFile(f'{output_dir}/test_y.npy.gz', 'w') as f:
        np.save(f, test_op)

    params = {'V': V, 'D': D, 'max_len': int(max_len)}
    print(params)

    with open(f'{output_dir}/extra_params.json', 'w') as f:
        json.dump(params, f)


def generate_mimic_mortality_dataset(data_path, output_dir):
    modes = ['train', 'val', 'test']
    var_to_id = {}
    ts_len = []
    dataset_samples = {}
    D = 2
    CATEGORICAL_COL = {
        'Glascow coma scale eye opening': {
            'To Pain': 2.0,
            'To Speech': 3.0,
            '2 To pain': 2.0,
            'Spontaneously': 4.0,
            '1 No Response': 1.0,
            'None': 1.0,
            '4 Spontaneously': 4.0,
            '3 To speech': 3.0
        },
        'Glascow coma scale motor response': {
            'Abnormal Flexion': 3.0,
            'Localizes Pain': 5.0,
            'Obeys Commands': 6.0,
            '2 Abnorm extensn': 2.0,
            '1 No Response': 1.0,
            'Flex-withdraws': 4.0,
            '4 Flex-withdraws': 4.0,
            '6 Obeys Commands': 6.0,
            '5 Localizes Pain': 5.0,
            '3 Abnorm flexion': 3.0,
            'Abnormal extension': 2.0,
            'No response': 1.0
        },
        'Glascow coma scale verbal response': {
            '3 Inapprop words': 3.0,
            'Inappropriate Words': 3.0,
            '2 Incomp sounds': 2.0,
            '1 No Response': 1.0,
            'Confused': 4.0,
            '4 Confused': 4.0,
            'Incomprehensible sounds': 2.0,
            'No Response-ETT': 1.0,
            'No Response': 1.0,
            '1.0 ET/Trach': 1.0,
            '5 Oriented': 5.0,
            'Oriented': 5.0
        },
    }

    static_, times_, values_, varis_, y_ = [], [], [], [], []
    for mode in modes: # for each dataset
        ind = pd.read_csv(f'{data_path}/in-hospital-mortality-24/{mode}_listfile.csv')
        dataset_samples[mode] = len(ind)

        for (i, ind_row) in tqdm(ind.iterrows()): # for each sample
            filename, y = ind_row['stay'], ind_row['y_true']
            patient_id = filename.split('_')[0]
            eps_id = filename.split('_')[1]
            y_.append(y)

            ts = pd.read_csv(f'{data_path}/in-hospital-mortality-24/{"train" if mode=="val" else mode}/{filename}')
            static = pd.read_csv(f'{data_path}/root/{"train" if mode=="val" else mode}/{patient_id}/{eps_id}.csv').iloc[0]
            static_.append([static['Age'], static['Gender']])

            if len(var_to_id)==0:
                columns = set(ts.columns)
                columns.remove('Hours')
                columns = sorted(list(columns))
                var_to_id = {k:x+1 for x,k in enumerate(columns)}
                print(var_to_id)
                V = len(var_to_id)
            times, values, varis = [], [], []
            for (j, row) in ts.iterrows(): # for each timestamp in ts

                for col, id in var_to_id.items(): # for each variable
                    if isinstance(row[col], float) and math.isnan(row[col]):
                        continue
                    if col in CATEGORICAL_COL:
                        values.append(int(CATEGORICAL_COL[col][row[col]]))
                    else:
                        values.append(float(row[col]))
                    times.append(float(row['Hours']))
                    varis.append(int(id))
            
            times = np.array(times);values = np.array(values);varis = np.array(varis);ts_len.append(len(times))
            assert values.dtype in ['int32', 'float32', 'int64', 'float64']
            times_.append(times);values_.append(values);varis_.append(varis)

    ts_len = np.array(ts_len)
    print(np.quantile(ts_len, [0.25, 0.5, 0.75, 0.9, 0.99, 1]))
    print(f'Num Samples: {len(times_)}')
    max_len = int(np.quantile(ts_len, 0.99))
    # max_len = max(ts_len)
    print(f'Max Length: {max_len}')


    for i in tqdm(range(sum(dataset_samples.values()))): # Pad all ts to max_len
        length = len(times_[i])
        if length<max_len:
            times_[i] = np.pad(times_[i], pad_width=(0, max_len-length))
            values_[i] = np.pad(values_[i], pad_width=(0, max_len-length))
            varis_[i] = np.pad(varis_[i], pad_width=(0, max_len-length))
        elif max_len<length:
            times_[i] = times_[i][:max_len]
            values_[i] = values_[i][:max_len]
            varis_[i] = varis_[i][:max_len]
    
    times_ = np.vstack(times_);values_ = np.vstack(values_);varis_ = np.vstack(varis_);static_ = np.vstack(static_)
    
    for v in range(1, np.max(varis_)+1): # Normalise
        mean = np.mean(values_[varis_==v])        
        std = np.std(values_[varis_==v])
        std = 1 if std==0 else std
        values_[varis_==v] = (values_[varis_==v]-mean) / std
        assert int(np.mean(values_[varis_==v]))==0 and (int(np.std(values_[varis_==v]))==1 or int(np.std(values_[varis_==v]))==0)

    for i in range(static_.shape[-1]):
        print(i)
        mean = np.mean(static_[:,i])
        std = np.std(static_[:,i])
        std = 1 if np.std(static_[:,i])==0 else std
        static_[:,i] = (static_[:,i] - mean)/std
        print(np.mean(static_[:,i]), np.std(static_[:,i]))
        assert int(np.mean(static_[:,i]))==0 and (int(np.std(static_[:,i]))==1 or int(np.std(static_[:,i]))==0)
    

    start = 0
    for mode in modes: #Saving dataset
        static = static_[start: start+dataset_samples[mode]]
        times = times_[start: start+dataset_samples[mode]]
        values = values_[start: start+dataset_samples[mode]]
        varis = varis_[start: start+dataset_samples[mode]]
        y = np.array(y_[start: start+dataset_samples[mode]])
        start += dataset_samples[mode]

        print(f'{mode}: Time: {times.shape} Values: {values.shape} Varis: {varis.shape} Static: {static.shape} y: {y.shape}')
        
        os.makedirs(output_dir, exist_ok=True)

        with gzip.GzipFile(f'{output_dir}/{mode}_demos.npy.gz', 'w') as f:
            np.save(f, static)
        with gzip.GzipFile(f'{output_dir}/{mode}_times.npy.gz', 'w') as f:
            np.save(f, times)
        with gzip.GzipFile(f'{output_dir}/{mode}_values.npy.gz', 'w') as f:
            np.save(f, values)
        with gzip.GzipFile(f'{output_dir}/{mode}_varis.npy.gz', 'w') as f:
            np.save(f, varis)
        with gzip.GzipFile(f'{output_dir}/{mode}_y.npy.gz', 'w') as f:
            np.save(f, y)
            
    params = {'V': V, 'D': D, 'max_len': int(max_len)}
    print(params)

    with open(f'{output_dir}/extra_params.json', 'w') as f:
        json.dump(params, f)


def load_forecast_dataset(path, with_demo=True):
    train_ip, valid_ip = [], []

    if with_demo:
        with gzip.GzipFile(f'{path}/train_demos.npy.gz', 'r') as f:
            train_ip.append(np.load(f))
    with gzip.GzipFile(f'{path}/train_times.npy.gz', 'r') as f:
        train_ip.append(np.load(f))
    with gzip.GzipFile(f'{path}/train_values.npy.gz', 'r') as f:
        train_ip.append(np.load(f))
    with gzip.GzipFile(f'{path}/train_varis.npy.gz', 'r') as f:
        train_ip.append(np.load(f))
    with gzip.GzipFile(f'{path}/train_y.npy.gz', 'r') as f:
        train_op = np.load(f)

    if with_demo:
        with gzip.GzipFile(f'{path}/val_demos.npy.gz', 'r') as f:
            valid_ip.append(np.load(f))
    with gzip.GzipFile(f'{path}/val_times.npy.gz', 'r') as f:
        valid_ip.append(np.load(f))
    with gzip.GzipFile(f'{path}/val_values.npy.gz', 'r') as f:
        valid_ip.append(np.load(f))
    with gzip.GzipFile(f'{path}/val_varis.npy.gz', 'r') as f:
        valid_ip.append(np.load(f))
    with gzip.GzipFile(f'{path}/val_y.npy.gz', 'r') as f:
        valid_op = np.load(f)
    
    with open(f'{path}/extra_params.json') as f:
        params = json.load(f)
    
    V = int(params['V'])
    D = int(params['D'])
    max_len = int(params['max_len'])

    return train_ip, train_op, valid_ip, valid_op, D, V, max_len


def generate_physionet_forecast_dataset(data_path, output_dir, num_obs=0.99):
    pred_window = 2 # hours
    obs_windows = range(12, 44, 4)

    # Read data.
    data, oc, train_ind, valid_ind, test_ind = pickle.load(open(data_path, 'rb'))
    # Remove test patients.
    # data = data.merge(oc[['ts_ind', 'SUBJECT_ID']], on='ts_ind', how='left')
    test_sub = oc.loc[oc.ts_ind.isin(test_ind)].ts_ind.unique()
    data = data.loc[~data.ts_ind.isin(test_sub)]
    oc = oc.loc[~oc.ts_ind.isin(test_sub)]

    print(f'Total: {data["ts_ind"].nunique()} ({oc["ts_ind"].nunique()})')

    assert data['ts_ind'].nunique() == oc['ts_ind'].nunique()
    assert len(train_ind) + len(valid_ind) == data['ts_ind'].nunique()

    # Fix age.
    data.loc[(data.variable=='Age')&(data.value>200), 'value'] = 91.4
    # Get static data with mean fill and missingness indicator.

    static_varis = ['Age', 'Gender', 'Height', 'ICUType_1', 'ICUType_2', 'ICUType_3', 'ICUType_4']

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
    N = data.ts_ind.max()+1
    demo = np.zeros((N, D))
    for row in tqdm(static_data.itertuples()):
        demo[row.ts_ind, static_var_to_ind[row.variable]] = row.value
    # Normalize static data.
    means = demo.mean(axis=0, keepdims=True)
    stds = demo.std(axis=0, keepdims=True)
    stds = (stds==0)*1 + (stds!=0)*stds
    demo = (demo-means)/stds
    # Get variable indices.
    varis = sorted(list(set(data.variable)))
    V = len(varis)
    print(f'V: {V}')
    var_to_ind = inv_list(varis, start=1)
    data['vind'] = data.variable.map(var_to_ind)
    data = data[['ts_ind', 'vind', 'hour', 'value']].sort_values(by=['ts_ind', 'vind', 'hour'])
    # Find max_len.
    if num_obs<=1:
        num_obs = int(data.groupby('ts_ind')['hour'].count().quantile([num_obs]).iloc[0])
        data = data.groupby('ts_ind').head(num_obs)
    else:
        data = data.groupby('ts_ind').head(num_obs)

    print(f'Fore Max Len: {num_obs}')
    fore_max_len = num_obs
    # Get forecast inputs and outputs.
    fore_times_ip = []
    fore_values_ip = []
    fore_varis_ip = []
    fore_obs_window = []
    fore_op = []
    fore_inds = []
    def f(x):
        mask = [0 for i in range(V)]
        values = [0 for i in range(V)]
        for vv in x:
            v = int(vv[0])-1
            mask[v] = 1
            values[v] = vv[1]
        return values+mask
    
    def pad(x):
        return x+[0]*(fore_max_len-len(x))
    
    for w in tqdm(obs_windows): 
        pred_data = data.loc[(data.hour>=w)&(data.hour<=w+pred_window)]
        pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value':'first'}).reset_index()
        pred_data['vind_value'] = pred_data[['vind', 'value']].values.tolist()
        pred_data = pred_data.groupby('ts_ind').agg({'vind_value':list}).reset_index()
        pred_data['vind_value'] = pred_data['vind_value'].apply(f)    
        obs_data = data.loc[(data.hour>=0)&(data.hour<w)]
        obs_data = obs_data.loc[obs_data.ts_ind.isin(pred_data.ts_ind)]
        obs_data = obs_data.groupby('ts_ind').head(fore_max_len)
        obs_data = obs_data.groupby('ts_ind').agg({'vind':list, 'hour':list, 'value':list}).reset_index()
        obs_data = obs_data.merge(pred_data, on='ts_ind')
        obs_data['obs_window_end'] = w

        for col in ['vind', 'hour', 'value']:
            obs_data[col] = obs_data[col].apply(pad)
        fore_op.append(np.array(list(obs_data.vind_value)))
        fore_inds.append(np.array(list(obs_data.ts_ind)))
        fore_times_ip.append(np.array(list(obs_data.hour)))
        fore_values_ip.append(np.array(list(obs_data.value)))
        fore_varis_ip.append(np.array(list(obs_data.vind)))
        fore_obs_window.append(np.array(list(obs_data.obs_window_end)))
    del data
    fore_times_ip = np.concatenate(fore_times_ip, axis=0)
    fore_values_ip = np.concatenate(fore_values_ip, axis=0)
    fore_varis_ip = np.concatenate(fore_varis_ip, axis=0)
    fore_obs_window = np.concatenate(fore_obs_window, axis=0)
    fore_op = np.concatenate(fore_op, axis=0)
    fore_inds = np.concatenate(fore_inds, axis=0)
    fore_demo = demo[fore_inds]

    print(f'Final shape: {fore_times_ip.shape}, {fore_values_ip.shape}, {fore_varis_ip.shape}, {fore_obs_window.shape}, {fore_demo.shape}, {fore_inds.shape}, {fore_op.shape}')

    # Get train and valid ts_ind for forecast task.
    train_sub = oc.loc[oc.ts_ind.isin(train_ind)].ts_ind.unique()
    valid_sub = oc.loc[oc.ts_ind.isin(valid_ind)].ts_ind.unique()
    rem_sub = oc.loc[~oc.ts_ind.isin(np.concatenate((train_ind, valid_ind)))].ts_ind.unique()
    print(f'Rem sub: {rem_sub}')
    bp = int(0.8*len(rem_sub))
    train_sub = np.concatenate((train_sub, rem_sub[:bp]))
    valid_sub = np.concatenate((valid_sub, rem_sub[bp:]))
    train_ind = oc.loc[oc.ts_ind.isin(train_sub)].ts_ind.unique() # Add remaining ts_ind s of train subjects.
    valid_ind = oc.loc[oc.ts_ind.isin(valid_sub)].ts_ind.unique() # Add remaining ts_ind s of train subjects.

    # Generate 3 sets of inputs and outputs.
    train_ind = np.argwhere(np.in1d(fore_inds, train_ind)).flatten()
    valid_ind = np.argwhere(np.in1d(fore_inds, valid_ind)).flatten()
    fore_train_ip = [ip[train_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip, fore_obs_window]]
    fore_valid_ip = [ip[valid_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip, fore_obs_window]]
    del fore_times_ip, fore_values_ip, fore_varis_ip, demo, fore_demo
    fore_train_op = fore_op[train_ind]
    fore_valid_op = fore_op[valid_ind]
    del fore_op

    os.makedirs(output_dir, exist_ok=True)
    with gzip.GzipFile(f'{output_dir}/train_demos.npy.gz', 'w') as f:
        np.save(f, fore_train_ip[0])
    with gzip.GzipFile(f'{output_dir}/train_times.npy.gz', 'w') as f:
        np.save(f, fore_train_ip[1])
    with gzip.GzipFile(f'{output_dir}/train_values.npy.gz', 'w') as f:
        np.save(f, fore_train_ip[2])
    with gzip.GzipFile(f'{output_dir}/train_varis.npy.gz', 'w') as f:
        np.save(f, fore_train_ip[3])
    with gzip.GzipFile(f'{output_dir}/train_obs_window.npy.gz', 'w') as f:
        np.save(f, fore_train_ip[4])
    with gzip.GzipFile(f'{output_dir}/train_y.npy.gz', 'w') as f:
        np.save(f, fore_train_op)

    with gzip.GzipFile(f'{output_dir}/val_demos.npy.gz', 'w') as f:
        np.save(f, fore_valid_ip[0])
    with gzip.GzipFile(f'{output_dir}/val_times.npy.gz', 'w') as f:
        np.save(f, fore_valid_ip[1])
    with gzip.GzipFile(f'{output_dir}/val_values.npy.gz', 'w') as f:
        np.save(f, fore_valid_ip[2])
    with gzip.GzipFile(f'{output_dir}/val_varis.npy.gz', 'w') as f:
        np.save(f, fore_valid_ip[3])
    with gzip.GzipFile(f'{output_dir}/val_obs_window.npy.gz', 'w') as f:
        np.save(f, fore_valid_ip[4])
    with gzip.GzipFile(f'{output_dir}/val_y.npy.gz', 'w') as f:
        np.save(f, fore_valid_op)
    
    params = {'V': V, 'D': D, 'max_len': int(fore_max_len)}
    print(params)

    with open(f'{output_dir}/extra_params.json', 'w') as f:
        json.dump(params, f)
    

def load_mtand_mortality_dataset(path, max_time, with_demo=True, len_time_query=48):

    X = {'train': [], 'val': [], 'test': []}
    Y = {'train': [], 'val': [], 'test': []}

    for mode in ['train', 'val', 'test']:
        if with_demo:
            with gzip.GzipFile(f'{path}/{mode}_demos.npy.gz', 'r') as f:
                X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_times.npy.gz', 'r') as f:
            X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_values.npy.gz', 'r') as f:
            X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_varis.npy.gz', 'r') as f:
            X[mode].append(np.load(f))


        with gzip.GzipFile(f'{path}/{mode}_flatten_times.npy.gz', 'r') as f:
            X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_feature_matrices.npy.gz', 'r') as f:
            X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_mask_matrices.npy.gz', 'r') as f:
            X[mode].append(np.load(f))
        
        dataset_len = X[mode][-1].shape[0]
        X[mode].append(np.stack([np.linspace(0, max_time, len_time_query)]*dataset_len, axis=0))

        with gzip.GzipFile(f'{path}/{mode}_y.npy.gz', 'r') as f:
            Y[mode] = np.load(f)

    with open(f'{path}/extra_params.json') as f:
        params = json.load(f)
    
    V = int(params['V'])
    D = int(params['D'])
    max_len = int(params['max_len'])

    len_time_key = X['train'][-3].shape[-2]
    

    return X['train'], Y['train'], X['val'], Y['val'], X['test'], Y['test'], D, V, max_len, len_time_key


def load_mtand_forecast_dataset(path, with_demo=True, len_time_query=48):
    X = {'train': [], 'val': []}
    Y = {'train': [], 'val': []}

    for mode in ['train', 'val']:
        if with_demo:
            with gzip.GzipFile(f'{path}/{mode}_demos.npy.gz', 'r') as f:
                X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_times.npy.gz', 'r') as f:
            X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_values.npy.gz', 'r') as f:
            X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_varis.npy.gz', 'r') as f:
            X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_obs_window.npy.gz', 'r') as f:
            obs_window = np.load(f)


        with gzip.GzipFile(f'{path}/{mode}_flatten_times.npy.gz', 'r') as f:
            X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_feature_matrices.npy.gz', 'r') as f:
            X[mode].append(np.load(f))
        with gzip.GzipFile(f'{path}/{mode}_mask_matrices.npy.gz', 'r') as f:
            X[mode].append(np.load(f))

        X[mode].append(np.stack([np.linspace(0, obs_window_end, len_time_query) for obs_window_end in obs_window], axis=0))

        with gzip.GzipFile(f'{path}/{mode}_y.npy.gz', 'r') as f:
            Y[mode] = np.load(f)

    with open(f'{path}/extra_params.json') as f:
        params = json.load(f)
    
    V = int(params['V'])
    D = int(params['D'])
    max_len = int(params['max_len'])

    len_time_key = X['train'][-3].shape[-2]
    

    return X['train'], Y['train'], X['val'], Y['val'], D, V, max_len, len_time_key


def generate_mtand_dataset(data_path, output_dir, start_hour=0.0, end_hour=48.0, percentile_timestep=100, type='mortality'):
    X = {'train': [], 'val': [], 'test': []}
    if type=='mortality':
        modes = ['train', 'val', 'test']
        X['train'], _, X['val'], _, X['test'], _, D, V, max_len = load_mortality_dataset(data_path, with_demo=False, with_imputation=False)
    elif type=='forecast':
        modes = ['train', 'val']
        X['train'], _, X['val'], _, D, V, max_len = load_forecast_dataset(data_path, with_demo=False)
    else:
        print('Type must be forecast or mortality')

    new_X = {'train': [], 'val': [], 'test': []}

    # Converting time in matrices
    for mode in modes:
        times, values, varis = X[mode]
        print('raw_inputs')
        print(times.shape)
        print(values.shape)
        print(varis.shape)
        flatten_time, feature_matrices, mask_matrics = [], [],[]
        for i in tqdm(range(times.shape[0])):
            cur_times = times[i]
            cur_values = values[i]
            cur_varis = varis[i]
            # Round to nearest minute as in the mTAND paper
            cur_times *= 60
            cur_times = np.round(cur_times)
            cur_times /= 60
            
            # Remove masked observations
            mask_index = cur_varis!=0
            cur_times = cur_times[mask_index]
            cur_values = cur_values[mask_index]
            cur_varis = cur_varis[mask_index]
            assert (cur_varis!=0).all()

            # Filter for time
            time_filter = (cur_times>=start_hour)&(cur_times<=end_hour)
            cur_times = cur_times[time_filter]
            cur_values = cur_values[time_filter]
            cur_varis = cur_varis[time_filter]
            # Data with no time series
            if len(cur_times)==0:
                flatten_time.append(np.zeros(1,))
                feature_matrices.append(np.zeros(shape=(1, V)))
                mask_matrics.append(np.zeros(shape=(1, V)))
                continue
            else:
                assert (np.min(cur_times)>=start_hour and np.max(cur_times)<=end_hour)

            # Sorting according to time
            time_index = np.argsort(cur_times)
            cur_times = cur_times[time_index]
            cur_values = cur_values[time_index]
            cur_varis = cur_varis[time_index]
            assert cur_times.shape==cur_varis.shape and cur_times.shape==cur_values.shape
            
            timesteps = np.unique(cur_times)
            timesteps = np.sort(timesteps)
            assert np.min(timesteps)>=start_hour and np.max(timesteps)<=end_hour

            matrix = np.zeros(shape=(timesteps.shape[0], V))
            mask = np.zeros(shape=(timesteps.shape[0], V))

            double_count = 0
            # For each timestep
            for j in range(cur_times.shape[0]):
                # Check if there is only 1 row index in timesteps
                assert len(np.where(timesteps==cur_times[j])[0])==1
                row = np.where(timesteps==cur_times[j])[0][0]
                col = cur_varis[j]-1
                matrix[row][col] = cur_values[j]
                if mask[row][col]==1:
                    double_count+=1
                else:
                    mask[row][col] = 1

            assert np.sum(mask)+double_count==len(cur_varis)
            # Make sure all timesteps has at least 1 recording
            assert np.sum(mask,axis=-1).all()

            flatten_time.append(timesteps)
            feature_matrices.append(matrix)
            mask_matrics.append(mask)

        new_X[mode] = [flatten_time, feature_matrices, mask_matrics]
        print(f'{mode} length: {len(new_X[mode][0])}')
        
    len_timesteps = []
    for mode in modes:
        flatten_time = new_X[mode][0]
        len_timesteps += [len(x) for x in flatten_time]
    print(f'Timestamp length: {np.percentile(np.array(len_timesteps), [25,50,75,90,99,100])}')

    max_len_timesteps = int(np.percentile(np.array(len_timesteps), [percentile_timestep]))

    # Padding all samples to same length
    for mode in modes:
        flatten_time, feature_matrices, mask_matrics = new_X[mode]

        for i in range(len(flatten_time)):
            len_timesteps = flatten_time[i].shape[0]
            if len_timesteps>max_len_timesteps:
                flatten_time[i] = flatten_time[i][:max_len_timesteps]
                feature_matrices[i] = feature_matrices[i][:max_len_timesteps, :]
                mask_matrics[i] = mask_matrics[i][:max_len_timesteps, :]
            elif len_timesteps<max_len_timesteps:
                flatten_time[i] = np.pad(flatten_time[i], (0,max_len_timesteps-len_timesteps))
                feature_matrices[i] = np.pad(feature_matrices[i], ((0,max_len_timesteps-len_timesteps),(0,0)))
                mask_matrics[i] = np.pad(mask_matrics[i], ((0,max_len_timesteps-len_timesteps), (0,0)))
            
            feature_matrices[i] = np.expand_dims(feature_matrices[i], axis=0)
            mask_matrics[i] = np.expand_dims(mask_matrics[i], axis=0)
        
        flatten_time = np.vstack(flatten_time)
        feature_matrices = np.vstack(feature_matrices)
        mask_matrics = np.vstack(mask_matrics)

        assert np.min(flatten_time[np.any(mask_matrics!=0, axis=-1)])>=start_hour and np.max(flatten_time[np.any(mask_matrics!=0, axis=-1)])<=end_hour

        new_X[mode] = [flatten_time, feature_matrices, mask_matrics]

        print(f'Final Shape: {flatten_time.shape}, {feature_matrices.shape}, {mask_matrics.shape}')

        os.makedirs(output_dir, exist_ok=True)
        with gzip.GzipFile(f'{output_dir}/{mode}_flatten_times.npy.gz', 'w') as f:
            np.save(f, new_X[mode][0])
        with gzip.GzipFile(f'{output_dir}/{mode}_feature_matrices.npy.gz', 'w') as f:
            np.save(f, new_X[mode][1])
        with gzip.GzipFile(f'{output_dir}/{mode}_mask_matrices.npy.gz', 'w') as f:
            np.save(f, new_X[mode][2])


def generate_missing_dataset(data_path, output_dir, hour_time_diff=0.5):
    X = {'train': [], 'val': [], 'test': []}
    Y = {}
    X['train'], Y['train'], X['val'], Y['val'], X['test'], Y['test'], D, V, max_len = load_mortality_dataset(data_path, with_demo=True, with_imputation=False)

    new_X = {}

    max_len = 0
    for mode in ['train', 'val', 'test']:
        demo, times, values, varis = X[mode]
        new_times, new_values, new_varis = [], [], []

        all_time_diff = []
        for i in range(len(demo)):
            cur_time = times[i]
            cur_varis = varis[i]
            cur_values = values[i]
            mask = cur_varis!=0

            cur_time = cur_time[mask]
            cur_varis = cur_varis[mask]
            cur_values = cur_values[mask]
            unique_time = np.unique(cur_time)

            if len(unique_time)>1: # For cases where only 1 timestamp was observed
                unique_time = np.sort(unique_time)
                time_diff = np.diff(unique_time, n=1)
                # Filter observations that were taken within `hour_time_diff`
                unique_time = np.concatenate((unique_time[0:1], unique_time[1:][time_diff>=hour_time_diff]), axis=-1)
                matches = np.in1d(cur_time, unique_time)
                cur_time = cur_time[matches]
                cur_varis = cur_varis[matches]
                cur_values = cur_values[matches]

                unique_time = np.sort(unique_time)
                time_diff = np.diff(unique_time, n=1)

                # If the # timestamp after filtering is 0, then the time diff is 0
                if len(unique_time)>1:
                    all_time_diff.append(np.mean(time_diff))
            
            new_times.append(cur_time); new_varis.append(cur_varis); new_values.append(cur_values)
            max_len = max(max_len, cur_time.shape[0])

        print(f'{mode} avg patient observation interval: {np.percentile(all_time_diff,[25,50,75])}')
        
        new_X[mode] = [demo, new_times, new_values, new_varis]
    
    print(f'Max Len: {max_len}')
    os.makedirs(output_dir, exist_ok=True)
    for mode in ['train', 'val', 'test']:
        demo, times, values, varis = new_X[mode]
        new_times, new_values, new_varis = [], [], []

        for i in range(len(demo)):
            cur_time = times[i]
            cur_varis = varis[i]
            cur_values = values[i]
            cur_len = cur_time.shape[0]
            cur_time = np.pad(cur_time, (0, max_len-cur_len))
            cur_varis = np.pad(cur_varis, (0, max_len-cur_len))
            cur_values = np.pad(cur_values, (0, max_len-cur_len))
        
            new_times.append(cur_time); new_varis.append(cur_varis); new_values.append(cur_values)
        
        new_X[mode] = [demo, np.vstack(new_times), np.vstack(new_values), np.vstack(new_varis)]

        print(f'{mode} Final Shape: {[new_X[mode][1].shape for x in range(1,4)]}')

        with gzip.GzipFile(f'{output_dir}/{mode}_demos.npy.gz', 'w') as f:
            np.save(f, new_X[mode][0])
        with gzip.GzipFile(f'{output_dir}/{mode}_times.npy.gz', 'w') as f:
            np.save(f, new_X[mode][1])
        with gzip.GzipFile(f'{output_dir}/{mode}_values.npy.gz', 'w') as f:
            np.save(f, new_X[mode][2])
        with gzip.GzipFile(f'{output_dir}/{mode}_varis.npy.gz', 'w') as f:
            np.save(f, new_X[mode][3])
        with gzip.GzipFile(f'{output_dir}/{mode}_y.npy.gz', 'w') as f:
            np.save(f, Y[mode])


    params = {'V': V, 'D': D, 'max_len': max_len}
    with open(f'{output_dir}/extra_params.json', 'w') as f:
        json.dump(params, f)
    

def generate_random_dataset(data_path, output_dir, random_drop=0.5):
    X = {'train': [], 'val': [], 'test': []}
    Y = {}
    X['train'], Y['train'], X['val'], Y['val'], X['test'], Y['test'], D, V, max_len = load_mortality_dataset(data_path, with_demo=True, with_imputation=False)

    new_X = {}

    max_len = 0
    for mode in ['train', 'val', 'test']:
        demo, times, values, varis = X[mode]
        new_times, new_values, new_varis = [], [], []

        all_time_diff = []
        for i in range(len(demo)):
            cur_time = times[i]
            cur_varis = varis[i]
            cur_values = values[i]
            mask = cur_varis!=0

            cur_time = cur_time[mask]
            cur_varis = cur_varis[mask]
            cur_values = cur_values[mask]
            unique_time = np.unique(cur_time)

            

            if len(unique_time)>1: # For cases where only 1 timestamp was observed
                num_time = max(1, int(((1-random_drop)*len(unique_time))))

                unique_time = np.random.choice(unique_time, size=num_time, replace=False)

                unique_time = np.sort(unique_time)
                time_diff = np.diff(unique_time, n=1)
                
                matches = np.in1d(cur_time, unique_time)
                cur_time = cur_time[matches]
                cur_varis = cur_varis[matches]
                cur_values = cur_values[matches]

                # If the # timestamp after filtering is 0, then the time diff is 0
                if len(unique_time)>1:
                    all_time_diff.append(np.mean(time_diff))
            
            new_times.append(cur_time); new_varis.append(cur_varis); new_values.append(cur_values)
            max_len = max(max_len, cur_time.shape[0])

        print(f'{mode} avg patient observation interval: {np.percentile(all_time_diff,[25,50,75])}')
        
        new_X[mode] = [demo, new_times, new_values, new_varis]
    
    print(f'Max Len: {max_len}')
    os.makedirs(output_dir, exist_ok=True)
    for mode in ['train', 'val', 'test']:
        demo, times, values, varis = new_X[mode]
        new_times, new_values, new_varis = [], [], []

        for i in range(len(demo)):
            cur_time = times[i]
            cur_varis = varis[i]
            cur_values = values[i]
            cur_len = cur_time.shape[0]
            cur_time = np.pad(cur_time, (0, max_len-cur_len))
            cur_varis = np.pad(cur_varis, (0, max_len-cur_len))
            cur_values = np.pad(cur_values, (0, max_len-cur_len))
        
            new_times.append(cur_time); new_varis.append(cur_varis); new_values.append(cur_values)
        
        new_X[mode] = [demo, np.vstack(new_times), np.vstack(new_values), np.vstack(new_varis)]

        print(f'{mode} Final Shape: {[new_X[mode][1].shape for x in range(1,4)]}')

        with gzip.GzipFile(f'{output_dir}/{mode}_demos.npy.gz', 'w') as f:
            np.save(f, new_X[mode][0])
        with gzip.GzipFile(f'{output_dir}/{mode}_times.npy.gz', 'w') as f:
            np.save(f, new_X[mode][1])
        with gzip.GzipFile(f'{output_dir}/{mode}_values.npy.gz', 'w') as f:
            np.save(f, new_X[mode][2])
        with gzip.GzipFile(f'{output_dir}/{mode}_varis.npy.gz', 'w') as f:
            np.save(f, new_X[mode][3])
        with gzip.GzipFile(f'{output_dir}/{mode}_y.npy.gz', 'w') as f:
            np.save(f, Y[mode])


    params = {'V': V, 'D': D, 'max_len': max_len}
    with open(f'{output_dir}/extra_params.json', 'w') as f:
        json.dump(params, f)
    




if __name__=='__main__':

    import argparse
    def parse_args():
        def list_of_ints(arg):
            return list(map(int, arg.split(',')))
        parser = argparse.ArgumentParser(description="")
        parser.add_argument(
            '--type', type=str, help='forecast or mortality or mtand or mtand_forecast or missing'
        )
        parser.add_argument(
            "--data_path", type=str, help="A path to dataset folder"
        )
        parser.add_argument(
            "--output_dir", type=str, help="A path to dataset folder"
        )
        parser.add_argument(
            "--start_hour", type=list_of_ints, default=[0], help="Start Time of dataset"
        )
        parser.add_argument(
            "--end_hour", type=list_of_ints, default=[48], help="End Time of dataset"
        )
        parser.add_argument(
            "--num_obs", type=float, default=None, help="No. of observations"
        )
        parser.add_argument(
            "--percentile_timestep", type=int, default=99, help="No. of timestep to take"
        )
        parser.add_argument(
            "--hour_time_diff", type=float, default=0.5, help="Minimum time difference between observations (in hours)"
        )
        parser.add_argument(
            "--random_drop", type=float, default=0.2, help="Proportion of unique timestamp to drop"
        )
        parser.add_argument(
            "--dataset", type=str, default='mimic', help="mimic or physionet_2012"
        )
        args = parser.parse_args()

        return args

    args = parse_args()
    if args.dataset=='physionet_2012':
        if args.type=='forecast':
            generate_physionet_forecast_dataset(
                data_path=args.data_path,
                output_dir=args.output_dir,
                num_obs=args.num_obs
            )
        elif args.type=='mortality':
            generate_physionet_mortality_dataset(
                output_dir=args.output_dir,
                data_path=args.data_path,
                start_hour=args.start_hour,
                end_hour=args.end_hour,
                num_obs=args.num_obs
            )
        elif args.type=='mtand':
            generate_mtand_dataset(
                output_dir=args.output_dir,
                data_path=args.data_path,
                percentile_timestep=args.percentile_timestep,
                type='mortality'
            )
        elif args.type=='mtand_forecast':
            generate_mtand_dataset(
                output_dir=args.output_dir,
                data_path=args.data_path,
                start_hour=args.start_hour,
                end_hour=args.end_hour,
                percentile_timestep=args.percentile_timestep,
                type='forecast'
            )
        elif args.type=='missing':
            generate_missing_dataset(
                output_dir=args.output_dir,
                data_path=args.data_path,
                hour_time_diff=args.hour_time_diff
            )
        elif args.type=='random':
            generate_random_dataset(
                output_dir=args.output_dir,
                data_path=args.data_path,
                random_drop=args.random_drop
            )
        else:
            raise NotImplementedError('Dataset not implemented')
    elif args.dataset=='mimic':
        if args.type=='mortality':
            generate_mimic_mortality_dataset(
                output_dir=args.output_dir,
                data_path=args.data_path
            )
        elif args.type=='mtand':
            generate_mtand_dataset(
                output_dir=args.output_dir,
                data_path=args.data_path,
                percentile_timestep=args.percentile_timestep,
                type='mortality'
            )
        else:
            raise NotImplementedError('Dataset not implemented')
    else:
        raise NotImplementedError('Dataset not implemented')