from torch.utils.data import Dataset
import torch
import pickle
import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

class MultipleInputsDataset(Dataset):
    def __init__(self, X_demos, X_times, X_values, X_varis, Y):
        super(MultipleInputsDataset, self).__init__()
        
        self.X_demos = torch.tensor(X_demos, dtype=torch.float32)
        self.X_times = torch.tensor(X_times, dtype=torch.float32)
        self.X_values = torch.tensor(X_values, dtype=torch.float32)
        self.X_varis = torch.tensor(X_varis, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __getitem__(self, idx):
        return self.X_demos[idx], self.X_times[idx], self.X_values[idx], self.X_varis[idx], self.Y[idx]
        
    def __len__(self):
        return len(self.Y)


def generate_forecast_dataset(data_path, output_dir):
    pred_window = 2 # hours
    obs_windows = range(20, 124, 4)

    # Read data.
    data, oc, train_ind, valid_ind, test_ind = pickle.load(open(data_path, 'rb'))
    # Remove test patients.
    data = data.merge(oc[['ts_ind', 'SUBJECT_ID']], on='ts_ind', how='left')
    test_sub = oc.loc[oc.ts_ind.isin(test_ind)].SUBJECT_ID.unique()
    data = data.loc[~data.SUBJECT_ID.isin(test_sub)]
    oc = oc.loc[~oc.SUBJECT_ID.isin(test_sub)]
    data.drop(columns=['SUBJECT_ID', 'TABLE'], inplace=True)
    # Fix age.
    data.loc[(data.variable=='Age')&(data.value>200), 'value'] = 91.4
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
    var_to_ind = inv_list(varis, start=1)
    data['vind'] = data.variable.map(var_to_ind)
    data = data[['ts_ind', 'vind', 'hour', 'value']].sort_values(by=['ts_ind', 'vind', 'hour'])
    # Find max_len.
    fore_max_len = 880
    # Get forecast inputs and outputs.
    fore_times_ip = []
    fore_values_ip = []
    fore_varis_ip = []
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
        obs_data = data.loc[(data.hour<w)&(data.hour>=w-24)]
        obs_data = obs_data.loc[obs_data.ts_ind.isin(pred_data.ts_ind)]
        obs_data = obs_data.groupby('ts_ind').head(fore_max_len)
        obs_data = obs_data.groupby('ts_ind').agg({'vind':list, 'hour':list, 'value':list}).reset_index()
        obs_data = obs_data.merge(pred_data, on='ts_ind')
        for col in ['vind', 'hour', 'value']:
            obs_data[col] = obs_data[col].apply(pad)
        fore_op.append(np.array(list(obs_data.vind_value)))
        fore_inds.append(np.array(list(obs_data.ts_ind)))
        fore_times_ip.append(np.array(list(obs_data.hour)))
        fore_values_ip.append(np.array(list(obs_data.value)))
        fore_varis_ip.append(np.array(list(obs_data.vind)))
    del data
    fore_times_ip = np.concatenate(fore_times_ip, axis=0)
    fore_values_ip = np.concatenate(fore_values_ip, axis=0)
    fore_varis_ip = np.concatenate(fore_varis_ip, axis=0)
    fore_op = np.concatenate(fore_op, axis=0)
    fore_inds = np.concatenate(fore_inds, axis=0)
    fore_demo = demo[fore_inds]
    # Get train and valid ts_ind for forecast task.
    train_sub = oc.loc[oc.ts_ind.isin(train_ind)].SUBJECT_ID.unique()
    valid_sub = oc.loc[oc.ts_ind.isin(valid_ind)].SUBJECT_ID.unique()
    rem_sub = oc.loc[~oc.SUBJECT_ID.isin(np.concatenate((train_ind, valid_ind)))].SUBJECT_ID.unique()
    bp = int(0.8*len(rem_sub))
    train_sub = np.concatenate((train_sub, rem_sub[:bp]))
    valid_sub = np.concatenate((valid_sub, rem_sub[bp:]))
    train_ind = oc.loc[oc.SUBJECT_ID.isin(train_sub)].ts_ind.unique() # Add remaining ts_ind s of train subjects.
    valid_ind = oc.loc[oc.SUBJECT_ID.isin(valid_sub)].ts_ind.unique() # Add remaining ts_ind s of train subjects.
    # Generate 3 sets of inputs and outputs.
    train_ind = np.argwhere(np.in1d(fore_inds, train_ind)).flatten()
    valid_ind = np.argwhere(np.in1d(fore_inds, valid_ind)).flatten()
    fore_train_ip = [ip[train_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]]
    fore_valid_ip = [ip[valid_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]]
    del fore_times_ip, fore_values_ip, fore_varis_ip, demo, fore_demo
    fore_train_op = fore_op[train_ind]
    fore_valid_op = fore_op[valid_ind]
    del fore_op

    with gzip.GzipFile(f'{output_dir}/train_demos.npy.gz', 'w') as f:
        np.save(f, fore_train_ip[0])
    with gzip.GzipFile(f'{output_dir}/train_times.npy.gz', 'w') as f:
        np.save(f, fore_train_ip[1])
    with gzip.GzipFile(f'{output_dir}/train_values.npy.gz', 'w') as f:
        np.save(f, fore_train_ip[2])
    with gzip.GzipFile(f'{output_dir}/train_varis.npy.gz', 'w') as f:
        np.save(f, fore_train_ip[3])
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
    with gzip.GzipFile(f'{output_dir}/val_y.npy.gz', 'w') as f:
        np.save(f, fore_valid_op)
    
    params = {'V': V, 'D': D, 'fore_max_len': fore_max_len}

    with open(f'{output_dir}/extra_params.json', 'w') as f:
        json.dump(params, f)


def load_forecast_dataset(data_dir):
    with gzip.GzipFile(f'{data_dir}/train_demos.npy.gz', 'r') as f:
        train_demos = np.load(f)
    with gzip.GzipFile(f'{data_dir}/train_times.npy.gz', 'r') as f:
        train_times = np.load(f)
    with gzip.GzipFile(f'{data_dir}/train_values.npy.gz', 'r') as f:
        train_values = np.load(f)
    with gzip.GzipFile(f'{data_dir}/train_varis.npy.gz', 'r') as f:
        train_varis = np.load(f)
    with gzip.GzipFile(f'{data_dir}/train_y.npy.gz', 'r') as f:
        train_y = np.load(f)

    with gzip.GzipFile(f'{data_dir}/val_demos.npy.gz', 'r') as f:
        val_demos = np.load(f)
    with gzip.GzipFile(f'{data_dir}/val_times.npy.gz', 'r') as f:
        val_times = np.load(f)
    with gzip.GzipFile(f'{data_dir}/val_values.npy.gz', 'r') as f:
        val_values = np.load(f)
    with gzip.GzipFile(f'{data_dir}/val_varis.npy.gz', 'r') as f:
        val_varis = np.load(f)
    with gzip.GzipFile(f'{data_dir}/val_y.npy.gz', 'r') as f:
        val_y = np.load(f)


    train_dataset = MultipleInputsDataset(
        X_demos=train_demos, 
        X_times=train_times, 
        X_values=train_values, 
        X_varis=train_varis, 
        Y=train_y
    )
    val_dataset = MultipleInputsDataset(
        X_demos=val_demos, 
        X_times=val_times, 
        X_values=val_values, 
        X_varis=val_varis,
        Y=val_y
    )

    with open(f'{data_dir}/extra_params.json') as f:
        params = json.load(f)
    
    V = int(params['V'])
    D = int(params['D'])
    fore_max_len = int(params['fore_max_len'])

    return train_dataset, val_dataset, fore_max_len, V, D


def generate_mortality_dataset(data_path, output_dir):
    # Read data.
    data, oc, train_ind, valid_ind, test_ind = pickle.load(open(data_path, 'rb'))
    # Filter labeled data in first 24h.
    data = data.loc[data.ts_ind.isin(np.concatenate((train_ind, valid_ind, test_ind), axis=-1))]
    data = data.loc[(data.hour>=0)&(data.hour<=24)]
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
    data = data.groupby('ts_ind').head(880)
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
    
    params = {'V': V, 'D': D}

    with open(f'{output_dir}/extra_params.json', 'w') as f:
        json.dump(params, f)


def load_mortality_dataset(data_dir):

    with gzip.GzipFile(f'{data_dir}/train_demos.npy.gz', 'r') as f:
        train_demos = np.load(f)
    with gzip.GzipFile(f'{data_dir}/train_times.npy.gz', 'r') as f:
        train_times = np.load(f)
    with gzip.GzipFile(f'{data_dir}/train_values.npy.gz', 'r') as f:
        train_values = np.load(f)
    with gzip.GzipFile(f'{data_dir}/train_varis.npy.gz', 'r') as f:
        train_varis = np.load(f)
    with gzip.GzipFile(f'{data_dir}/train_y.npy.gz', 'r') as f:
        train_y = np.load(f)

    with gzip.GzipFile(f'{data_dir}/val_demos.npy.gz', 'r') as f:
        val_demos = np.load(f)
    with gzip.GzipFile(f'{data_dir}/val_times.npy.gz', 'r') as f:
        val_times = np.load(f)
    with gzip.GzipFile(f'{data_dir}/val_values.npy.gz', 'r') as f:
        val_values = np.load(f)
    with gzip.GzipFile(f'{data_dir}/val_varis.npy.gz', 'r') as f:
        val_varis = np.load(f)
    with gzip.GzipFile(f'{data_dir}/val_y.npy.gz', 'r') as f:
        val_y = np.load(f)

    with gzip.GzipFile(f'{data_dir}/test_demos.npy.gz', 'r') as f:
        test_demos = np.load(f)
    with gzip.GzipFile(f'{data_dir}/test_times.npy.gz', 'r') as f:
        test_times = np.load(f)
    with gzip.GzipFile(f'{data_dir}/test_values.npy.gz', 'r') as f:
        test_values = np.load(f)
    with gzip.GzipFile(f'{data_dir}/test_varis.npy.gz', 'r') as f:
        test_varis = np.load(f)
    with gzip.GzipFile(f'{data_dir}/test_y.npy.gz', 'r') as f:
        test_y = np.load(f)


    train_dataset = MultipleInputsDataset(
        X_demos=train_demos, 
        X_times=train_times, 
        X_values=train_values, 
        X_varis=train_varis, 
        Y=train_y
    )

    val_dataset = MultipleInputsDataset(
        X_demos=train_demos, 
        X_times=train_times, 
        X_values=train_values, 
        X_varis=train_varis, 
        Y=train_y
    )

    test_dataset = MultipleInputsDataset(
        X_demos=test_demos, 
        X_times=test_times, 
        X_values=test_values, 
        X_varis=test_varis,
        Y=test_y
    )

    with open(f'{data_dir}/extra_params.json') as f:
        params = json.load(f)
    
    V = int(params['V'])
    D = int(params['D'])
    

    return train_dataset, val_dataset, test_dataset, V, D


# def load_forecast_dataset(data_path):
#     pred_window = 2 # hours
#     obs_windows = range(20, 124, 4)

#     # Read data.
#     data, oc, train_ind, valid_ind, test_ind = pickle.load(open(data_path, 'rb'))
#     # Remove test patients.
#     data = data.merge(oc[['ts_ind', 'SUBJECT_ID']], on='ts_ind', how='left')
#     test_sub = oc.loc[oc.ts_ind.isin(test_ind)].SUBJECT_ID.unique()
#     data = data.loc[~data.SUBJECT_ID.isin(test_sub)]
#     oc = oc.loc[~oc.SUBJECT_ID.isin(test_sub)]
#     data.drop(columns=['SUBJECT_ID', 'TABLE'], inplace=True)
#     # Fix age.
#     data.loc[(data.variable=='Age')&(data.value>200), 'value'] = 91.4
#     # Get static data with mean fill and missingness indicator.
#     static_varis = ['Age', 'Gender']
#     ii = data.variable.isin(static_varis)
#     static_data = data.loc[ii]
#     data = data.loc[~ii]
#     def inv_list(l, start=0):
#         d = {}
#         for i in range(len(l)):
#             d[l[i]] = i+start
#         return d
#     static_var_to_ind = inv_list(static_varis)
#     D = len(static_varis)
#     N = data.ts_ind.max()+1
#     demo = np.zeros((N, D))
#     for row in tqdm(static_data.itertuples()):
#         demo[row.ts_ind, static_var_to_ind[row.variable]] = row.value
#     # Normalize static data.
#     means = demo.mean(axis=0, keepdims=True)
#     stds = demo.std(axis=0, keepdims=True)
#     stds = (stds==0)*1 + (stds!=0)*stds
#     demo = (demo-means)/stds
#     # Get variable indices.
#     varis = sorted(list(set(data.variable)))
#     V = len(varis)
#     var_to_ind = inv_list(varis, start=1)
#     data['vind'] = data.variable.map(var_to_ind)
#     data = data[['ts_ind', 'vind', 'hour', 'value']].sort_values(by=['ts_ind', 'vind', 'hour'])
#     # Find max_len.
#     fore_max_len = 880
#     # Get forecast inputs and outputs.
#     fore_times_ip = []
#     fore_values_ip = []
#     fore_varis_ip = []
#     fore_op = []
#     fore_inds = []
#     def f(x):
#         mask = [0 for i in range(V)]
#         values = [0 for i in range(V)]
#         for vv in x:
#             v = int(vv[0])-1
#             mask[v] = 1
#             values[v] = vv[1]
#         return values+mask
#     def pad(x):
#         return x+[0]*(fore_max_len-len(x))
#     for w in tqdm(obs_windows):
#         pred_data = data.loc[(data.hour>=w)&(data.hour<=w+pred_window)]
#         pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value':'first'}).reset_index()
#         pred_data['vind_value'] = pred_data[['vind', 'value']].values.tolist()
#         pred_data = pred_data.groupby('ts_ind').agg({'vind_value':list}).reset_index()
#         pred_data['vind_value'] = pred_data['vind_value'].apply(f)    
#         obs_data = data.loc[(data.hour<w)&(data.hour>=w-24)]
#         obs_data = obs_data.loc[obs_data.ts_ind.isin(pred_data.ts_ind)]
#         obs_data = obs_data.groupby('ts_ind').head(fore_max_len)
#         obs_data = obs_data.groupby('ts_ind').agg({'vind':list, 'hour':list, 'value':list}).reset_index()
#         obs_data = obs_data.merge(pred_data, on='ts_ind')
#         for col in ['vind', 'hour', 'value']:
#             obs_data[col] = obs_data[col].apply(pad)
#         fore_op.append(np.array(list(obs_data.vind_value)))
#         fore_inds.append(np.array(list(obs_data.ts_ind)))
#         fore_times_ip.append(np.array(list(obs_data.hour)))
#         fore_values_ip.append(np.array(list(obs_data.value)))
#         fore_varis_ip.append(np.array(list(obs_data.vind)))
#     del data
#     fore_times_ip = np.concatenate(fore_times_ip, axis=0)
#     fore_values_ip = np.concatenate(fore_values_ip, axis=0)
#     fore_varis_ip = np.concatenate(fore_varis_ip, axis=0)
#     fore_op = np.concatenate(fore_op, axis=0)
#     fore_inds = np.concatenate(fore_inds, axis=0)
#     fore_demo = demo[fore_inds]
#     # Get train and valid ts_ind for forecast task.
#     train_sub = oc.loc[oc.ts_ind.isin(train_ind)].SUBJECT_ID.unique()
#     valid_sub = oc.loc[oc.ts_ind.isin(valid_ind)].SUBJECT_ID.unique()
#     rem_sub = oc.loc[~oc.SUBJECT_ID.isin(np.concatenate((train_ind, valid_ind)))].SUBJECT_ID.unique()
#     bp = int(0.8*len(rem_sub))
#     train_sub = np.concatenate((train_sub, rem_sub[:bp]))
#     valid_sub = np.concatenate((valid_sub, rem_sub[bp:]))
#     train_ind = oc.loc[oc.SUBJECT_ID.isin(train_sub)].ts_ind.unique() # Add remaining ts_ind s of train subjects.
#     valid_ind = oc.loc[oc.SUBJECT_ID.isin(valid_sub)].ts_ind.unique() # Add remaining ts_ind s of train subjects.
#     # Generate 3 sets of inputs and outputs.
#     train_ind = np.argwhere(np.in1d(fore_inds, train_ind)).flatten()
#     valid_ind = np.argwhere(np.in1d(fore_inds, valid_ind)).flatten()
#     fore_train_ip = [ip[train_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]]
#     fore_valid_ip = [ip[valid_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]]
#     del fore_times_ip, fore_values_ip, fore_varis_ip, demo, fore_demo
#     fore_train_op = fore_op[train_ind]
#     fore_valid_op = fore_op[valid_ind]
#     del fore_op

#     train_dataset = MultipleInputsDataset(
#         X_demos=fore_train_ip[0], 
#         X_times=fore_train_ip[1], 
#         X_values=fore_train_ip[2], 
#         X_varis=fore_train_ip[3], 
#         Y=fore_train_op
#     )
#     val_dataset = MultipleInputsDataset(
#         X_demos=fore_valid_ip[0], 
#         X_times=fore_valid_ip[1], 
#         X_values=fore_valid_ip[2], 
#         X_varis=fore_valid_ip[3],
#         Y=fore_valid_op
#     )

#     return train_dataset, val_dataset, fore_max_len, V, D


# def load_mortality_dataset(data_path):
    # Read data.
    data, oc, train_ind, valid_ind, test_ind = pickle.load(open(data_path, 'rb'))
    # Filter labeled data in first 24h.
    data = data.loc[data.ts_ind.isin(np.concatenate((train_ind, valid_ind, test_ind), axis=-1))]
    data = data.loc[(data.hour>=0)&(data.hour<=24)]
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
    data = data.groupby('ts_ind').head(880)
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

    train_dataset = MultipleInputsDataset(
        X_demos=train_ip[0], 
        X_times=train_ip[1], 
        X_values=train_ip[2], 
        X_varis=train_ip[3], 
        Y=train_op
    )

    val_dataset = MultipleInputsDataset(
        X_demos=valid_ip[0], 
        X_times=valid_ip[1], 
        X_values=valid_ip[2], 
        X_varis=valid_ip[3], 
        Y=valid_op
    )

    test_dataset = MultipleInputsDataset(
        X_demos=test_ip[0], 
        X_times=test_ip[1], 
        X_values=test_ip[2], 
        X_varis=test_ip[3],
        Y=test_iop
    )

    return train_dataset, val_dataset, test_dataset, V, D