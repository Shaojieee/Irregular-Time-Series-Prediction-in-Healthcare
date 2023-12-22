from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import pickle
import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

class MultipleInputsDataset(Dataset):
    def __init__(self, X_demos, X_times, X_values, X_varis, Y, X_text_tokens=None, X_text_attention_mask=None, X_text_times=None, X_text_time_mask=None, X_text_feature_varis=None):
        super(MultipleInputsDataset, self).__init__()
        
        self.X_demos = torch.tensor(X_demos, dtype=torch.float32)
        self.X_times = torch.tensor(X_times, dtype=torch.float32)
        self.X_values = torch.tensor(X_values, dtype=torch.float32)
        self.X_varis = torch.tensor(X_varis, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        
        self.has_text = X_text_tokens!=None

        if self.has_text:
            # X_text_tokens is a list of list of tensor. Cannot convert to tensor as text tokens are not yet padded. Will be padded in DataLoader
            self.X_text_tokens = X_text_tokens
            self.X_text_attention_mask = X_text_attention_mask
            self.X_text_times = torch.tensor(X_text_times, dtype=torch.float)
            self.X_text_time_mask = torch.tensor(X_text_time_mask, dtype=torch.long)
            self.X_text_feature_varis = torch.tensor(X_text_feature_varis, dtype=torch.long)
    
    def __getitem__(self, idx):
        if self.has_text:
            return self.X_demos[idx], self.X_times[idx], self.X_values[idx], self.X_varis[idx], self.Y[idx], self.X_text_tokens[idx], self.X_text_attention_mask[idx], self.X_text_times[idx], self.X_text_time_mask[idx], self.X_text_feature_varis[idx]
        else:
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


def load_mortality_dataset(data_dir, with_text=False, tokenizer=None, text_padding=None, text_max_len=None, text_model=None, period_length=48, num_notes=5, debug=False):
    has_demos = True

    if 'train_demos.npy.gz' not in os.listdir(data_dir):
        has_demos = False

    datasets_ = {}
    for mode in ['train', 'val', 'test']:
        
        with gzip.GzipFile(f'{data_dir}/{mode}_times.npy.gz', 'r') as f:
            ts_times = np.load(f)
        with gzip.GzipFile(f'{data_dir}/{mode}_values.npy.gz', 'r') as f:
            ts_values = np.load(f)
        with gzip.GzipFile(f'{data_dir}/{mode}_varis.npy.gz', 'r') as f:
            ts_varis = np.load(f)
        with gzip.GzipFile(f'{data_dir}/{mode}_y.npy.gz', 'r') as f:
            y = np.load(f)

        if has_demos:
            with gzip.GzipFile(f'{data_dir}/{mode}_demos.npy.gz', 'r') as f:
                demos = np.load(f)
        else: 
            demos = np.zeros((len(y), 1))

        if with_text:
            raw_texts = pickle.load(open(f'{data_dir}/{mode}_texts.pkl', 'rb'))
            raw_text_time_from_start = pickle.load(open(f'{data_dir}/{mode}_text_times.pkl', 'rb'))
        else:
            raw_texts, raw_text_time_from_start = None, None

        if debug:
            ts_times = ts_times[:100]
            ts_values = ts_times[:100]
            ts_varis = ts_varis[:100]
            y = y[:100]
            raw_texts = raw_texts[:100] if with_text else None
            raw_text_time_from_start = raw_text_time_from_start[:100] if with_text else None

        if with_text:
            all_text_time_from_start = []
            all_text_time_mask = []
            all_text_token = []
            all_text_atten_mask = []
            all_text_feature_varis = []
            text_feature_varis = np.max(ts_varis) + 1
            for texts, times in tqdm(zip(raw_texts, raw_text_time_from_start)):
                text_time_from_start = []
                text_time_mask = []
                text_token=[]
                text_atten_mask=[]
                for text, time in zip(texts, times):

                    text_time_from_start.append(time)
                    text_time_mask.append(1)

                    inputs = tokenizer.encode_plus(
                        text, 
                        padding='max_length' if text_padding else False,
                        max_length=text_max_len,
                        add_special_tokens=True,
                        return_attention_mask = True,
                        truncation=True
                    )

                    text_token.append(torch.tensor(inputs['input_ids'],dtype=torch.long))
                    attention_mask=inputs['attention_mask']
                    if "Longformer" in text_model :

                        attention_mask[0]+=1
                        text_atten_mask.append(torch.tensor(attention_mask,dtype=torch.long))
                    else:
                        text_atten_mask.append(torch.tensor(attention_mask,dtype=torch.long))
                
                while len(text_token)<num_notes:
                    text_token.append(torch.tensor([0],dtype=torch.long))
                    text_atten_mask.append(torch.tensor([0],dtype=torch.long))
                    text_time_from_start.append(0)
                    text_time_mask.append(0)
                
                text_token = text_token[-num_notes:]
                text_atten_mask = text_atten_mask[-num_notes:]
                text_time_from_start = text_time_from_start[-num_notes:]
                text_time_mask = text_time_mask[-num_notes:]

                all_text_token.append(text_token)
                all_text_atten_mask.append(text_atten_mask)
                all_text_time_from_start.append(text_time_from_start)
                all_text_time_mask.append(text_time_mask)
                all_text_feature_varis.append([text_feature_varis]*num_notes)
        else:
            all_text_token = None
            all_text_atten_mask = None
            all_text_time_from_start = None
            all_text_time_mask = None
            all_text_feature_varis = None


        dataset = MultipleInputsDataset(
            X_demos=demos, 
            X_times=ts_times, 
            X_values=ts_values, 
            X_varis=ts_varis, 
            Y=y,
            X_text_tokens=all_text_token,
            X_text_attention_mask=all_text_atten_mask,
            X_text_times=all_text_time_from_start,
            X_text_time_mask=all_text_time_mask,
            X_text_feature_varis=all_text_feature_varis
        )

        datasets_[mode] = dataset


    with open(f'{data_dir}/extra_params.json') as f:
        params = json.load(f)
    
    V = int(params['V'])
    D = int(params['D'])
    
    return datasets_['train'], datasets_['val'], datasets_['test'], V, D


# max_len for max number of measurements
# Using 75% percentile of number of measurements in trainig set
def generate_mortality_dataset_mimic_iii_benchmark(
    data_path,
    text_data_path,
    text_start_time_path,
    output_dir, 
    train_listfile, 
    val_listfile, 
    test_listfile, 
    channel_file,
    dis_config_file,
    max_len=500,
    period_length=48
):

    variables = {}
    demo_variables = {}
    mean__ = {}
    std__ = {}

    # For time series data
    channel_info_file = open(channel_file)
    dis_config_file=open(dis_config_file)
    channel_info = json.load(channel_info_file)
    dis_config=json.load(dis_config_file)
    is_catg=dis_config['is_categorical_channel']
    # List of categorical Variables
    is_catg = [k for k,v in is_catg.items() if v==True]
    channel_info_file.close()
    dis_config_file.close()


    os.makedirs(output_dir, exist_ok=True)

    for mode in ['train', 'val', 'test']:
        if mode=='train':
            file_df = pd.read_csv(train_listfile)
            with open(text_start_time_path + '/starttime.pkl', 'rb') as f:
                text_episode_to_start_time = pickle.load(f)
        elif mode=='val':
            file_df = pd.read_csv(val_listfile)
            with open(text_start_time_path + '/starttime.pkl', 'rb') as f:
                text_episode_to_start_time = pickle.load(f)
        elif mode=='test':
            file_df = pd.read_csv(test_listfile)
            with open(text_start_time_path + '/test_starttime.pkl', 'rb') as f:
                text_episode_to_start_time = pickle.load(f)

        N = len(file_df)

        times_inp = np.zeros((N, max_len), dtype='float32')
        values_inp = np.zeros((N, max_len), dtype='float32')
        varis_inp = np.zeros((N, max_len), dtype='int32')

        text_inp = [[]] * N
        text_time_from_start_inp = [[]] * N


        y = np.zeros(N, dtype='int32')

        
        for i, file in tqdm(file_df.iterrows()):
            # Extracting time series data
            data = pd.read_csv(os.path.join(data_path, 'test' if mode=='test' else 'train', file['stay']))
            data = data[data['Hours']<=period_length]

            if len(variables)==0:
                temp_variables = data.columns.tolist()
                if 'Hours' in temp_variables:
                    temp_variables.remove('Hours')
                
                variables = {temp_variables[i]:i+1 for i in range(len(temp_variables))}

            melted_data = data.melt(id_vars=['Hours'], var_name='feature', value_name='value')
            melted_data = melted_data.dropna()

            melted_data['feature_encoding'] = melted_data['feature'].map(variables)
            melted_data = melted_data.sort_values(by=['Hours'], ascending=True)
            melted_data = melted_data.head(n=max_len)

            melted_data['value_encoding'] = melted_data['value']
            for j, row in melted_data.iterrows():
                if row['feature'] in is_catg:
                    feature = row['feature']
                    if feature == 'Glascow coma scale total':
                        value = int(row['value'])
                    else:
                        value = row['value']
                    melted_data.at[j,'value_encoding'] = channel_info[row['feature']]['values'][str(value)]

            # Populating time series data in numpy array
            times_inp[i, :len(melted_data)] = melted_data['Hours'] 
            values_inp[i, :len(melted_data)] = melted_data['value_encoding']
            varis_inp[i, :len(melted_data)] = melted_data['feature_encoding']
            y[i] = file['y_true']

            # Extracting text data
            # text_filename = patient_id + '_' + episode_id
            text_file_name = file['stay'].split('_')[0] + '_' + file['stay'].split('_')[1].strip('episode')
            text_file_path = os.path.join(text_data_path, 'test_text_fixed' if mode=='test' else 'text_fixed', text_file_name)
            if os.path.exists(text_file_path):
                with open(text_file_path, 'r') as f:
                    text_dict = json.load(f)
                time_list = sorted(text_dict.keys())
                text_list = [" ".join(text_dict[time]) for time in time_list]
                assert len(time_list) == len(text_list)

                text_start_time = text_episode_to_start_time[text_file_name]
                if not (len(text_list) == 0 or text_start_time == -1):
                    
                    final_text_list = []
                    final_time_list = []
                    for (time, text) in zip(time_list, text_list):
                        time_diff = (np.datetime64(time) - np.datetime64(text_start_time)).astype('timedelta64[h]').astype(int)
                        # Checking if within period_length
                        if time_diff <= period_length + 1e-6: #and  diff(start_time, t)>=(-24-1e-6)
                            final_text_list.append(text)
                            time_from_start = (np.datetime64(time) - np.datetime64(text_start_time)).astype('timedelta64[m]').astype(int)/60.0
                            final_time_list.append(time_from_start)
                        else:
                            break

                    if len(final_text_list)>0:
                        text_inp[i] = final_text_list
                        text_time_from_start_inp[i] = final_time_list


        # Normalising values of features
        for i in variables.values():

            indexes = varis_inp==i
            if mode=='train':
                mean__[i] = values_inp[indexes].mean()
                std__[i] = values_inp[indexes].std()
            
            values_inp[indexes] = (values_inp[indexes] - mean__[i]) / std__[i]

            
        # TODO: Currently no demographics
        # with gzip.GzipFile(f'{output_dir}/{mode}_demos.npy.gz', 'w') as f:
        #     np.save(f, train_ip[0])

        with gzip.GzipFile(f'{output_dir}/{mode}_times.npy.gz', 'w') as f:
            np.save(f, times_inp)
        with gzip.GzipFile(f'{output_dir}/{mode}_values.npy.gz', 'w') as f:
            np.save(f, values_inp)
        with gzip.GzipFile(f'{output_dir}/{mode}_varis.npy.gz', 'w') as f:
            np.save(f, varis_inp)
        with gzip.GzipFile(f'{output_dir}/{mode}_y.npy.gz', 'w') as f:
            np.save(f, y)

        with open(f'{output_dir}/{mode}_texts.pkl', 'wb') as f:
            pickle.dump(text_inp, f)

        with open(f'{output_dir}/{mode}_text_times.pkl', 'wb') as f:
            pickle.dump(text_time_from_start_inp, f)


    params = {'V': len(variables), 'D': len(demo_variables)}

    with open(f'{output_dir}/extra_params.json', 'w') as f:
        json.dump(params, f)


def pad_text_data(batch):
    X_demos, X_times, X_values, X_varis, Y, X_text_tokens, X_text_attention_mask, X_text_times, X_text_time_mask, X_text_feature_varis = zip(*batch)

    X_text_tokens = [pad_sequence(tokens, batch_first=True, padding_value=0).transpose(0,1) for tokens in X_text_tokens]
    X_text_attention_mask = [pad_sequence(atten, batch_first=True, padding_value=0).transpose(0,1) for atten in X_text_attention_mask]

    X_text_tokens = pad_sequence(X_text_tokens, batch_first=True, padding_value=0).transpose(1,2)
    X_text_attention_mask = pad_sequence(X_text_attention_mask, batch_first=True, padding_value=0).transpose(1,2)

    X_text_times = pad_sequence([torch.tensor(time, dtype=torch.float) for time in X_text_times],batch_first=True,padding_value=0)
    X_text_time_mask = pad_sequence([torch.tensor(time_mask, dtype=torch.long) for time_mask in X_text_time_mask],batch_first=True,padding_value=0)
    
    X_demos = torch.stack(X_demos)
    X_times = torch.stack(X_times)
    X_values = torch.stack(X_values)
    X_varis = torch.stack(X_varis)
    X_text_feature_varis = torch.stack(X_text_feature_varis)
    Y = torch.stack(Y)


    return X_demos, X_times, X_values, X_varis, Y, X_text_tokens, X_text_attention_mask, X_text_times, X_text_time_mask, X_text_feature_varis


def combine_values_varis_with_text(batch, normalise_varis=False):

    X_demos, X_times, X_values, X_varis, Y, X_text_tokens, X_text_attention_mask, X_text_times, X_text_time_mask, X_text_feature_varis = batch

    if normalise_varis:
        varis_min, varis_max = X_varis.min(), X_varis.max()
        temp_varis = ((X_varis - varis_min) / (varis_max - varis_min) * (1 - 0)) + 0
    else:
        temp_varis = X_varis

    X_values = torch.cat([torch.unsqueeze(X_values,dim=-1), torch.unsqueeze(temp_varis, dim=-1)], dim=-1)
    
    return X_demos, X_times, X_values, X_varis, Y, X_text_tokens, X_text_attention_mask, X_text_times, X_text_time_mask, X_text_feature_varis


def combine_values_varis(batch, normalise_varis=False):
    
    X_demos, X_times, X_values, X_varis, Y = zip(*batch)

    X_demos = torch.stack(X_demos)
    X_times = torch.stack(X_times)
    X_values = torch.stack(X_values)
    X_varis = torch.stack(X_varis)
    Y = torch.stack(Y)

    if normalise_varis:
        varis_min, varis_max = X_varis.min(), X_varis.max()
        temp_varis = ((X_varis - varis_min) / (varis_max - varis_min) * (1 - 0)) + 0
    else:
        temp_varis = X_varis

    X_values = torch.cat([torch.unsqueeze(X_values,dim=-1), torch.unsqueeze(temp_varis, dim=-1)], dim=-1)
    
    return X_demos, X_times, X_values, X_varis, Y

