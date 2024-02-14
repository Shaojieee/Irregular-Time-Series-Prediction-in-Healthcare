import gzip
import json
import numpy as np
from tqdm import tqdm
import os

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def generate_imputed_matrix(demo, times, values, varis, mice=None):
    
    num_features = (np.max(varis)) + 1 + demo.shape[-1] + 1 # num_varis(without masking 0) + time_col + num_demo + patient_id
    num_patients = times.shape[0]

    flatten_matrix = []
    for i in tqdm(range(num_patients)):
        non_mask_index = varis[i,:]!=0
        cur_times = times[i,:][non_mask_index]
        cur_values = values[i,:][non_mask_index]
        cur_varis = varis[i,:][non_mask_index]
        
        index = np.argsort(cur_times, axis=-1)
        cur_times = cur_times[index]
        cur_values = cur_values[index]
        cur_varis = cur_varis[index]

        timesteps = np.unique(cur_times)
        matrix = np.empty(shape=(len(timesteps), num_features))
        matrix[:] = np.nan
        matrix[:,0] = timesteps 
        time_index = np.array([np.where(timesteps == x)[0][0] for x in cur_times])

        matrix[time_index, cur_varis] = cur_values
        matrix[:, (np.max(varis)) + 1:-1] = demo[i,:]
        matrix[:, -1] = i

        flatten_matrix.append(matrix)
    
    flatten_matrix = np.vstack(flatten_matrix)
    print(f'New Shape: {flatten_matrix.shape}')

    if mice is None:
        mice = IterativeImputer(random_state=42, min_value=-3, max_value=3, verbose=0, max_iter=50)
        mice.fit(flatten_matrix)
        
    
    imputed_times, imputed_values, imputed_varis, imputed_mask = [],[],[],[]
    for i in tqdm(range(num_patients)):

        non_mask_index = varis[i,:]!=0
        cur_times = times[i,:][non_mask_index]
        cur_values = values[i,:][non_mask_index]
        cur_varis = varis[i,:][non_mask_index]
        
        index = np.argsort(cur_times, axis=-1)
        cur_times = cur_times[index]
        cur_values = cur_values[index]
        cur_varis = cur_varis[index]
        
        matrix = flatten_matrix[flatten_matrix[:,-1]==i]

        imputation_mask = []
        to_impute = []
        for j in range(1,np.max(varis)+1):
            temp = matrix[:,[0,j]]
            temp = temp[~np.isnan(temp[:,1])]
            time_diff = np.diff(temp[:,0])
            for k in range(len(time_diff)):
                if time_diff[k]>=1:
                    impute = np.empty(shape=(num_features))
                    impute[:] = np.nan
                    # Impute the midpoint of 2 records
                    impute[0] = temp[k,0]+ (time_diff[k]/2)
                    # Stating the patient id
                    impute[-1] = i
                    # Stating the demographics
                    impute[(np.max(varis)) + 1:-1] = demo[i,:]
                    to_impute.append(impute)
                    mask = np.zeros(shape=(num_features))
                    # Feature to impute
                    mask[j] = 1
                    # Timing mask
                    mask[0] = 1
                    imputation_mask.append(mask)
                    
        if len(to_impute)>0:
            to_impute = np.vstack(to_impute)
            imputation_mask = np.vstack(imputation_mask).astype('bool')

            imputed = mice.transform(to_impute)

            # Finding rows that were not imputed
            nan_mask = np.isnan(imputed[:,1:-1][imputation_mask[:,1:-1]])
            # Timing for imputed rows
            cur_imputed_time = imputed[~nan_mask,0]

            # Set values for imputations that we are not interested in
            imputed[~imputation_mask] = np.nan
            # Get the varis for each row
            cur_imputed_varis = np.where(~np.isnan(imputed[~nan_mask,1:-1]))[1]

            # Set the values for the other imputations as 0 so that we can sum and get the imputation that we are interested in
            imputed[~imputation_mask] = 0
            cur_imputed_values = np.sum(imputed[~nan_mask,1:-1], axis=-1)

            imputed_times.append(np.concatenate((cur_times, cur_imputed_time)))
            imputed_varis.append(np.concatenate((cur_varis, cur_imputed_varis)))
            imputed_values.append(np.concatenate((cur_values, cur_imputed_values)))
            imputed_mask.append(np.concatenate((np.zeros_like(cur_times), np.ones_like(cur_imputed_time))))
        else:
            imputed_times.append(cur_times)
            imputed_varis.append(cur_varis)
            imputed_values.append(cur_values)
            imputed_mask.append(np.zeros_like(cur_times))

        assert imputed_times[i].shape==imputed_values[i].shape
        assert imputed_times[i].shape==imputed_varis[i].shape
        assert imputed_times[i].shape==imputed_mask[i].shape


    return imputed_times, imputed_values, imputed_varis, imputed_mask, mice


debug = False
input_path = './data_physionet_mortality_0_48_1498'
output_path = './data_all_imputed_physionet_mortality'
mice = None
for mode in ['train', 'val', 'test']:
    with gzip.GzipFile(f'{input_path}/{mode}_demos.npy.gz', 'r') as f:
        demo = np.load(f)
    with gzip.GzipFile(f'{input_path}/{mode}_times.npy.gz', 'r') as f:
        times = np.load(f)
    with gzip.GzipFile(f'{input_path}/{mode}_values.npy.gz', 'r') as f:
        values = np.load(f)
    with gzip.GzipFile(f'{input_path}/{mode}_varis.npy.gz', 'r') as f:
        varis = np.load(f)
    with gzip.GzipFile(f'{input_path}/{mode}_y.npy.gz', 'r') as f:
        y = np.load(f)

    with open(f'{input_path}/extra_params.json') as f:
        params = json.load(f)
        
        V = int(params['V'])
        D = int(params['D'])
    
    if debug:
        demo = demo[:10]
        times = times[:10]
        values = values[:10]
        varis = varis[:10]
        y = y[:10]

    imputed_times, imputed_values, imputed_varis, imputed_mask, mice = generate_imputed_matrix(demo, times, values, varis, mice=mice)

    longest = max(imputed_times, key=lambda x: x.shape[0])
    max_len = longest.shape[0]

    padded_times, padded_values, padded_varis, padded_mask = [],[],[],[]
    for i in range(len(imputed_times)):
        assert imputed_times[i].shape==imputed_values[i].shape
        assert imputed_times[i].shape==imputed_varis[i].shape
        assert imputed_times[i].shape==imputed_mask[i].shape
        time = np.pad(imputed_times[i], (0,max_len-imputed_times[i].shape[0]))
        values = np.pad(imputed_values[i], (0,max_len-imputed_values[i].shape[0]))
        varis = np.pad(imputed_varis[i], (0,max_len-imputed_varis[i].shape[0]))
        mask = np.pad(imputed_mask[i], (0,max_len-imputed_mask[i].shape[0]))

        padded_times.append(time)
        padded_values.append(values)
        padded_varis.append(varis)
        padded_mask.append(mask)


    padded_times = np.vstack(padded_times)
    padded_values = np.vstack(padded_values)
    padded_varis = np.vstack(padded_varis)
    padded_mask = np.vstack(padded_mask)

    
    os.makedirs(output_path, exist_ok=True)
    with gzip.GzipFile(f'{output_path}/{mode}_demos.npy.gz', 'w') as f:
        np.save(f, demo)
    with gzip.GzipFile(f'{output_path}/{mode}_times.npy.gz', 'w') as f:
        np.save(f, padded_times)
    with gzip.GzipFile(f'{output_path}/{mode}_values.npy.gz', 'w') as f:
        np.save(f, padded_values)
    with gzip.GzipFile(f'{output_path}/{mode}_varis.npy.gz', 'w') as f:
        np.save(f, padded_varis)
    with gzip.GzipFile(f'{output_path}/{mode}_imputed_mask.npy.gz', 'w') as f:
        np.save(f, padded_mask)
    with gzip.GzipFile(f'{output_path}/{mode}_y.npy.gz', 'w') as f:
        np.save(f, y)

    with open(f'{output_path}/extra_params.json', 'w') as f:
        json.dump(params, f)
        