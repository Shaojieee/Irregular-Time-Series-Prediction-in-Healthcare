import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from datetime import datetime
import time
import json
import argparse
from tqdm import tqdm
import pandas as pd

def parse_args():

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument(
        "--train_job", type=str, help="forecasting_model or mortality_model"
    )
    parser.add_argument(
        "--data_dir", type=str, help="A path to dataset folder"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="forecasting_model or mortality_model"
    )
    
    parser.add_argument(
        "--model_weights", type=str, default=None, help="forecasting_model or mortality_model"
    )
    parser.add_argument('--with_text', action='store_true')
    parser.add_argument(
        "--text_padding",
        action='store_true'
    )
    parser.add_argument(
        "--text_max_length",
        type=int,
        default=1024,
        help="mMximum total input sequence length after tokenization. Sequences longer than this will be truncated,sequences shorter will be padded if `--text_padding` is passed.",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=50,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=2,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument(
        "--he",
        type=int,
        default=4,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.02,
        help="Learning Rate",
    )
    parser.add_argument(
        "--text_num_notes",
        type=int,
        default=5,
        help="Maximum no. of notes to use",
    )
    parser.add_argument(
        "--period_length",
        type=int,
        default=48,
        help="Max hours of data to use",
    )
    parser.add_argument(
        "--text_encoder_model",
        type=str,
        default='bioLongformer',
        help="Text Encoder Model to use",
    )
    parser.add_argument(
        "--ts_learning_rate",
        type=float,
        default=0.0005,
        help="Time Series Learning Rate",
    )
    parser.add_argument(
        "--text_learning_rate",
        type=float,
        default=0.00002,
        help="Text Model Learning Rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument(
        "--early_stopper_min_delta",
        type=float,
        default=0.0005,
        help="Learning Rate",
    )
    parser.add_argument(
        "--early_stopper_mode", type=str, default='min', help="forecasting_model or mortality_model"
    )
    parser.add_argument('--early_stopper_restore_best_weights', action='store_true')
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size  for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=None,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument(
        "--lds",
        type=list_of_ints,
        default=None,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="Total number of training epochs to perform.")

    args = parser.parse_args()

    return args


def forecast_loss(y_true, y_pred, V):
    return torch.sum(y_true[:,V:]*(y_true[:,:V]- y_pred)**2, dim=-1).mean()

def forecast_results(y_true, y_pred, **kwargs):
    V = kwargs['V']
    return {'LOSS': torch.sum(y_true[:,V:]*(y_true[:,:V]- y_pred)**2, dim=-1).mean().item()}


def mortality_loss(y_true, y_pred):
    return torch.nn.functional.binary_cross_entropy(y_pred, y_true)

def mortality_results(y_true, y_pred, **kwargs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    min_rp = np.minimum(precision, recall).max()
    roc_auc = roc_auc_score(y_true, y_pred)
    bce_loss = torch.nn.functional.binary_cross_entropy(y_pred, y_true).item()

    f1 = f1_score(y_true, (y_pred>0.5))

    return {'PR_AUC': pr_auc, 'ROC_AUC': roc_auc, 'MIN_RP': min_rp, 'LOSS': bce_loss, 'F1': f1}


class EvaluationCallback():
    def __init__(self, val_dataloader, evaluation_fn):
        self.val_dataloader = val_dataloader
        self.evaluation_fn = evaluation_fn
        self.logs = []

    def on_epoch_end(self, model, epoch, with_text, **kwargs):
        model.eval()
        Y_ = []
        Y_pred_ = []
        with torch.no_grad():
            for step, batch in tqdm(enumerate(self.val_dataloader)):
                if with_text:
                    X_demos, X_times, X_values, X_varis, Y, X_text_tokens, X_text_attention_mask, X_text_times, X_text_time_mask, X_text_feature_varis = batch
                    Y_pred = model(X_demos, X_times, X_values, X_varis, X_text_tokens, X_text_attention_mask, X_text_times, X_text_feature_varis)
                else:
                    X_demos, X_times, X_values, X_varis, Y = batch
                    Y_pred = model(X_demos, X_times, X_values, X_varis)

                Y_.append(Y); Y_pred_.append(Y_pred)
        Y_ = torch.cat(Y_)
        Y_pred_ = torch.cat(Y_pred_)
        Y_ = Y_.cpu()
        Y_pred_ = Y_pred_.cpu()
        results = self.evaluation_fn(Y_, Y_pred_, **kwargs)

        output_string = f'Val Metrics: Epoch: {epoch}'
        for k, v in results.items():
            output_string += f' {k}: {v:.6f}'
        print(output_string)
        results['epoch'] = epoch
        self.logs.append(results)
    
        return results
                   
    def get_logs(self):
        return pd.DataFrame(self.logs)
    
    def save_results(self, output_dir, file_name):
        results = pd.DataFrame(self.logs)
        results.to_csv(f'{output_dir}/{file_name}')
            
              
class EarlyStopper():
    def __init__(self, patience=5, min_delta=0, mode='max', restore_best_weights=False):
        self.patience = patience
        self.wait = 0
        self.min_delta = min_delta
        self.mode = mode
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best = np.inf if mode=='min' else -np.inf
        
        if mode=='min':
            self.monitor_op = np.less
        elif mode=='max':
            self.monitor_op = np.greater
              
    def on_epoch_end(self, model, loss, epoch):
        
        if self.monitor_op(loss - self.min_delta, self.best):
            self.best = loss
            self.wait = 0
            if self.restore_best_weights:
                model.eval()
                self.best_weights = model.state_dict()
        else:
            self.wait += 1
            
            if self.wait>=self.patience:
                self.stopped_epoch = epoch
                print(f'Early Stopping at Epoch {epoch} with best loss of {self.best:.6f}')
                if self.restore_best_weights:
                      print(f'Restoring best weights at Epoch {self.stopped_epoch-self.wait}')
                      model.eval()
                      model.load_state_dict(self.best_weights)
                return True
        return False
    
    def save_best_weights(self, accelerator, output_dir, file_name=None):
        if file_name==None:
            file_name = 'best_weights.pth'
        accelerator.save(self.best_weights, f'{output_dir}/{file_name}')
    
    def load_best_weights(self, model):
        if self.restore_best_weights:
            print(f'Restoring best weights at Epoch {self.stopped_epoch-self.wait}')
            model.eval()
            model.load_state_dict(self.best_weights)


def time_check(start_time, mins_left=30):
    cur_time = datetime.now()

    if (cur_time - start_time).total_seconds() > (6*60*60 - mins_left*60):
        return True
    return False


def save_experiment_config(args, output_dir=None, file_name=None):
    if output_dir==None:
        output_dir = args.output_dir
    if file_name==None:
        file_name = f'{args.train_job}_details.json'
    
    config_dict = vars(args)
    config_dict['start_time'] = config_dict['start_time'].strftime('%H:%M %d_%m_%Y')

    with open(f'{output_dir}/{file_name}', 'w') as f:

        json.dump(config_dict, f)