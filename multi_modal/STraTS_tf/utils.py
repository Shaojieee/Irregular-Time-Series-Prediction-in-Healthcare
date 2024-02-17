from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
import argparse
import numpy as np


def parse_args():

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data_dir", type=str, help="A path to dataset folder"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=880,
        help="No. of obs in the dataset for each sequence",
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=48,
        help="Max Timing for time query",
    )
    parser.add_argument('--with_demo', action='store_true')
    parser.add_argument(
        "--output_dir", type=str, default=None, help="forecasting_model or mortality_model"
    )
    parser.add_argument(
        "--model_weights", type=str, default=None, help="file path of model weights"
    )
    parser.add_argument(
        "--model_type", type=str, default='default', help="custom or special or imputed or mtand_strats or mtand"
    )
    parser.add_argument(
        "--len_time_query", type=int, default=48, help="len of time query vector in mtand"
    )
    parser.add_argument(
        "--d_strats",
        type=int,
        default=32,
        help="",
    )
    parser.add_argument(
        "--N_strats",
        type=int,
        default=2,
        help="",
    )
    parser.add_argument(
        "--he_strats",
        type=int,
        default=4,
        help="",
    )
    parser.add_argument(
        "--dropout_strats",
        type=float,
        default=0.2,
        help="",
    )
    parser.add_argument(
        "--d_mtand",
        type=int,
        default=32,
        help="",
    )
    parser.add_argument(
        "--N_mtand",
        type=int,
        default=0,
        help="",
    )
    parser.add_argument(
        "--he_mtand",
        type=int,
        default=8,
        help="",
    )
    parser.add_argument(
        "--dropout_mtand",
        type=float,
        default=0.2,
        help="",
    )
    parser.add_argument(
        "--d_demo",
        type=int,
        default=32,
        help="",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Time Series Learning Rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopper"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
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


def forecast_parse_args():

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data_dir", type=str, help="A path to dataset folder"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=880,
        help="No. of obs in the dataset for each sequence",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="forecasting_model or mortality_model"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="forecasting_model or mortality_model"
    )
    parser.add_argument('--custom_strats', action='store_true')
    parser.add_argument('--special_transformer', action='store_true')
    parser.add_argument('--with_demo', action='store_true')
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
        "--lr",
        type=float,
        default=0.0005,
        help="Time Series Learning Rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience for early stopper"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=102400,
        help="Batch size",
    )

    args = parser.parse_args()

    return args


def mortality_loss(y_true, y_pred):
    class_weights = [0.56439914, 4.38203957]
    sample_weights = (1-y_true)*class_weights[0] + y_true*class_weights[1]
    bce = K.binary_crossentropy(y_true, y_pred)
    return K.mean(sample_weights*bce, axis=-1)


def build_forecast_loss(V):
    def forecast_loss(y_true, y_pred):
        return K.sum(y_true[:,V:]*(y_true[:,:V]-y_pred)**2, axis=-1)
    return forecast_loss


def get_min_loss(weight):
    def min_loss(y_true, y_pred):
        return weight*y_pred
    return min_loss


def get_res(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    minrp = np.minimum(precision, recall).max()
    roc_auc = roc_auc_score(y_true, y_pred)
    loss = K.eval(mortality_loss(K.variable(y_true), K.variable(y_pred.reshape(-1))))
    return [roc_auc, pr_auc, minrp, loss]


class CustomCallback(Callback):
    def __init__(self, validation_data, batch_size):
        self.val_x, self.val_y = validation_data
        self.batch_size = batch_size
        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.val_x, verbose=0, batch_size=self.batch_size)
        if type(y_pred)==type([]):
            y_pred = y_pred[0]
        precision, recall, thresholds = precision_recall_curve(self.val_y, y_pred)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(self.val_y, y_pred)
        logs['custom_metric'] = pr_auc + roc_auc
        loss = K.eval(mortality_loss(K.variable(self.val_y), K.variable(y_pred.reshape(-1))))
        logs['val_loss'] = loss
        logs['val_pr_auc'] = pr_auc
        logs['val_roc_auc'] = roc_auc
        print ('val_aucs:', pr_auc, roc_auc, loss, pr_auc+roc_auc)