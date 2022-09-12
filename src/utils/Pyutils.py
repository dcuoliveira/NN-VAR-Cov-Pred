import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from training import loss_functions as lf

np.seterr(all="ignore")

def save_pkl(data,
             path):

    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

def load_pkl(path):

    with open(path, 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
    return data

def mse_melt(v):
    mse = mean_squared_error(y_true=v["y"],
                             y_pred=v["pred"])
    return pd.Series(dict(mse=mse))

def mae_melt(v):
    mae = mean_absolute_error(y_true=v["y"],
                             y_pred=v["pred"])
    return pd.Series(dict(mae=mae))

def wmse_melt(v):
    wmse = lf.weighted_mean_squared_error(y_true=v["y"],
                                          y_pred=v["pred"])
    return pd.Series(dict(wmse=wmse.numpy()))

def true_false_positive(threshold_vector, y_true):
    true_positive = np.equal(threshold_vector, 1) & np.equal(y_true, 1)
    true_negative = np.equal(threshold_vector, 0) & np.equal(y_true, 0)
    false_positive = np.equal(threshold_vector, 1) & np.equal(y_true, 0)
    false_negative = np.equal(threshold_vector, 0) & np.equal(y_true, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr

def roc_from_scratch(y_pred,
                     y_true,
                     partitions=100):
    
    y_pred = y_pred.abs()
    
    roc = np.array([])
    for i in np.linspace(start=0, stop=y_pred.abs().max(), num=partitions):
        
        threshold_vector = np.less_equal(y_pred, i).astype(int)
        tpr, fpr = true_false_positive(threshold_vector, y_true)
        roc = np.append(roc, [fpr, tpr])
        
    return roc.reshape(-1, 2)

DEBUG = False

if __name__ == '__main__':

    if DEBUG:
        import os

        data = {"a": 1, "b": 2}
        save_pkl(data=data,
                 path=os.path.join(os.getcwd(), "test.pickle"))

        new_data = load_pkl(path=os.path.join(os.getcwd(), "test.pickle"))