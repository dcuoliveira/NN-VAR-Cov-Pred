import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from training import loss_functions as lf

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

DEBUG = False

if __name__ == '__main__':

    if DEBUG:
        import os

        data = {"a": 1, "b": 2}
        save_pkl(data=data,
                 path=os.path.join(os.getcwd(), "test.pickle"))

        new_data = load_pkl(path=os.path.join(os.getcwd(), "test.pickle"))