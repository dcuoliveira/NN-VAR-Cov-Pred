from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import SGD

from training import loss_functions as lf

def FFNN(n_hidden,
         n_neurons,
         input_shape,
         learning_rate,
         activation,
         loss_name):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation=activation))
    model.add(Dense(1))

    if loss_name == "mse":
        loss = loss_name
    elif loss_name == "wmse":
        loss = lf.weighted_mean_squared_error

    optimizer = SGD(lr=learning_rate)
    model.compile(loss=loss, optimizer=optimizer)

    return model


class FFNNWrapper():
    def __init__(self, model_params=None):
        self.model_name = "ffnn"
        self.search_type = 'random'
        self.param_grid = {"n_hidden": np.arange(1, 10+1),
                           "n_neurons": np.arange(1, 100+1),
                           "learning_rate": list(1e-3 * (10 ** (np.arange(100) / 30))),
                           "activation": ["relu"],
                           "loss_name": ["mse"]}
        self.epochs = 50
        self.callbacks = [keras.callbacks.EarlyStopping(patience=10)]

        self.ModelClass = keras.wrappers.scikit_learn.KerasRegressor(FFNN, verbose=0)

        if model_params is not None:
            self.param_grid.update(model_params)


class FFNNWrapperWMSE():
    def __init__(self, model_params=None):
        self.model_name = "ffnn"
        self.search_type = 'random'
        self.param_grid = {"n_hidden": np.arange(1, 10+1),
                           "n_neurons": np.arange(1, 100+1),
                           "learning_rate": list(1e-3 * (10 ** (np.arange(100) / 30))),
                           "activation": ["relu"],
                           "loss_name": ["wmse"]}
        self.epochs = 50
        self.callbacks = [keras.callbacks.EarlyStopping(patience=10)]

        self.ModelClass = keras.wrappers.scikit_learn.KerasRegressor(FFNN, verbose=0)

        if model_params is not None:
            self.param_grid.update(model_params)


class DNN1Wrapper():
    def __init__(self, model_params=None):
        self.model_name = "ffnn"
        self.search_type = 'random'
        self.param_grid = {"n_hidden": np.arange(1, 1000+1),
                           "n_neurons": np.arange(1, 1000+1),
                           "learning_rate": list(1e-3 * (10 ** (np.arange(100) / 30))),
                           "activation": ["relu"],
                           "loss_name": ["mse"]}
        self.epochs = 50
        self.callbacks = [keras.callbacks.EarlyStopping(patience=10)]

        self.ModelClass = keras.wrappers.scikit_learn.KerasRegressor(FFNN, verbose=0)

        if model_params is not None:
            self.param_grid.update(model_params)


class FFNNFixedWrapper():
    def __init__(self, model_params=None):
        self.model_name = "ffnn"
        self.search_type = 'random'
        self.param_grid = {"n_hidden": None,
                           "n_neurons": None,
                           "learning_rate": list(1e-3 * (10 ** (np.arange(100) / 30))),
                           "activation": None,
                           "loss_name": ["mse"]}
        self.epochs = 50
        self.callbacks = [keras.callbacks.EarlyStopping(patience=10)]

        self.ModelClass = keras.wrappers.scikit_learn.KerasRegressor(FFNN, verbose=0)

        if model_params is not None:
            self.param_grid.update(model_params)