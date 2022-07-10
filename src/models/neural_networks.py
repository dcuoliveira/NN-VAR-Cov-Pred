from tensorflow import keras
import numpy as np
from scipy.stats import reciprocal

def FFNN(n_hidden,
         n_neurons,
         input_shape,
         learning_rate=3e-3,
         activation="relu",
         loss="mse"):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation=activation))
    model.add(keras.layers.Dense(1))

    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss=loss, optimizer=optimizer)

    return model


class FFNNWrapper():
    def __init__(self, model_params=None):
        self.model_name = "ffnn"
        self.search_type = 'random'
        self.param_grid = {"n_hidden": [1, 2, 3, 4, 5],
                           "n_neurons": np.arange(1, 100),
                           "learning_rate": reciprocal(3e-4, 3e-2),
                           "activation": ["relu"],
                           "loss": ["mse"]}
        self.ModelClass = keras.wrappers.scikit_learn.KerasRegressor(FFNN)

        if model_params is not None:
            self.param_grid.update(model_params)