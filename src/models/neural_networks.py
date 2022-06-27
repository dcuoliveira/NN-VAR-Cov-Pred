import random
import numpy as np

from sklearn.neural_network import MLPRegressor

class LayerSizeGenerator:

    def __init__(self):
        self.num_layers = [1, 2, 3, 4, 5, 5, 7, 8, 9, 10]
        self.num_neurons = np.arange(1, 50+1, 1)

    def rvs(self, random_state=42):
        random.seed(random_state)
        # first randomly define num of layers, then pick the neuron size for each of them
        num_layers = random.choice(self.num_layers)
        layer_sizes = random.choices(self.num_neurons, k=num_layers)
        return layer_sizes


class NNCombWrapper():
    def __init__(self, model_params=None):
        self.model_name = "nncomb"
        self.search_type = 'random'
        self.param_grid = {"early_stopping": [True],
                           "learning_rate": ["invscaling"],
                           "learning_rate_init": np.linspace(0.001, 0.999, 100),
                           'alpha': np.linspace(0.001, 0.999, 100),
                           'solver': ["adam"],
                           'activation': ["relu"],
                           "hidden_layer_sizes": LayerSizeGenerator()}
        if model_params is None:
            self.ModelClass = MLPRegressor()
        else:
            self.ModelClass = MLPRegressor(**model_params)