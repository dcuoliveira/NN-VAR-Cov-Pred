from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor

class LayerSizeGenerator:

    def __init__(self):
        self.num_layers = [1, 2]
        self.num_neurons = np.arange(1, 15+1, 1)

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

class RandomForestWrapper():
    def __init__(self, model_params=None):
        self.model_name = "random_forest"
        self.search_type = 'random'
        self.param_grid = {"max_features": ['auto', 'sqrt', 'log2'],
                           "min_samples_split": sp_randint(2, 31),
                           "n_estimators": sp_randint(2, 301),
                           "max_depth": sp_randint(2, 20)}
        if model_params is None:
            self.ModelClass = RandomForestRegressor()
        else:
            self.ModelClass = RandomForestRegressor(**model_params)


class LGBWrapper():
    def __init__(self, model_params=None):
        self.model_name = "lgb_regression"
        self.search_type = 'random'
        self.param_grid = {'num_leaves': sp_randint(6, 50),
                           'min_child_samples': sp_randint(100, 500),
                           'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                           'subsample': sp_uniform(loc=0.2, scale=0.8),
                           "n_estimators": sp_randint(500, 1000),
                           "max_depth": sp_randint(3, 100),
                           "learning_rate": np.linspace(0.001, 0.99, 100),
                           'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                           'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                           'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
                           "objective": ["huber"]}
        if model_params is None:
            self.ModelClass = LGBMRegressor()
        else:
            self.ModelClass = LGBMRegressor(**model_params)