import numpy as np

from sklearn.linear_model import LogisticRegression

class LogisticRegWrapper():
    def __init__(self, model_params={'fit_intercept': True, 'penalty': 'none'}):
        self.model_name = "logit"
        self.search_type = 'random'
        self.param_grid = {}
        if model_params is None:
            self.ModelClass = LogisticRegression()
        else:
            self.ModelClass = LogisticRegression(**model_params)


class LassoWrapper():
    def __init__(self, model_params={'fit_intercept': True, 'penalty': 'l1', 'solver': 'liblinear'}):
        self.model_name = "lasso"
        self.search_type = 'random'
        self.param_grid = {'C': np.linspace(0.001, 50, 200)}
        if model_params is None:
            self.ModelClass = LogisticRegression()
        else:
            self.ModelClass = LogisticRegression(**model_params)


class RidgeWrapper():
    def __init__(self, model_params={'fit_intercept': True, 'penalty': 'l2', 'solver': 'lbfgs'}):
        self.model_name = "ridge"
        self.search_type = 'random'
        self.param_grid = {'C': np.linspace(0.001, 50, 200)}
        if model_params is None:
            self.ModelClass = LogisticRegression()
        else:
            self.ModelClass = LogisticRegression(**model_params)


class ElasticNetWrapper():
    def __init__(self, model_params={'fit_intercept': True, 'penalty': 'elasticnet', 'solver': 'saga'}):
        self.model_name = "elastic_net"
        self.search_type = 'random'
        self.param_grid = {'C': np.linspace(0.001, 50, 200),
                           'l1_ratio': np.linspace(0.001, 0.999, 200)}
        if model_params is None:
            self.ModelClass = LogisticRegression()
        else:
            self.ModelClass = LogisticRegression(**model_params)