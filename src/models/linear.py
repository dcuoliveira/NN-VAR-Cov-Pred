import numpy as np

from sklearn.linear_model import LinearRegression

class LinearReWrapper():
    def __init__(self, model_params={'fit_intercept': True}):
        self.model_name = "linear_reg"
        self.search_type = 'direct_fit'
        self.param_grid = {}
        if model_params is None:
            self.ModelClass = LinearRegression()
        else:
            self.ModelClass = LinearRegression(**model_params)
