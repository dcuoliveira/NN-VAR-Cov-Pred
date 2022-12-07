from sklearn.linear_model import LinearRegression, LogisticRegression

class LinearRegWrapper():
    def __init__(self, model_params={'fit_intercept': False}):
        self.model_name = "linear_reg"
        self.search_type = 'direct_fit'
        self.param_grid = {}
        if model_params is None:
            self.ModelClass = LinearRegression()
        else:
            self.ModelClass = LinearRegression(**model_params)

class LogisticRegWrapper():
    def __init__(self, model_params={'fit_intercept': False, 'penalty': 'none', "solver": "lbfgs"}):
        self.model_name = "logit"
        self.search_type = 'direct_fit'
        self.param_grid = {}
        if model_params is None:
            self.ModelClass = LogisticRegression()
        else:
            self.ModelClass = LogisticRegression(**model_params)
