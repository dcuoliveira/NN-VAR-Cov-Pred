from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold

def hyper_params_search(X,
                        y,
                        validation_data,
                        wrapper,
                        n_iter,
                        n_splits,
                        n_jobs,
                        verbose,
                        seed):
    """
    Use the dataframe 'df' to search for the best
    params for the model 'wrapper'.
    The CV split is performed using the TimeSeriesSplit
    class.
    We can define the size of the test set using the formula
    ``n_samples//(n_splits + 1)``,
    where ``n_samples`` is the number of samples. Hence,
    we can define
    n_splits = (n - test_size) // test_size
    :param df: train data
    :type df: pd.DataFrame
    :param wrapper: predictive model
    :type wrapper: sklearn model wrapper
    :param n_iter: number of hyperparameter searchs
    :type n_iter: int
    :param n_splits: number of splits for the cross-validation
    :type n_splits: int
    :param n_jobs: number of concurrent workers
    :type n_jobs: int
    :param verbose: param to print iteration status
    :type verbose: bool, int
    :param target_name: name of the target column in 'df'
    :type target_name: str
    :return: R2 value
    :rtype: float
    """

    cv_splits = KFold(n_splits=n_splits)
    mse_scorer = make_scorer(mean_squared_error)

    if wrapper.search_type == 'random':
        model_search = RandomizedSearchCV(estimator=wrapper.ModelClass,
                                          param_distributions=wrapper.param_grid,
                                          n_iter=n_iter,
                                          cv=cv_splits,
                                          verbose=verbose,
                                          n_jobs=n_jobs,
                                          scoring=mse_scorer,
                                          random_state=seed)
    elif wrapper.search_type == 'grid':
        model_search = GridSearchCV(estimator=wrapper.ModelClass,
                                    param_grid=wrapper.param_grid,
                                    cv=cv_splits,
                                    verbose=verbose,
                                    n_jobs=n_jobs,
                                    scoring=mse_scorer)
    else:
        raise Exception('search type method not registered')

    model_search = model_search.fit(X, y)

    return model_search
