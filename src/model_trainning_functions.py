import os
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold


def hyper_params_search(df,
                        wrapper,
                        n_iter,
                        n_splits,
                        n_jobs,
                        verbose,
                        seed,
                        target_name):
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

    wrapper = wrapper()

    X = df.drop(target_name, 1).values
    y = df[target_name].values

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


def run_train_test_setup(target_name,
                         inputs_path,
                         outputs_path,
                         dataset_names,
                         model_tag,
                         wrapper,
                         n_jobs,
                         n_splits,
                         n_iter,
                         seed,
                         verbose):

    for dir_name in os.listdir(inputs_path):
        for d_name in dataset_names:
            train_data = pd.read_csv(os.path.join(inputs_path, dir_name, d_name + ".csv"))
            train_data.set_index(["Var1", "Var2"], inplace=True)

            test_data = pd.read_csv(os.path.join(inputs_path, dir_name, d_name + "_test.csv"))
            test_data.set_index(["Var1", "Var2"], inplace=True)

            model_search = hyper_params_search(df=train_data,
                                               target_name=target_name,
                                               wrapper=wrapper,
                                               n_jobs=n_jobs,
                                               n_splits=n_splits,
                                               n_iter=n_iter,
                                               seed=seed,
                                               verbose=verbose)

            X_test = test_data.drop(target_name, 1).values
            test_pred = model_search.best_estimator_.predict(X_test)

            test_data['prediction'] = test_pred

            # Check dir 1
            if not os.path.isdir(os.path.join(outputs_path, model_tag)):
                os.mkdir(os.path.join(outputs_path, model_tag))

            # Check dir 2
            if not os.path.isdir(os.path.join(outputs_path, model_tag, dir_name)):
                os.mkdir(os.path.join(outputs_path, model_tag, dir_name))

            test_data.reset_index().to_csv(os.path.join(outputs_path, model_tag, dir_name, d_name + "_result.csv"))
