from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

import training.loss_functions as lf

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

    if wrapper.model_name == "ffnn":
        scorer = None
    else:
        if wrapper.param_grid['loss_name'][0] == "mse":
            scorer = make_scorer(mean_squared_error)
        else:
            scorer = None

    if wrapper.search_type == 'random':
        model_search = RandomizedSearchCV(estimator=wrapper.ModelClass,
                                          param_distributions=wrapper.param_grid,
                                          n_iter=n_iter,
                                          cv=cv_splits,
                                          verbose=verbose,
                                          n_jobs=n_jobs,
                                          pre_dispatch=n_jobs,
                                          scoring=scorer,
                                          random_state=seed)
    elif wrapper.search_type == 'grid':
        model_search = GridSearchCV(estimator=wrapper.ModelClass,
                                    param_grid=wrapper.param_grid,
                                    cv=cv_splits,
                                    verbose=verbose,
                                    n_jobs=n_jobs,
                                    scoring=scorer)
    else:
        raise Exception('search type method not registered')

    model_search_output = model_search.fit(X=X,
                                           y=y.flatten(),
                                           epochs=wrapper.epochs,
                                           callbacks=wrapper.callbacks,
                                           validation_data=validation_data)

    return model_search_output

def train_and_evaluate(y_train, X_train, y_validation, X_validation, X_test, model_wrapper, criterion, verbose, trial):

    model_wrapper = model_wrapper(input_size=X_train.shape[1], trial=trial)
    
    model = model_wrapper.ModelClass
    param = model_wrapper.params
    epochs = model_wrapper.epochs

    y_train = torch.FloatTensor(y_train)
    X_train = torch.FloatTensor(X_train)

    y_validation = torch.FloatTensor(y_validation)
    X_validation = torch.FloatTensor(X_validation)

    X_test = torch.FloatTensor(X_test)

    optimizer = getattr(torch.optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])

    loss_values = []
    for i in tqdm(range(epochs), total=epochs, desc="Running backpropagation", disable=not verbose):
        # computer forward prediction
        y_hat = model.forward(X_train)

        # computes the loss function
        loss = criterion(y_hat, y_train)
        loss_values.append(loss.item())
        
        # report loss result
        trial.report(loss.item(), i)

        # set all previous gradients to zero
        optimizer.zero_grad()

        # backpropagation
        # computes gradient of current tensor given loss and opt procedure
        loss.backward()
        
        # update parameters
        optimizer.step()

    loss_values_df = pd.DataFrame(loss_values, columns=["loss"])
    trial.set_user_attr("loss_values", loss_values_df) 

    preds = []
    for val in X_validation:
        y_hat = model.forward(val)
        preds.append(y_hat.item())
    preds = torch.FloatTensor(preds).unsqueeze(0).t()
    test_loss = criterion(preds, y_validation)
    trial.set_user_attr("test_loss", test_loss) 

    test_preds = []
    for val in X_test:
        y_hat = model.forward(val)
        test_preds.append(y_hat.item())
    test_preds_df = torch.FloatTensor(test_preds).unsqueeze(0).t()
    trial.set_user_attr("test_predictions", test_preds_df) 

    return test_loss


def objective(y_train, X_train, y_validation, X_validation, X_test, model_wrapper, criterion, verbose, trial):
         
     loss = train_and_evaluate(y_train=y_train,
                               X_train=X_train,
                               y_validation=y_validation,
                               X_validation=X_validation,
                               X_test=X_test,
                               model_wrapper=model_wrapper,
                               criterion=criterion,
                               verbose=verbose,
                               trial=trial)

     return loss