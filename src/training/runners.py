import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.training.optimization import hyper_params_search

def run_model_training(target_name,
                       inputs_path,
                       outputs_path,
                       dataset_names,
                       model_tag,
                       standardize,
                       train_size,
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
            X_train = train_data[[target_name]].to_numpy()
            y_train = train_data.drop([target_name], axis=1).to_numpy()

            test_data = pd.read_csv(os.path.join(inputs_path, dir_name, d_name + "_test.csv"))
            test_data.set_index(["Var1", "Var2"], inplace=True)
            X_test = test_data[[target_name]].to_numpy()
            y_test = test_data.drop([target_name], axis=1).to_numpy()

            if standardize:
                X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, train_size=train_size)

                scaler = StandardScaler()
                X_train_zscore = scaler.fit_transform(X_train)
                X_validation_zscore = scaler.transform(X_validation)
                X_test_zscore = scaler.transform(X_test)

            wrapper = wrapper()

            if wrapper.search_type == "direct_fit":
                model_search = wrapper.ModelClass.fit(X=X_train_zscore,
                                                      y=y_train)
            else:
                model_search = hyper_params_search(X=X_train_zscore,
                                                   y=y_train,
                                                   validation_data=(X_validation_zscore, y_validation),
                                                   wrapper=wrapper,
                                                   n_jobs=n_jobs,
                                                   n_splits=n_splits,
                                                   n_iter=n_iter,
                                                   seed=seed,
                                                   verbose=verbose)

            test_pred = model_search.best_estimator_.predict(X_test_zscore)
            test_data = pd.DataFrame({"y": y_test,
                                      "pred": test_pred})

            # Check dir 1
            if not os.path.isdir(os.path.join(outputs_path, model_tag)):
                os.mkdir(os.path.join(outputs_path, model_tag))

            # Check dir 2
            if not os.path.isdir(os.path.join(outputs_path, model_tag, dir_name)):
                os.mkdir(os.path.join(outputs_path, model_tag, dir_name))

            test_data.reset_index().to_csv(os.path.join(outputs_path, model_tag, dir_name, d_name + "_result.csv"),
                                           index=False)