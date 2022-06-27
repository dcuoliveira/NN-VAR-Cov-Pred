import os
import pandas as pd

from src.training.optimization import hyper_params_search

def run_training_test_setup(target_name,
                            inputs_path,
                            outputs_path,
                            dataset_names,
                            model_tag,
                            standardize,
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

            if standardize:
                train_mean = train_data.mean(axis=0)
                train_std = train_data.std(axis=0)
                orig_train_data = train_data.copy()
                train_data = (train_data - train_mean) / train_std

                test_mean = test_data.mean(axis=0)
                test_std = test_data.std(axis=0)
                orig_test_data = test_data.copy()
                test_data = (test_data - test_mean) / test_std


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

            test_mean['prediction'] = test_mean[target_name]
            test_std['prediction'] = test_std[target_name]

            test_data['prediction'] = test_pred

            if standardize:
                test_data = test_data * test_std - test_mean

            # Check dir 1
            if not os.path.isdir(os.path.join(outputs_path, model_tag)):
                os.mkdir(os.path.join(outputs_path, model_tag))

            # Check dir 2
            if not os.path.isdir(os.path.join(outputs_path, model_tag, dir_name)):
                os.mkdir(os.path.join(outputs_path, model_tag, dir_name))

            test_data.reset_index().to_csv(os.path.join(outputs_path, model_tag, dir_name, d_name + "_result.csv"),
                                           index=False)