import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import traceback
import numpy as np

from training import optimization as opt
from utils import Pyutils as pyutils

def run_model_training(target_name,
                       inputs_path,
                       outputs_path,
                       log_path,
                       dataset_names,
                       model_tag,
                       standardize,
                       train_size,
                       wrapper,
                       wrapper_ovrd,
                       n_jobs,
                       n_splits,
                       n_iter,
                       seed,
                       verbose,
                       output_ovrd,
                       dir_name_ovrd=None,
                       classification=False):

    # check if output dir for model_tag exists
    if not os.path.isdir(os.path.join(outputs_path, model_tag)):
        os.mkdir(os.path.join(outputs_path, model_tag))

    # check if log dir for model_tag exists
    if not os.path.isdir(os.path.join(log_path, model_tag)):
        os.mkdir(os.path.join(log_path, model_tag))

    if dir_name_ovrd is not None:
        list_dir_names = dir_name_ovrd
    else:
        list_dir_names = os.listdir(inputs_path)

    for dir_name in tqdm(list_dir_names,
                         desc="Running " + model_tag + " model for all DGPs"):
        for d_name in dataset_names:

            if output_ovrd:
                check_pickle = os.path.exists(os.path.join(outputs_path, model_tag, dir_name, d_name + "_model.pickle"))
                check_pred = os.path.exists(os.path.join(outputs_path, model_tag, dir_name, d_name + "_result.csv"))
                if check_pickle and check_pred:
                    continue

            train_data = pd.read_csv(os.path.join(inputs_path, dir_name, d_name + ".csv"))
            train_data.set_index(["Var1", "Var2"], inplace=True)
            y_train = train_data[[target_name]].to_numpy()
            X_train = train_data.drop([target_name], axis=1).to_numpy()

            test_data = pd.read_csv(os.path.join(inputs_path, dir_name, d_name + "_test.csv"))
            test_data.set_index(["Var1", "Var2"], inplace=True)
            y_test = test_data[[target_name]].to_numpy()
            X_test = test_data.drop([target_name], axis=1).to_numpy()

            if classification:
                y_train = np.where(train_data[[target_name]].to_numpy().__abs__() > 0, 1, 0)
                y_test = np.where(test_data[[target_name]].to_numpy().__abs__() > 0, 1, 0)

            if standardize:
                X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, train_size=train_size)

                scaler = StandardScaler()
                X_train_zscore = scaler.fit_transform(X_train)
                X_validation_zscore = scaler.transform(X_validation)
                X_test_zscore = scaler.transform(X_test)

            # check which model we will run
            if ("ffnn" in model_tag):
                if wrapper_ovrd is not None:
                    ModelWrapper = wrapper(model_params={"input_shape": [X_train.shape[1]],
                                                         "n_hidden": [wrapper_ovrd["n_hidden"]],
                                                         "n_neurons": [wrapper_ovrd["n_neurons"]],
                                                         "activation": [wrapper_ovrd["activation"]],
                                                         "loss_name": [wrapper_ovrd["loss_name"]]})
                else:
                    ModelWrapper = wrapper(model_params={"input_shape": [X_train.shape[1]]})
            else:
                ModelWrapper = wrapper()

            # check nan's in dataset
            if np.any(np.isnan(X_train_zscore)) or np.any(np.isnan(X_validation_zscore)) or np.any(np.isnan(y_train)) or np.any(np.isnan(y_validation)):
                print(d_name + " " + dir_name)
                break

            if ModelWrapper.search_type == "direct_fit":
                model_search = ModelWrapper.ModelClass.fit(X=X_train_zscore,
                                                           y=y_train)
                test_pred = model_search.predict_proba(X_test_zscore)[:, 1]

            else:
                try:
                    model_search = opt.hyper_params_search(X=X_train_zscore,
                                                           y=y_train,
                                                           validation_data=(X_validation_zscore, y_validation),
                                                           wrapper=ModelWrapper,
                                                           n_jobs=n_jobs,
                                                           n_splits=n_splits,
                                                           n_iter=n_iter,
                                                           seed=seed,
                                                           verbose=verbose)
                    test_pred = model_search.best_estimator_.predict_proba(X_test_zscore)
                except:
                    str_traceback = traceback.format_exc()

                    # check if dir exists
                    if not os.path.isdir(os.path.join(log_path, model_tag, dir_name)):
                        os.mkdir(os.path.join(log_path, model_tag, dir_name))

                    # open file
                    log_file = open(os.path.join(log_path, model_tag, dir_name, d_name + ".log"), "w")

                    # write error
                    log_file.write(str(str_traceback))

                    # close file
                    log_file.close()

                    continue

            output = pd.DataFrame({"Var1": test_data.reset_index()["Var1"],
                                   "Var2": test_data.reset_index()["Var2"],
                                   "y": y_test.ravel(),
                                   "pred": test_pred.ravel()})
            model_output = {"model_search": model_search}

            # check if output dir for model_tag AND dir_name exists
            if not os.path.isdir(os.path.join(outputs_path, model_tag, dir_name)):
                os.mkdir(os.path.join(outputs_path, model_tag, dir_name))

            output.to_csv(os.path.join(outputs_path, model_tag, dir_name, d_name + "_result.csv"), index=False)

            if ModelWrapper.search_type == "direct_fit":
                out = {"coef": model_search.coef_}
                pyutils.save_pkl(data=out,
                                 path=os.path.join(outputs_path, model_tag, dir_name, d_name + "_model.pickle"))
            else:
                out = model_search.best_params_
                pyutils.save_pkl(data=out,
                                 path=os.path.join(outputs_path, model_tag, dir_name, d_name + "_model.pickle"))