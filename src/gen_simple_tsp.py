import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

alpha = 0.05
i = 0
summary = {}
while i < 10000:

    arparams = np.array([np.random.uniform(low=-1, high=1, size=1)[0], np.random.uniform(low=-1, high=1, size=1)[0]])
    ar = np.r_[1, -arparams]
    maparams = np.array([0, 0])
    ma = np.r_[1, maparams]
    y = sm.tsa.arma_generate_sample(ar, ma, 100)

    if sm.tsa.stattools.adfuller(y)[1] <= alpha:
        summary[i] = {"y": y, "coefs": arparams}
        i += 1

target_path = os.path.join(os.getcwd(), "src", "data", "inputs", "simple_ar")

if not os.path.isdir(os.path.join(target_path)):
    os.mkdir(os.path.join(target_path))

j = 0
k = 0
m = 0

betadgp_beta2x2 = []
cov2x2 = []
corr2x2 = []
for i in range(len(summary.keys())):
    data = pd.DataFrame(summary[i]["y"], columns=["y"])

    for l in range(1, len(summary[i]["coefs"]) + 1):
        data.loc[:, "y_lag{}".format(str(l))] = data["y"].shift(l) 

    data = data.dropna()

    # beta 2x2
    for l in range(1, len(summary[i]["coefs"]) + 1):
        ols_fit = sm.OLS(endog=data["y"], exog=data["y_lag{}".format(str(l))]).fit()
        betadgp_beta2x2.append(pd.DataFrame({j: {"betas_dgp": summary[i]["coefs"][l-1], "beta_2x2": ols_fit.params[0]}}).T)
        j += 1

    # covariance 2x2
    for l in range(1, len(summary[i]["coefs"]) + 1):
        cov2x2.append(pd.DataFrame({k: {"betas_dgp": summary[i]["coefs"][l-1], "cov_dgp": data.cov().iloc[0][l]}}).T)
        k += 1

    # correlation 2x2
    for l in range(1, len(summary[i]["coefs"]) + 1):
        corr2x2.append(pd.DataFrame({k: {"betas_dgp": summary[i]["coefs"][l-1], "corr_dgp": data.corr().iloc[0][l]}}).T)
        m += 1

betadgp_beta2x2_df = pd.concat(betadgp_beta2x2, axis=0)
cov2x2_df = pd.concat(cov2x2, axis=0)
corr2x2_df = pd.concat(corr2x2, axis=0)

corr2x2_df.reset_index().reset_index().rename(columns={"level_0": "Var1", "index": "Var2"}).to_csv(os.path.join(target_path, "betadgp_corrdgp_data_train.csv"), index=False)
cov2x2_df.reset_index().reset_index().rename(columns={"level_0": "Var1", "index": "Var2"}).to_csv(os.path.join(target_path, "betadgp_covdgp_data_train.csv"), index=False)
betadgp_beta2x2_df.reset_index().reset_index().rename(columns={"level_0": "Var1", "index": "Var2"}).to_csv(os.path.join(target_path, "betadgp_beta2x2_data_train.csv"), index=False)

j = 0
k = 0
m = 0

betadgp_beta2x2 = []
cov2x2 = []
corr2x2 = []
for i in range(len(summary.keys())):
    data = pd.DataFrame(summary[i]["y"], columns=["y"])

    for l in range(1, len(summary[i]["coefs"]) + 1):
        data.loc[:, "y_lag{}".format(str(l))] = data["y"].shift(l) 

    data = data.dropna()

    # beta 2x2
    for l in range(1, len(summary[i]["coefs"]) + 1):
        ols_fit = sm.OLS(endog=data["y"], exog=data["y_lag{}".format(str(l))]).fit()
        betadgp_beta2x2.append(pd.DataFrame({j: {"betas_dgp": summary[i]["coefs"][l-1], "beta_2x2": ols_fit.params[0]}}).T)
        j += 1

    # covariance 2x2
    for l in range(1, len(summary[i]["coefs"]) + 1):
        cov2x2.append(pd.DataFrame({k: {"betas_dgp": summary[i]["coefs"][l-1], "cov_dgp": data.cov().iloc[0][l]}}).T)
        k += 1

    # correlation 2x2
    for l in range(1, len(summary[i]["coefs"]) + 1):
        corr2x2.append(pd.DataFrame({k: {"betas_dgp": summary[i]["coefs"][l-1], "corr_dgp": data.corr().iloc[0][l]}}).T)
        m += 1

betadgp_beta2x2_df = pd.concat(betadgp_beta2x2, axis=0)
cov2x2_df = pd.concat(cov2x2, axis=0)
corr2x2_df = pd.concat(corr2x2, axis=0)

corr2x2_df.reset_index().reset_index().rename(columns={"level_0": "Var1", "index": "Var2"}).to_csv(os.path.join(target_path, "betadgp_corrdgp_data_test.csv"), index=False)
cov2x2_df.reset_index().reset_index().rename(columns={"level_0": "Var1", "index": "Var2"}).to_csv(os.path.join(target_path, "betadgp_covdgp_data_test.csv"), index=False)
betadgp_beta2x2_df.reset_index().reset_index().rename(columns={"level_0": "Var1", "index": "Var2"}).to_csv(os.path.join(target_path, "betadgp_beta2x2_data_test.csv"), index=False)