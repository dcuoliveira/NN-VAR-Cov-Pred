rm(list=ls())
gc()
library('SparseTSCGM')
library('dplyr')
library("here")

source(here("src", "utils.R"))

OUTPUT_PATH = here("src", "data", "inputs")
MODELS = c("ar1")
N = 100
K_INIT = 150
K = 200
J = 1
SEED = 02021994
PROB_OF_CONNECTION = 0.1
NETWORKS = c("random")

for (model in MODELS){
  for (k in K_INIT:K){
    for (network in NETWORKS){
      output_name = paste0(model, "_", k, "_", network)
      new_folder = here(OUTPUT_PATH, output_name)
      dir.create(new_folder)
      
      mts = sim.data(model=model,
                     time=1,
                     n.obs=N, 
                     n.var=k,
                     seed=SEED,
                     prob0=PROB_OF_CONNECTION,
                     network=network)
      sigma = mts$sigma %>% as.data.table()
      colnames(sigma) = gsub("V", "", colnames(sigma))
      fwrite(x = sigma,
             file = here(new_folder, "sigma.csv"),
             row.names = FALSE)
      beta = mts$gamma %>% as.data.table()
      colnames(beta) = gsub("V", "", colnames(beta))
      fwrite(x = beta,
             file = here(new_folder, "beta.csv"),
             row.names = FALSE)
      
      # betas_dgp = f(cov_dgp)
      betas_dgp = mts$gamma
      y_dgp = melt(betas_dgp) %>% as.data.table() %>% rename(betas_dgp=value)
      cov_dgp = melt(mts$sigma) %>% as.data.table() %>% rename(cov_dgp=value)
      betadgp_covdgp_data = merge(y_dgp, cov_dgp)
      fwrite(x = betadgp_covdgp_data,
             file = here(new_folder, "betadgp_covdgp_data.csv"),
             row.names = FALSE)
      
      # betas_dgp = f(betas_2x2)
      simulation_dgp = mts$data1 %>% longitudinal_to_data.table()
      beta2x2_data = lm_combination_2x2(data=simulation_dgp)
      betadgp_beta2x2_data = merge(y_dgp, beta2x2_data)
      fwrite(x = betadgp_covdgp_data,
             file = here(new_folder, "betadgp_beta2x2_data.csv"),
             row.names = FALSE)
      
      # all covariates
      betadgp_data = merge(betadgp_covdgp_data, beta2x2_data)
      fwrite(x = betadgp_covdgp_data,
             file = here(new_folder, "betadgp_data.csv"),
             row.names = FALSE)
      
      # test data
      mts_test = sim.data(model=model,
                          time=1,
                          n.obs=N, 
                          n.var=k,
                          prob0=PROB_OF_CONNECTION,
                          network=network)
      betas_dgp_test = mts_test$gamma
      y_dgp_test = melt(betas_dgp_test) %>% as.data.table() %>% rename(betas_dgp=value)
      cov_dgp_test = melt(mts_test$sigma) %>% as.data.table() %>% rename(cov_dgp=value)
      betadgp_covdgp_data_test = merge(y_dgp_test, cov_dgp_test)
      fwrite(x = betadgp_covdgp_data_test,
             file = here(new_folder, "betadgp_covdgp_data_test.csv"),
             row.names = FALSE)
      simulation_dgp_test = mts_test$data1 %>% longitudinal_to_data.table()
      beta2x2_data_test = lm_combination_2x2(data=simulation_dgp_test)
      betadgp_beta2x2_data_test = merge(y_dgp_test, beta2x2_data_test)
      fwrite(x = betadgp_beta2x2_data_test,
             file = here(new_folder, "betadgp_beta2x2_data_test.csv"),
             row.names = FALSE)
      betadgp_data_test = merge(betadgp_covdgp_data_test, betadgp_beta2x2_data_test)
      fwrite(x = betadgp_data_test,
             file = here(new_folder, "betadgp_data_test.csv"),
             row.names = FALSE)
      
    }
  }
}

