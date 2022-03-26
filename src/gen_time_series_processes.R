rm(list=ls())
gc()
library('SparseTSCGM')
library('dplyr')
library("here")

source(here("src", "utils.R"))

OUTPUT_PATH = here("src", "data", "inputs")
MODELS = c("ar1","ar2")
N = 100
K_INIT = 100
K = 150
J = 1
SEED = 02021994
PROB_OF_CONNECTION = 0.1
NETWORKS = c("random", "scale-free", "hub")

models_output = list()
for (model in MODELS){
  for (k in K_INIT:K){
    for (network in NETWORKS){
      output_name = paste0(model, "_", k, "_", network)
      mts = sim.data(model=model,
                     time=1,
                     n.obs=N, 
                     n.var=k,
                     seed=SEED,
                     prob0=PROB_OF_CONNECTION,
                     network=network)
      
      # betas_dgp = f(cov_dgp)
      betas_dgp = mts$gamma
      y_dgp = melt(betas_dgp) %>% as.data.table() %>% rename(betas_dgp=value)
      cov_dgp = melt(mts$sigma) %>% as.data.table() %>% rename(cov_dgp=value)
      betadgp_covdgp_data = merge(y_dgp, cov_dgp)
      mts[["betadgp_covdgp_data"]] = betadgp_covdgp_data
      
      # betas_dgp = f(betas_2x2)
      simulation_dgp = mts$data1 %>% longitudinal_to_data.table()
      beta2x2_data = lm_combination_2x2(data=simulation_dgp)
      betadgp_beta2x2_data = merge(y_dgp, beta2x2_data)
      mts[["betadgp_beta2x2_data"]] = betadgp_beta2x2_data
      
      models_output[[output_name]] = mts
    }
  }
}

saveRDS(models_output, here(OUTPUT_PATH, "VAR-DGPs.rds"))

