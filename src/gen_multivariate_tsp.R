rm(list = ls())
gc()
library("sparsevar")
library("dplyr")
library("data.table")
library("foreach")
library("stringr")
library("doParallel")

FILE_PATH <- getwd()

source(file.path(FILE_PATH, "src", "utils", "Rutils.R"))

OUTPUT_PATH <- file.path(FILE_PATH, "src", "data", "inputs")
P <- 2
N <- 100
K_INIT <- 10
K <- 200
STEP <- 20
PROB_OF_CONNECTION <- c(0.05, 0.9)

seqs <- c(10, 200) # seq(K_INIT, K, by = STEP)

cores <- detectCores()
cl <- makeCluster(cores[1] - 1)
registerDoParallel(cl)

foreach(prob = PROB_OF_CONNECTION) %dopar% {

  library("sparsevar")
  library("dplyr")
  library("data.table")
  library("stringr")

  for (p in 1:P){
    for (k in seqs){
      output_name <- paste0("var_", prob, "_", p, "_", k)
      new_folder <- file.path(OUTPUT_PATH, output_name)
      dir.create(new_folder)
      
      ## TRAIN ##
      
      mts <- simulateVAR(N = k, p = p, nobs = N, sparsity=prob)
      
      # simulated time series
      dgp_data <- mts$series %>% as.data.table()
      fwrite(x = dgp_data,
             file = file.path(new_folder, "data_dgp_train.csv"),
             row.names = FALSE)
      
      # dgp covariance matrix ?
      sigma <- mts$sigma %>% as.data.table()
      fwrite(x = sigma,
             file = file.path(new_folder, "sigma_train.csv"),
             row.names = FALSE)
      
      # dgp betas
      betas <- mts$A %>% as.data.table()
      colnames(betas) <- stardize_var_lag_names(k = k, p = p)
      betas$eq <- stardize_var_names(k = k)
      betas <- betas %>% dplyr::select(eq, everything())
      fwrite(x = betas,
             file = file.path(new_folder, "beta_train.csv"),
             row.names = FALSE)
      
      # sample covariance matrix
      y_dgp <- melt(betas, id = c("eq")) %>%
        as.data.table() %>%
        rename(beta = value) %>%
        mutate(eq = as.character(eq), variable=as.character(variable))
      cov_dgp <- cov_combination_2x2(data = dgp_data, p = p)
      betadgp_covdgp_data <- merge(y_dgp, cov_dgp)
      fwrite(x = betadgp_covdgp_data,
             file = file.path(new_folder, "betadgp_covdgp_data_train.csv"),
             row.names = FALSE)
      
      # sample correlation matrix
      corr_dgp <- corr_combination_2x2(data = dgp_data, p = p)
      betadgp_corrdgp_data <- merge(y_dgp, corr_dgp)
      fwrite(x = betadgp_corrdgp_data,
             file = file.path(new_folder, "betadgp_corrdgp_data_train.csv"),
             row.names = FALSE)
      
      # beta 2x2 of each time series and lags
      beta2x2_data <- lm_combination_2x2(data = dgp_data, p = p)
      betadgp_beta2x2_data <- merge(y_dgp, beta2x2_data)
      fwrite(x = betadgp_beta2x2_data,
             file = file.path(new_folder, "betadgp_beta2x2_data_train.csv"),
             row.names = FALSE)
      
      # all covariates
      betadgp_data <- merge(merge(betadgp_covdgp_data,
                                  corr_dgp,
                                  by = c("eq", "variable")),
                            beta2x2_data,
                            by = c("eq", "variable"))
      fwrite(x = betadgp_data,
             file = file.path(new_folder, "betadgp_data_train.csv"),
             row.names = FALSE)
      
      ## TEST ##
      
      # data - test
      mts_test <- simulateVAR(N = k, p = p, nobs = N, sparsity=prob)
      
      # simulated time series - test
      dgp_data_test <- mts_test$series %>% as.data.table()
      fwrite(x = dgp_data_test,
             file = file.path(new_folder, "data_dgp_test.csv"),
             row.names = FALSE)
      
      # betas - test
      betas_test <- mts_test$A %>% as.data.table()
      colnames(betas_test) <- stardize_var_lag_names(k = k, p = p)
      betas_test$eq <- stardize_var_names(k = k)
      betas_test <- betas_test %>% dplyr::select(eq, everything())
      fwrite(x = betas_test,
             file = file.path(new_folder, "beta_test.csv"),
             row.names = FALSE)
      
      # sample covariance matrix - test
      y_dgp_test <- melt(betas_test, id = c("eq")) %>%
        as.data.table() %>%
        rename(beta = value) %>%
        mutate(eq = as.character(eq), variable=as.character(variable))
      cov_dgp_test <- cov_combination_2x2(data = dgp_data_test, p = p)
      betadgp_covdgp_data_test <- merge(y_dgp_test, cov_dgp_test)
      fwrite(x = betadgp_covdgp_data_test,
             file = file.path(new_folder, "betadgp_covdgp_data_test.csv"),
             row.names = FALSE)
      
      # sample correlation matrix - test
      corr_dgp_test <- corr_combination_2x2(data = dgp_data_test, p = p)
      betadgp_corrdgp_data_test <- merge(y_dgp_test, corr_dgp_test)
      fwrite(x = betadgp_corrdgp_data_test,
             file = file.path(new_folder, "betadgp_corrdgp_data_test.csv"),
             row.names = FALSE)
      
      # beta 2x2 of each time series and lags - test
      beta2x2_data_test <- lm_combination_2x2(data = dgp_data_test, p = p)
      betadgp_beta2x2_data_test <- merge(y_dgp_test, beta2x2_data_test)
      fwrite(x = betadgp_beta2x2_data_test,
             file = file.path(new_folder, "betadgp_beta2x2_data_test.csv"),
             row.names = FALSE)
      
      # all covariates - test
      betadgp_data_test <- merge(merge(betadgp_covdgp_data_test,
                                  corr_dgp_test,
                                  by = c("eq", "variable")),
                            beta2x2_data_test,
                            by = c("eq", "variable"))
      fwrite(x = betadgp_data_test,
             file = file.path(new_folder, "betadgp_data_test.csv"),
             row.names = FALSE)
    }
  }
}

stopCluster(cl)