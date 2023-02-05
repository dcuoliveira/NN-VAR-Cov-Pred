rm(list = ls())
gc()
library("SparseTSCGM")
library("dplyr")
library("data.table")
library("foreach")
library("doParallel")
library("stringr")

FILE_PATH <- getwd() # nolint

source(file.path(FILE_PATH, "src", "utils", "Rutils.R"))

OUTPUT_PATH <- file.path(FILE_PATH, "src", "data", "inputs") # nolint
MODELS <- c("ar1") # nolint
N <- 100 # nolint
K_INIT <- 150 # nolint
K <- 300 # nolint
STEP <- 50 # nolint
SEED <- 02021994 # nolint
PROB_OF_CONNECTION <- 0.1 # nolint
NETWORKS <- c("random") # nolint

cores <- detectCores()
cl <- makeCluster(cores[1] - 1)
registerDoParallel(cl)

foreach(model = MODELS)  %dopar% {

  library("SparseTSCGM")
  library("dplyr")
  library("data.table")
  library("stringr")

  for (k in seq(K_INIT, K, by = STEP)){
    for (network in NETWORKS){

       output_name <- paste0(model, "_", k, "_", network)
       new_folder <- file.path(OUTPUT_PATH, output_name)
       dir.create(new_folder)

       mts <- sim.data(model = model,
                      time = 1,
                      n.obs = N,
                      n.var = k,
                      seed = SEED,
                      prob0 = 0.9,
                      network = network)
        p <- as.integer(str_split(model, "ar")[[1]][2])

       # simulated time series
       dgp_data <- mts$data1 %>% longitudinal_to_data.table()
       fwrite(x = dgp_data,
              file = file.path(new_folder, "data_dgp_train.csv"),
              row.names = FALSE)

       # dgp covariance matrix ?
       sigma <- mts$sigma %>% as.data.table()
       colnames(sigma) <- gsub("V", "", colnames(sigma))
       fwrite(x = sigma,
              file = file.path(new_folder, "sigma_train.csv"),
              row.names = FALSE)

       # dgp betas
       beta <- mts$gamma %>% t() %>% as.data.table()
       colnames(beta) <- gsub("V", "", colnames(beta))
       fwrite(x = beta,
              file = file.path(new_folder, "beta_train.csv"),
              row.names = FALSE)

       # sample covariance matrix
       betas_dgp <- beta %>%
        mutate(eq = row.names(beta)) %>%
         dplyr::select(eq, everything())
       y_dgp <- melt(betas_dgp, id = c("eq")) %>%
        as.data.table() %>%
         rename(betas_dgp = value) %>%
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
       fwrite(x = betadgp_covdgp_data,
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

       # test data
       mts_test <- sim.data(model = model,
                           time = 1,
                           n.obs = N,
                           n.var = k,
                           prob0 = PROB_OF_CONNECTION,
                           network = network)
       betas_dgp_test <- mts_test$gamma

       y_dgp_test <- melt(betas_dgp_test) %>%
        as.data.table() %>%
        rename(eq = Var1, variable = Var2, betas_dgp = value) %>%
         mutate(eq = as.character(eq), variable = as.character(variable))

       dgp_data_test <- mts_test$data1 %>%
        longitudinal_to_data.table()
       fwrite(x = dgp_data_test,
              file = file.path(new_folder, "data_dgp_test.csv"),
              row.names = FALSE)

        # test covariance estimate
       cov_dgp_test <- cov_combination_2x2(data = dgp_data_test, p = p)
       betadgp_covdgp_data_test <- merge(y_dgp_test, cov_dgp_test)
       fwrite(x = betadgp_covdgp_data_test,
              file = file.path(new_folder, "betadgp_covdgp_data_test.csv"),
              row.names = FALSE)

        # test covariance estimate
       corr_dgp_test <- corr_combination_2x2(data = dgp_data_test, p = p)
       betadgp_corrdgp_data_test <- merge(y_dgp_test, corr_dgp_test)
       fwrite(x = betadgp_corrdgp_data_test,
              file = file.path(new_folder, "betadgp_corrdgp_data_test.csv"),
              row.names = FALSE)

       beta2x2_data_test <- lm_combination_2x2(data = dgp_data_test, p = p)
       betadgp_beta2x2_data_test <- merge(y_dgp_test, beta2x2_data_test)
       fwrite(x = betadgp_beta2x2_data_test,
              file = file.path(new_folder, "betadgp_beta2x2_data_test.csv"),
              row.names = FALSE)

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