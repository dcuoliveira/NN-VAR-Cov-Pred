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

       mts <- sim.data(model = "ar2",
                      time = 1,
                      n.obs = N,
                      n.var = 2,
                      seed = SEED,
                      prob0 = 0.9,
                      network = network)

       # simulated time series
       dgp_data <- mts$data1 %>% longitudinal_to_data.table()
       fwrite(x = beta,
              file = file.path(new_folder, "data_dgp.csv"),
              row.names = FALSE)

       # lagged dgp
       lag_dgp_data <- dgp_data
       for (p in 1:as.integer(str_split(model, "ar")[[1]][2])){
              for (col in colnames(dgp_data)){
                     col_number <- as.integer(str_split(colnames(lag_dgp_data)[dim(lag_dgp_data)[2]], "V")[[1]][2]) # nolint
                     lag_dgp_data[[paste0("V", col_number + 1)]] <- lag(lag_dgp_data[[paste0("V", col_number)]], p) # nolint
              }
       }

       # dgp covariance matrix ?
       sigma <- mts$sigma %>% as.data.table()
       colnames(sigma) <- gsub("V", "", colnames(sigma))
       fwrite(x = sigma,
              file = file.path(new_folder, "sigma.csv"),
              row.names = FALSE)

       # dgp betas
       beta <- mts$gamma %>% t() %>% as.data.table()
       colnames(beta) <- gsub("V", "", colnames(beta))
       fwrite(x = beta,
              file = file.path(new_folder, "beta.csv"),
              row.names = FALSE)

       # sample covariance matrix
       betas_dgp <- beta %>%
        mutate(eq = row.names(beta)) %>%
         dplyr::select(eq, everything())
       y_dgp <- melt(betas_dgp, id = c("eq")) %>%
        as.data.table() %>%
         rename(betas_dgp = value)
       cov_dgp <- cov(lag_dgp_data %>% tidyr::drop_na()) %>%
        melt() %>%
          as.data.table() %>%
           rename(eq = Var1, variable = Var2, cov_dgp = value)
       cov_dgp$eq <- gsub("V", "", cov_dgp$eq)
       cov_dgp$variable <- gsub("V", "", cov_dgp$variable)
       betadgp_covdgp_data <- merge(y_dgp, cov_dgp)
       fwrite(x = betadgp_covdgp_data,
              file = file.path(new_folder, "betadgp_covdgp_data.csv"),
              row.names = FALSE)

       # sample correlation matrix
       corr_dgp <- cor(lag_dgp_data %>% tidyr::drop_na()) %>%
        melt() %>%
          as.data.table() %>%
           rename(eq = Var1, variable = Var2, cov_dgp = value)
       corr_dgp$eq <- gsub("V", "", corr_dgp$eq)
       corr_dgp$variable <- gsub("V", "", corr_dgp$variable)
       betadgp_corrdgp_data <- merge(y_dgp, corr_dgp)
       fwrite(x = betadgp_corrdgp_data,
              file = file.path(new_folder, "betadgp_corrdgp_data.csv"),
              row.names = FALSE)

       # beta 2x2 of each time series and lags
       beta2x2_data <- lm_combination_2x2(data = dgp_data, p = 2) %>% # nolint
        rename(eq = Var1, variable = Var2) %>%
         mutate(eq = as.character(eq), variable = as.character(variable))
       betadgp_beta2x2_data <- merge(y_dgp, beta2x2_data)
       fwrite(x = betadgp_covdgp_data,
              file = file.path(new_folder, "betadgp_beta2x2_data.csv"),
              row.names = FALSE)

       # all covariates
       betadgp_data <- merge(betadgp_covdgp_data,
                             betadgp_corrdgp_data,
                             beta2x2_data,
                             by = c("eq", "variable"))
       fwrite(x = betadgp_data,
              file = file.path(new_folder, "betadgp_data.csv"),
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
        rename(betas_dgp = value)
       cov_dgp_test <- melt(mts_test$sigma) %>%
        as.data.table() %>%
         rename(cov_dgp = value)
       betadgp_covdgp_data_test <- merge(y_dgp_test, cov_dgp_test) %>%
        filter((Var1 != Var2))
       fwrite(x = betadgp_covdgp_data_test,
              file = file.path(new_folder, "betadgp_covdgp_data_test.csv"),
              row.names = FALSE)
       simulation_dgp_test <- mts_test$data1 %>% longitudinal_to_data.table()
       beta2x2_data_test <- lm_combination_2x2(data = simulation_dgp_test)
       betadgp_beta2x2_data_test <- merge(y_dgp_test, beta2x2_data_test) %>%
        filter((Var1 != Var2))
       fwrite(x = betadgp_beta2x2_data_test,
              file = file.path(new_folder, "betadgp_beta2x2_data_test.csv"),
              row.names = FALSE)
       betadgp_data_test <- merge(betadgp_covdgp_data_test,
                                  betadgp_beta2x2_data_test %>%
                                   select(Var1, Var2, beta_2x2)) %>%
                                    filter((Var1 != Var2))
       fwrite(x = betadgp_data_test,
              file = file.path(new_folder, "betadgp_data_test.csv"),
              row.names = FALSE)
       test_dgp_data <- mts_test$data1 %>% longitudinal_to_data.table()
       fwrite(x = test_dgp_data,
              file = file.path(new_folder, "data_dgp_test.csv"),
              row.names = FALSE)
    }
  }
}

stopCluster(cl)