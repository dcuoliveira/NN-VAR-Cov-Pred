rm(list=ls())
library("here")
library("dplyr")
library("BigVAR")
library("reticulate")

np <- import("numpy")

FILE_PATH <- file.path(here(), "src")
INPUTS_PATH <- file.path(FILE_PATH, "data", "inputs")
OUTPUTS_PATH <- file.path(FILE_PATH, "data", "outputs")
LOG_PATH <- file.path(FILE_PATH, "data", "log")
DATASET_NAMES <- c("betadgp_corrdgp_data", "betadgp_covdgp_data", "betadgp_beta2x2_data", "betadgp_data")
TARGET_NAME <- "betas_dgp"
OUTPUT_OVRD <- TRUE
dir_name_ovrd <- NULL # c("var_0.05_1_10")
max_p <- 6

source(file.path(FILE_PATH, "utils", "Rutils.R"))

## start training procedure ##

if (is.null(dir_name_ovrd)){
  list_dir_names <- list.files(INPUTS_PATH)
}else{
  list_dir_names <- dir_name_ovrd
}

for (dir_name in list_dir_names){
  train_data = read.csv(file.path(INPUTS_PATH, dir_name, "data_dgp_train.csv"))
  test_data = read.csv(file.path(INPUTS_PATH, dir_name, "data_dgp_test.csv"))
  
  # assumption: p is known
  p <- as.numeric(strsplit(dir_name, "_")[[1]][3])
  
  k <- dim(train_data)[2]
  
  # define cv hyperparameters
  cv_model <- constructModel(Y = train_data %>% as.matrix(),
                             p = max_p,
                             struct = "BasicEN",
                             gran = c(50, 10),
                             h = 1,
                             cv = "Rolling",
                             verbose = FALSE,
                             IC = TRUE,
                             model.controls = list(intercept=FALSE))
  
  # run cv 
  cv_model_results <- cv.BigVAR(cv_model)
  
  # fit on train data
  B = BigVAR.fit(Y = train_data %>% as.matrix(),
                 intercept = FALSE,
                 struct = 'BasicEN',
                 p = p,
                 lambda = cv_model_results@OptimalLambda)[,,1]
  
  B <- B[,2:dim(B)[2]] %>% as.data.table()
  colnames(B) <- standardize_var_lag_names(k = k, p = p)
  B$eq <- standardize_var_names(k = k)
  B <- B %>% dplyr::select(eq, everything())
  
  betas_predicted_train <- melt(B, "eq") %>% dplyr::rename(pred=value)
  y_train = read.csv(file.path(INPUTS_PATH, dir_name, "beta_train.csv")) %>% as.data.table() %>% melt("eq")
  results_train <- merge(y_train, betas_predicted_train)
  mse_train <- Metrics::mse(actual = results_train$value, predicted = results_train$pred)
  
  # fit on test data
  B = BigVAR.fit(Y = test_data %>% as.matrix(),
                 intercept = FALSE,
                 struct = 'BasicEN',
                 p = p,
                 lambda = cv_model_results@OptimalLambda)[,,1]
  
  B <- B[,2:dim(B)[2]] %>% as.data.table()
  colnames(B) <- standardize_var_lag_names(k = k, p = p)
  B$eq <- standardize_var_names(k = k)
  B <- B %>% dplyr::select(eq, everything())
  
  betas_predicted <- melt(B, "eq") %>% dplyr::rename(pred=value)
  y_test = read.csv(file.path(INPUTS_PATH, dir_name, "beta_test.csv")) %>% as.data.table() %>% melt("eq")
  results_test <- merge(y_test, betas_predicted) %>% dplyr::rename(actual=value)
  mse_test <- Metrics::mse(actual = results_test$value, predicted = results_test$pred)
  
  # check if dir exists
  if (!dir.exists(file.path(OUTPUTS_PATH, dir_name))){
    dir.create(file.path(OUTPUTS_PATH, dir_name))
  }
  
  # save all results
  np$savez(file.path(OUTPUTS_PATH, dir_name, "var_enet_results.npz"),
           prediction=results_test,
           train_loss=mse_train,
           test_loss=mse_test,
           parameters=list(lambda=cv_model_results@OptimalLambda))
}





