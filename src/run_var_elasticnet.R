

N_SPLITS <- 10  # number of splits (k) to be made within the k fold cv
N_ITER <- 50  # number of parameter settings that are sampled
SEED <- 2294
VERBOSE <- TRUE
INPUTS_PATH <- file.path(getwd(), "src", "data", "inputs")
OUTPUTS_PATH <- file.path(getwd(), "data", "outputs")
DATASET_NAMES <- c("betadgp_covdgp_data", "betadgp_beta2x2_data", "betadgp_data")
TARGET_NAME <- "betas_dgp"
MODEL_TAG <- "var_enet"
STANDARDIZE <- TRUE
TRAIN_SIZE <- 0.7
OUTPUT_OVRD <- TRUE

results <- run_model_training(inputs_path=INPUTS_PATH,
                              outputs_path=OUTPUTS_PATH,
                              target_name=TARGET_NAME,
                              dataset_names=DATASET_NAMES,
                              model_tag=MODEL_TAG,
                              standardize=STANDARDIZE,
                              train_size=TRAIN_SIZE,
                              wrapper=FFNNWrapper,
                              n_jobs=N_JOBS,
                              n_splits=N_SPLITS,
                              n_iter=N_ITER,
                              seed=SEED,
                              verbose=VERBOSE,
                              output_ovrd=OUTPUT_OVRD)
