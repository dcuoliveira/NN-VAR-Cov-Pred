
source(file.path(getwd(), "src", "models", "penalized_var.R"))
source(file.path(getwd(), "src", "training", "Runners.R"))

N_SPLITS <- 10  # number of splits (k) to be made within the k fold cv
N_ITER <- 50  # number of parameter settings that are sampled
SEED <- 2294
INPUTS_PATH <- file.path(getwd(), "src", "data", "inputs")
OUTPUTS_PATH <- file.path(getwd(), "data", "outputs")
DATASET_NAMES <- c("data_dgp")
TARGET_NAME <- "betas_dgp"
MODEL_TAG <- "var_enet"
STANDARDIZE <- TRUE
OUTPUT_OVRD <- TRUE

CV_TYPE <- "Rolling" # rolling cv type
TRAIN_SIZE <- 0.7 # stands for the window.size parameter on the BigVAR function
VERBOSE <- TRUE

results <- run_model_training(inputs_path=INPUTS_PATH,
                              outputs_path=OUTPUTS_PATH,
                              target_name=TARGET_NAME,
                              dataset_names=DATASET_NAMES,
                              model_tag=MODEL_TAG,
                              standardize=STANDARDIZE,
                              train_size=TRAIN_SIZE,
                              wrapper=EnetVARWrapper(),
                              n_jobs=N_JOBS,
                              n_splits=N_SPLITS,
                              n_iter=N_ITER,
                              seed=SEED,
                              verbose=VERBOSE,
                              output_ovrd=OUTPUT_OVRD)
