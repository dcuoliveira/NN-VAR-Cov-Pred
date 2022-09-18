import os

from training.runners import run_model_training
from models.linear import LinearRegWrapper

N_JOBS = None  # number of jobs to run in parallel
N_SPLITS = None  # number of splits (k) to be made within the k fold cv
N_ITER = None  # number of parameter settings that are sampled
SEED = 2294
VERBOSE = True
INPUTS_PATH = os.path.join(os.getcwd(), "data", "inputs")
OUTPUTS_PATH = os.path.join(os.getcwd(), "data", "outputs")
LOG_PATH = os.path.join(os.getcwd(), "data", "log")
DATASET_NAMES = ["betadgp_covdgp_data", "betadgp_beta2x2_data", "betadgp_data"]
TARGET_NAME = "betas_dgp"
MODEL_TAG = "linear_reg"
STANDARDIZE = True
TRAIN_SIZE = 0.7
OUTPUT_OVRD = False

if __name__ == '__main__':
    results = run_model_training(inputs_path=INPUTS_PATH,
                                 outputs_path=OUTPUTS_PATH,
                                 log_path=LOG_PATH,
                                 target_name=TARGET_NAME,
                                 dataset_names=DATASET_NAMES,
                                 model_tag=MODEL_TAG,
                                 standardize=STANDARDIZE,
                                 train_size=TRAIN_SIZE,
                                 wrapper=LinearRegWrapper,
                                 n_jobs=N_JOBS,
                                 n_splits=N_SPLITS,
                                 n_iter=N_ITER,
                                 seed=SEED,
                                 verbose=VERBOSE,
                                 output_ovrd=OUTPUT_OVRD)