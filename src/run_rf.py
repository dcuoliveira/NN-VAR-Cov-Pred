import os

from src.model_trainning_functions import run_train_test_setup
from src.models import RandomForestWrapper

N_JOBS = 2  # number of jobs to run in parallel
N_SPLITS = 10  # number of splits (k) to be made within the k fold cv
N_ITER = 10  # number of parameter settings that are sampled
SEED = 2294
VERBOSE = True
INPUTS_PATH = os.path.join(os.getcwd(), "data", "inputs")
OUTPUTS_PATH = os.path.join(os.getcwd(), "data", "outputs")
DATASET_NAMES = ["betadgp_covdgp_data", "betadgp_beta2x2_data", "betadgp_data"]
TARGET_NAME = "betas_dgp"
MODEL_TAG = "RandomForestWrapper"

if __name__ == '__main__':
    results = run_train_test_setup(inputs_path=INPUTS_PATH,
                                   outputs_path=OUTPUTS_PATH,
                                   target_name=TARGET_NAME,
                                   dataset_names=DATASET_NAMES,
                                   model_tag=MODEL_TAG,
                                   wrapper=RandomForestWrapper,
                                   n_jobs=N_JOBS,
                                   n_splits=N_SPLITS,
                                   n_iter=N_ITER,
                                   seed=SEED,
                                   verbose=VERBOSE)