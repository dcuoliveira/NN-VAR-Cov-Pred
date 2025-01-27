import os
import torch

from training.runners import run_new_model_training
from models.neural_networks import MLPWrapper
from training.loss_functions import WMSELoss

N_JOBS = 2  # number of jobs to run in parallel
N_SPLITS = 5  # number of splits (k) to be made within the k fold cv
N_ITER = 500  # number of parameter settings that are sampled
SEED = 2294
VERBOSE = False
INPUTS_PATH = os.path.join(os.getcwd(), "src", "data", "inputs")
OUTPUTS_PATH = os.path.join(os.getcwd(), "src", "data", "outputs")
LOG_PATH = os.path.join(os.getcwd(), "src", "data", "log")
DATASET_NAMES = ["betadgp_covdgp_data", "betadgp_corrdgp_data", "betadgp_beta2x2_data"]
TARGET_NAME = "betas_dgp"
MODEL_TAG = "mlp"
STANDARDIZE = True
TRAIN_SIZE = 0.7
OUTPUT_OVRD = True
CRITERION = torch.nn.MSELoss()

if __name__ == '__main__':
    results = run_new_model_training(inputs_path=INPUTS_PATH,
                                     outputs_path=OUTPUTS_PATH,
                                     log_path=LOG_PATH,
                                     target_name=TARGET_NAME,
                                     dataset_names=DATASET_NAMES,
                                     model_tag=MODEL_TAG,
                                     standardize=STANDARDIZE,
                                     train_size=TRAIN_SIZE,
                                     wrapper=MLPWrapper,
                                     criterion=CRITERION,
                                     n_jobs=N_JOBS,
                                     n_iter=N_ITER,
                                     dir_name_ovrd=["var_0.05_1_150"],
                                     seed=SEED,
                                     verbose=VERBOSE,
                                     output_ovrd=OUTPUT_OVRD)