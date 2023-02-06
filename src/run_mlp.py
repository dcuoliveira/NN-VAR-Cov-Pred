import os
import torch

from training.runners import run_new_model_training
from models.neural_networks import MLPWrapper

N_JOBS = 2
N_ITER = 50
SEED = 2294
VERBOSE = False
INPUTS_PATH = os.path.join(os.getcwd(), "src", "data", "inputs")
OUTPUTS_PATH = os.path.join(os.getcwd(), "src", "data", "outputs")
LOG_PATH = os.path.join(os.getcwd(), "src", "data", "log")
DATASET_NAMES = ["betadgp_corrdgp_data", "betadgp_covdgp_data", "betadgp_beta2x2_data"]
TARGET_NAME = "betas_dgp"
MODEL_TAG = "mlp"
STANDARDIZE = True
TRAIN_SIZE = 0.7
OUTPUT_OVRD = True
criterion = torch.nn.MSELoss()

if __name__ == '__main__':
    
    results = run_new_model_training(target_name=TARGET_NAME,
                                     inputs_path=INPUTS_PATH,
                                     outputs_path=OUTPUTS_PATH,
                                     log_path=LOG_PATH,
                                     dataset_names=DATASET_NAMES,
                                     model_tag=MODEL_TAG,
                                     standardize=STANDARDIZE,
                                     train_size=TRAIN_SIZE,
                                     wrapper=MLPWrapper,
                                     criterion=criterion,
                                     n_jobs=N_JOBS,
                                     n_iter=N_ITER,
                                     seed=SEED,
                                     verbose=VERBOSE,
                                     output_ovrd=OUTPUT_OVRD,
                                     dir_name_ovrd=["ar1_150_random", "ar1_200_random", "ar1_250_random", "ar1_300_random"],
                                     classification=False)