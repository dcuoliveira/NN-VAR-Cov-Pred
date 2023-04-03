import os
import argparse

from training.runners import run_new_model_training
from settings import loss_metadata, model_metadata

FILE_PATH = os.path.dirname(__file__)
INPUTS_PATH = os.path.join(FILE_PATH, "data", "inputs")
OUTPUTS_PATH = os.path.join(FILE_PATH, "data", "outputs")
LOG_PATH = os.path.join(FILE_PATH, "data", "log")
DATASET_NAMES = ["betadgp_corrdgp_data", "betadgp_covdgp_data", "betadgp_beta2x2_data", "betadgp_data"]
TARGET_NAME = "betas_dgp"
OUTPUT_OVRD = True

parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs', type=float, default=10, help='Dataset name.')
parser.add_argument('--n_iter', type=int, default=50, help='Model name.')
parser.add_argument('--seed', type=int, default=2294, help='Model name.')
parser.add_argument('--model_tag', type=str, default="mlp", help='Model name.')
parser.add_argument('--standardize', type=bool, default=True, help='Model name.')
parser.add_argument('--train_size', type=float, default=0.7, help='Model name.')
parser.add_argument('--criterion', type=str, default="mse", help='Model name.')
parser.add_argument('--verbose', type=bool, default=False, help='Model name.')

if __name__ == '__main__':
    
    args = parser.parse_args()
    results = run_new_model_training(target_name=TARGET_NAME,
                                     inputs_path=INPUTS_PATH,
                                     outputs_path=OUTPUTS_PATH,
                                     log_path=LOG_PATH,
                                     dataset_names=DATASET_NAMES,
                                     model_tag=args.model_tag,
                                     standardize=args.standardize,
                                     train_size=args.train_size,
                                     wrapper=model_metadata[args.model_tag],
                                     criterion=loss_metadata[args.criterion],
                                     n_jobs=args.n_jobs,
                                     n_iter=args.n_iter,
                                     seed=args.seed,
                                     verbose=args.verbose,
                                     output_ovrd=OUTPUT_OVRD,
                                     dir_name_ovrd=["var_0.05_1_10"],
                                     classification=False)