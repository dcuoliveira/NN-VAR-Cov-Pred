import os
import argparse

from training.runners import run_model_training
from models.linear import LogisticRegWrapper

INPUTS_PATH = os.path.join(os.getcwd(), "src", "data", "inputs")
OUTPUTS_PATH = os.path.join(os.getcwd(), "src", "data", "outputs")
LOG_PATH = os.path.join(os.getcwd(), "src", "data", "log")
DATASET_NAMES = ["betadgp_covdgp_data", "betadgp_beta2x2_data", "betadgp_data"]
TARGET_NAME = "betas_dgp"
MODEL_TAG = "logit"
STANDARDIZE = True
OUTPUT_OVRD = False
CLASS = True

parser = argparse.ArgumentParser()
parser.add_argument('n_jobs',
                    type=int,
                    default=None,
                    help='number of jobs to run in parallel')
parser.add_argument('n_splits',
                    type=int,
                    default=None,
                    help='number of splits (k) to be made within the k fold cv')
parser.add_argument('n_iter',
                    type=int,
                    help='number of parameter settings that are sampled')
parser.add_argument('seed',
                    type=int,
                    default=2294,
                    help='seed to fix any starting point of random processes')
parser.add_argument('verbose',
                    type=bool,
                    default=True,
                    help='print outputs')
parser.add_argument('n_iter',
                    type=int,
                    help='number of samples to be taken from the hyperparameter space')
parser.add_argument('train_size',
                    type=float,
                    default=0.7,
                    help='number of samples to be taken from the hyperparameter space')

if __name__ == '__main__':
    args = parser.parse_args()
    results = run_model_training(inputs_path=INPUTS_PATH,
                                 outputs_path=OUTPUTS_PATH,
                                 log_path=LOG_PATH,
                                 target_name=TARGET_NAME,
                                 dataset_names=DATASET_NAMES,
                                 model_tag=MODEL_TAG,
                                 standardize=STANDARDIZE,
                                 train_size=args.train_size,
                                 wrapper=LogisticRegWrapper,
                                 wrapper_ovrd=None,
                                 n_jobs=args.njobs,
                                 n_splits=args.n_splits,
                                 n_iter=args.n_iter,
                                 seed=args.seed,
                                 verbose=args.verbose,
                                 output_ovrd=OUTPUT_OVRD,
                                 classification=CLASS)