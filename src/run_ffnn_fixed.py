import os
import argparse

from training.runners import run_model_training
from models.neural_networks import FFNNFixedWrapper

DEBUG = False

parser = argparse.ArgumentParser()
parser.add_argument('n_hidden',
                    type=int,
                    help='number of hidden layers to use in the FFNN architecture')
parser.add_argument('n_neurons',
                    type=int,
                    help='number of neurons per hidden layer to use in the FFNN architecture')

if DEBUG:
    class Args:
        def __init__(self,
                     n_hidden=1,
                     n_neurons=10,
                     activation="relu"):
            self.n_hidden = n_hidden
            self.n_neurons = n_neurons
            self.activation = activation
    args = Args()
else:
    args = parser.parse_args()

N_JOBS = -1  # number of jobs to run in parallel
N_SPLITS = 5  # number of splits (k) to be made within the k fold cv
N_ITER = 20  # number of parameter settings that are sampled
SEED = 2294
VERBOSE = False
INPUTS_PATH = os.path.join(os.getcwd(), "data", "inputs")
OUTPUTS_PATH = os.path.join(os.getcwd(), "data", "outputs")
LOG_PATH = os.path.join(os.getcwd(), "data", "log")
DATASET_NAMES = ["betadgp_covdgp_data", "betadgp_beta2x2_data", "betadgp_data"]
TARGET_NAME = "betas_dgp"
MODEL_TAG = "ffnn" + "_" + str(args.n_hidden) + "_" + str(args.n_neurons) + "_" + str(args.activation)
STANDARDIZE = True
TRAIN_SIZE = 0.7
OUTPUT_OVRD = True

if __name__ == '__main__':
    results = run_model_training(inputs_path=INPUTS_PATH,
                                 outputs_path=OUTPUTS_PATH,
                                 log_path=LOG_PATH,
                                 target_name=TARGET_NAME,
                                 dataset_names=DATASET_NAMES,
                                 model_tag=MODEL_TAG,
                                 standardize=STANDARDIZE,
                                 train_size=TRAIN_SIZE,
                                 wrapper=FFNNFixedWrapper,
                                 wrapper_ovrd={"n_hidden": args.n_hidden, "n_neurons": args.n_neurons, "activation": str(args.activation)},
                                 n_jobs=N_JOBS,
                                 n_splits=N_SPLITS,
                                 n_iter=N_ITER,
                                 seed=SEED,
                                 verbose=VERBOSE,
                                 output_ovrd=OUTPUT_OVRD)