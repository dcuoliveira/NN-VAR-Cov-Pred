import os
import argparse

from training.runners import run_model_training
from models.neural_networks import FFNNFixedClassWrapper

DEBUG = False

parser = argparse.ArgumentParser()
parser.add_argument('n_hidden',
                    type=int,
                    help='number of hidden layers to use in the FFNN architecture')
parser.add_argument('n_neurons',
                    type=int,
                    help='number of neurons per hidden layer to use in the FFNN architecture')
parser.add_argument('activation',
                    type=str,
                    help='type of activation to use in the FFNN architecture')
parser.add_argument('loss_name',
                    type=str,
                    help='name of the loss function to be used in the FFNN training')
parser.add_argument('n_splits',
                    type=int,
                    help='number of cv splits')
parser.add_argument('n_iter',
                    type=int,
                    help='number of samples to be taken from the hyperparameter space')

if DEBUG:
    class Args:
        def __init__(self,
                     n_hidden=1,
                     n_neurons=10,
                     activation="swish",
                     loss_name="binaryx",
                     n_splits=10,
                     n_iter=50):
            self.n_hidden = n_hidden
            self.n_neurons = n_neurons
            self.activation = activation
            self.loss_name = loss_name
            self.n_splits = n_splits
            self.n_iter = n_iter

    args = Args()
else:
    args = parser.parse_args()

# number of jobs to run in parallel
if DEBUG:
    N_JOBS = 1
else:
    N_JOBS = os.cpu_count() - 5
N_SPLITS = args.n_splits  # number of splits (k) to be made within the k fold cv
N_ITER = args.n_iter  # number of parameter settings that are sampled
SEED = 2294
VERBOSE = False
INPUTS_PATH = os.path.join(os.getcwd(), "data", "inputs")
OUTPUTS_PATH = os.path.join(os.getcwd(), "data", "outputs")
LOG_PATH = os.path.join(os.getcwd(), "data", "log")
DATASET_NAMES = ["betadgp_covdgp_data", "betadgp_beta2x2_data", "betadgp_data"]
TARGET_NAME = "betas_dgp"
MODEL_TAG = "ffnn_class" + "_" + str(args.n_hidden) + "_" + str(args.n_neurons) + "_" + str(args.activation) + "_" + str(args.n_splits) + "_" + str(args.n_iter)
STANDARDIZE = True
TRAIN_SIZE = 0.7
OUTPUT_OVRD = True
CLASS = True

if __name__ == '__main__':
    results = run_model_training(inputs_path=INPUTS_PATH,
                                 outputs_path=OUTPUTS_PATH,
                                 log_path=LOG_PATH,
                                 target_name=TARGET_NAME,
                                 dataset_names=DATASET_NAMES,
                                 model_tag=MODEL_TAG,
                                 standardize=STANDARDIZE,
                                 train_size=TRAIN_SIZE,
                                 wrapper=FFNNFixedClassWrapper,
                                 wrapper_ovrd={"n_hidden": args.n_hidden, "n_neurons": args.n_neurons,
                                               "activation": str(args.activation), "loss_name": str(args.loss_name)},
                                 n_jobs=N_JOBS,
                                 n_splits=N_SPLITS,
                                 n_iter=N_ITER,
                                 seed=SEED,
                                 verbose=VERBOSE,
                                 output_ovrd=OUTPUT_OVRD,
                                 classification=CLASS)