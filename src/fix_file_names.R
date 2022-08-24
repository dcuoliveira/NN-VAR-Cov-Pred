rm(list=ls())
library("data.table")
library("dplyr")

FILE_PATH = getwd()

source(file.path(FILE_PATH, "src", "utils", "Rutils.R"))

OUTPUT_PATH = file.path(FILE_PATH, "src", "data", "inputs")
MODELS = c("ar1")
N = 100
K_INIT = 150
K = 500
J = 1
SEED = 02021994
PROB_OF_CONNECTION = 0.1
NETWORKS = c("random")

for (model in MODELS){
  for (k in K_INIT:K){
    for (network in NETWORKS){
      output_name = paste0(model, "_", k, "_", network)
      new_folder = file.path(OUTPUT_PATH, output_name)
      data <- fread(file = file.path(new_folder, "test_data_dgp.csv"))
      
      fwrite(x = data,
             file = file.path(new_folder, "data_dgp_test.csv"),
             row.names = FALSE)
      file.remove(file.path(new_folder, "test_data_dgp.csv"))
    }
  }
}
