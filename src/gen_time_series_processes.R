rm(list=ls())
gc()
library('SparseTSCGM')
library('dplyr')

MODELS = c("ar1","ar2")
N = 100
K = c(10)
J = 1
SEED = 02021994
PROB_OF_CONNECTION = 0.1
NETWROKS = c("random", "scale-free", "hub")

for (model in MODELS){
  for (k in K){
    for (network in NETWROKS){
      mts = sim.data(model=model,
                     time=N,
                     n.obs=J, 
                     n.var=k,
                     seed=SEED,
                     prob0=PROB_OF_CONNECTION,
                     network=network)
    }
  }
}

