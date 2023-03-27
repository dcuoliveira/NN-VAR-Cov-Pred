import torch

from models.neural_networks import MLPWrapper
from models.linear import LinearRegWrapper

loss_metadata = {"mse": torch.nn.MSELoss()}

model_metadata = {"mlp": MLPWrapper,
                  "linear_reg": LinearRegWrapper}
