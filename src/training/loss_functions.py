import torch

def WMSELoss(pred, true, w=0.1):
    error = torch.square(true - pred)
    
    weights = (torch.where((true != 0) & (pred == 0), 1, 0) * w) + 1
    weighted_error = error * weights

    return torch.mean(weighted_error)