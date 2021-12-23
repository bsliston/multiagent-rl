import numpy as np
import torch


def torch_to_numpy(x):
    try:
        return x.data.numpy()
    except:
        return x.cpu().data.numpy()


def numpy_to_torch_float(x, device="cpu", dtype=torch.float):
    return torch.tensor(x, dtype=dtype).to(device)
