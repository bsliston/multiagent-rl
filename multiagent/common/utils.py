from typing import Dict, List
import numpy as np
import torch


def extend_summary_logs_list(
    summary_logs: Dict[str, List[float]], extension_logs: Dict[str, float]
):
    for key, val in extension_logs.items():
        if key not in summary_logs:
            summary_logs[key] = []
        summary_logs[key].append(val)


def get_average_summary_logs(
    summary_logs: Dict[str, List[float]]
) -> Dict[str, float]:
    return {key: np.average(val) for key, val in summary_logs.items()}


def torch_to_numpy(x):
    try:
        return x.data.numpy()
    except:
        return x.cpu().data.numpy()


def numpy_to_torch_float(x, device="cpu", dtype=torch.float):
    return torch.tensor(x, dtype=dtype).to(device)
