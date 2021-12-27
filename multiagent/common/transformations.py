import numpy as np


def normalize_array(x: np.ndarray, low: np.ndarray, high: np.ndarray):
    return (x - low) / (high - low)


def state_flatten(x: np.ndarray) -> np.ndarray:
    return x.flatten()
