import numpy as np


def compute_expected_return(
    rewards: np.ndarray, gamma: float, normalize: bool = True
):
    n_rewards = rewards.size
    expected_returns = np.zeros_like(rewards)
    expected_return = 0.0

    # Expected return works backwards from last seen reward to get Gt value
    # expectation
    for ri, reward in enumerate(rewards[::-1]):
        expected_return = reward + (expected_return * gamma)
        expected_returns[n_rewards - ri - 1] = expected_return

    # Normalize expected returns per episode for stability during training usage
    # (e.g. advantage objective training).
    if normalize:
        expected_returns = (expected_returns - np.average(expected_returns)) / (
            np.std(expected_returns) + 1e-8
        )

    return expected_returns
