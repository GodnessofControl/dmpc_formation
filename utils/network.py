"""
Network simulation: random communication delays and data dropout.
"""

import numpy as np
from typing import Tuple, List


def generate_random_sequence(
    sequence_length: int,
    sigma_L: float,
    sigma_D: float,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random dropout and delay binary sequences.

    Args:
        sequence_length: Length of the sequence (typically N_steps)
        sigma_L: Packet dropout probability
        sigma_D: Communication delay probability
        seed: Random seed for reproducibility

    Returns:
        (isdropout, isdelay): Binary arrays (1 = event occurred)
    """
    if seed is not None:
        np.random.seed(seed)

    # Number of 1s to place
    L_num_ones = int(round(sequence_length * sigma_L))
    D_num_ones = int(round(sequence_length * sigma_D))

    # Generate sequences with random positions
    isdropout = np.zeros(sequence_length, dtype=int)
    isdelay = np.zeros(sequence_length, dtype=int)

    # Random positions for dropout events
    if L_num_ones > 0:
        L_indices = np.random.permutation(sequence_length)[:L_num_ones]
        isdropout[L_indices] = 1

    # Random positions for delay events
    if D_num_ones > 0:
        D_indices = np.random.permutation(sequence_length)[:D_num_ones]
        isdelay[D_indices] = 1

    return isdropout, isdelay


def dropout_delay_to_level(isdropout: int, isdelay: int) -> int:
    """
    Convert dropout/delay flags to combined level for terminal calculation.

    Level meanings:
        0: no dropout, no delay
        1: dropout only (count_L consecutive)
        2: delay only (count_D consecutive)
        3: both dropout and delay
    """
    return isdropout + 2 * isdelay


class NetworkSimulator:
    """
    Simulates network conditions for multi-UAV communication.

    Tracks consecutive dropout/delay counts for each communication link.
    """

    def __init__(self, isdropout: np.ndarray, isdelay: np.ndarray):
        """
        Args:
            isdropout: Binary array of dropout events per step
            isdelay: Binary array of delay events per step
        """
        self.isdropout = isdropout
        self.isdelay = isdelay
        self.n_steps = len(isdropout)

        # Consecutive event counters (per link)
        self.count_L = 0  # consecutive dropout counter
        self.count_D = 0  # consecutive delay counter

    def step(self, k: int) -> Tuple[int, int, int]:
        """
        Update counters and return current network state.

        Args:
            k: Current time step

        Returns:
            (count_L, count_D, level)
        """
        dropout = self.isdropout[k] if k < self.n_steps else 0
        delay = self.isdelay[k] if k < self.n_steps else 0

        if dropout:
            self.count_L += 1
        else:
            self.count_L = 0

        if delay:
            self.count_D += 1
        else:
            self.count_D = 0

        level = dropout_delay_to_level(dropout, delay)

        return self.count_L, self.count_D, level

    def reset_counters(self):
        """Reset consecutive event counters."""
        self.count_L = 0
        self.count_D = 0

    @staticmethod
    def create_random(
        n_steps: int,
        sigma_L: float = 0.15,
        sigma_D: float = 0.2,
        seed: int = None
    ) -> 'NetworkSimulator':
        """Factory: create with random sequences."""
        isdropout, isdelay = generate_random_sequence(n_steps, sigma_L, sigma_D, seed)
        return NetworkSimulator(isdropout, isdelay)
