"""
LMI-based terminal feedback gain computation.
Mirrors MATLAB LMI_OnlineRandom.m and LMI_randomdelay.m

Computes K matrix for terminal feedback control under
random communication delays and data dropout.

Note: This uses a simplified gain computation. For full LMI
solution, integrate with scipy or a dedicated LMI solver.
"""

import numpy as np
from typing import Tuple, Optional


# Default gains (from MATLAB when LMI fails to converge)
DEFAULT_K1 = np.array([
    [-0.0067, 0, 0, 0, 0],
    [0, 0, 0, -0.0202, 0],
    [0, 0, 0, 0, -0.0202],
])

DEFAULT_K2 = np.array([
    [-0.0067, 0, 0, 0, 0],
    [0, 0, 0, -0.0202, 0],
    [0, 0, 0, 0, -0.0202],
])

DEFAULT_K3 = np.array([
    [-0.0067, 0, 0, 0, 0],
    [0, 0, 0, -0.0202, 0],
    [0, 0, 0, 0, -0.0202],
])

DEFAULT_K4 = np.array([
    [-0.0067, 0, 0, 0, 0],
    [0, 0, 0, -0.0202, 0],
    [0, 0, 0, 0, -0.0202],
])


class LMISolver:
    """
    Simplified LMI solver for terminal feedback gain computation.

    In the original MATLAB code, this uses the Robust Control Toolbox's
    lmivar/lmiterm/getlmis/feasp functions. This Python version provides
    a simplified implementation using predefined gain matrices.

    For full LMI support, consider integrating with:
    - YALMIP + MPATENTS
    - CVXPY with SDP support
    - sklearn-gmm (for specific problem structures)
    """

    def __init__(self):
        self.sigma_d = 0.125   # dropout probability
        self.sigma_L = 0.15625  # delay probability
        self.dmax = 3           # maximum delay steps
        self.Dk = 1.0 / 3.0     # dmax inverse
        self.Tau_d = 3.0
        self.Tk = 1.0 / 3.0
        self.DLmax = -(self.Dk + self.Tk)

    def compute_A_matrix(
        self, omega: float, v: float, phi: float
    ) -> np.ndarray:
        """
        Compute system matrix A(omega, v, phi) for linearized dynamics.

        A = [1, omega*cos(phi), -zeta, 0, 0;
             0, -omega+1, 0, 0, 0;
             zeta, omega*sin(phi), 1, 0, 0;
             0, 0, 0, 2, 0;
             0, 0, 0, 0, 2]
        """
        zeta = 0.0  # pitch rate (approximated as 0 in linearized model)

        A = np.array([
            [1, omega * np.cos(phi), -zeta, 0, 0],
            [0, -omega + 1, 0, 0, 0],
            [zeta, omega * np.sin(phi), 1, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 2],
        ])
        return A

    def compute_B_matrix(self) -> np.ndarray:
        """Compute input matrix B."""
        return np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

    def compute_delay_dropout_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute delay/dropout probability matrices.

        DLno = (1-sigma_d) * (1-sigma_L)
        DnoL = (1-sigma_d) * sigma_L
        LnoD = sigma_d * (1-sigma_L)
        DL = sigma_d * sigma_L
        """
        B_sigma_d = 1 - self.sigma_d
        B_sigma_L = 1 - self.sigma_L

        DLno = B_sigma_d * B_sigma_L
        DnoL = B_sigma_d * self.sigma_L
        LnoD = self.sigma_L * B_sigma_d
        DL = self.sigma_d * self.sigma_L

        return DLno, DnoL, LnoD, DL

    def compute_gain(
        self,
        v: float,
        omega: float,
        zeta: float,
        phi: float,
        k: int,
        use_lmi: bool = False
    ) -> np.ndarray:
        """
        Compute terminal feedback gain K.

        Args:
            v, omega, zeta: Current control inputs
            phi: Current pitch angle
            k: Time step (for potential scheduling)
            use_lmi: If True, attempt full LMI computation (requires custom SDP solver)

        Returns:
            K: (3, 5) feedback gain matrix
        """
        if not use_lmi:
            # Return default gains (from successful MATLAB runs)
            # In practice, these would be scheduled based on operating point
            return DEFAULT_K1.copy()

        # Full LMI computation would go here
        # This requires an SDP solver integration
        raise NotImplementedError(
            "Full LMI solver not implemented. "
            "Set use_lmi=False to use default gains, "
            "or integrate with CVXPY/scipy for SDP solving."
        )


def terminal_feedback_control(
    state_error: np.ndarray,
    K: np.ndarray
) -> Tuple[float, float, float]:
    """
    Apply terminal feedback control: u = vr + K * error

    Args:
        state_error: [ex, ey, eh, etheta, ephi] - state error
        K: (3, 5) feedback gain matrix

    Returns:
        vk, wk, zetak: Adjusted control inputs
    """
    Kx = K @ state_error
    vk = Kx[0]
    wk = Kx[1]
    zetak = Kx[2]
    return vk, wk, zetak
