"""
RhoController.py
================
Implements the Rho Algorithm (Predictive Convolutional Feedback)
for underactuated inverted pendulum systems.

Architecture:
  1. Convolutional History Buffer : Exponential FIR filtering over N steps.
  2. Predictive Horizon (H >= 2)  : Evaluates error slope and curvature.
  3. Decoupled Multi-Channel PID  : Independent gains tailored to link dynamics.
  4. Actuator Output Synthesis   : Projects channel corrections to scalar actuator force.
"""

import numpy as np
from collections import deque


class RhoController:
    """
    Predictive Convolutional Controller (Rho Algorithm).

    Parameters
    ----------
    kp_rho : float or array-like
        Proportional gains for each degree of freedom.
    ki_rho : float or array-like
        Integral gains for steady-state error elimination.
    kd_rho : float or array-like
        Derivative / rate damping gains.
    N : int, default=15
        History buffer depth (number of past timesteps stored).
    alpha : float, default=0.5
        Exponential decay parameter for the FIR filter kernel.
    H : int, default=2
        Prediction horizon order (1 = first difference, 2 = with curvature).
    channel_weights : array-like, optional
        Weight vector projecting multichannel errors into scalar actuator force.
    dt : float, default=0.02
        Integration / control sample timestep in seconds.
    """

    def __init__(
        self,
        kp_rho,
        ki_rho,
        kd_rho,
        N=15,
        alpha=0.5,
        H=2,
        channel_weights=None,
        dt=0.02,
    ):
        if channel_weights is None:
            channel_weights = [0.0, -1.0, 1.0]
        self.weights = np.array(channel_weights, dtype=float)
        self.n_ch = len(self.weights)

        # Broadcast gains to match the number of channels
        self.kp = np.broadcast_to(np.atleast_1d(kp_rho), (self.n_ch,)).copy().astype(float)
        self.ki = np.broadcast_to(np.atleast_1d(ki_rho), (self.n_ch,)).copy().astype(float)
        self.kd = np.broadcast_to(np.atleast_1d(kd_rho), (self.n_ch,)).copy().astype(float)

        self.N = N
        self.alpha = alpha
        self.H = H
        self.dt = dt

        # Precompute normalized exponential FIR kernel
        j = np.arange(N, dtype=float)
        k = np.exp(-alpha * j)
        self.kernel = k / k.sum()

        # Sliding history buffer
        self.history = deque(maxlen=N)

        # State memory
        self.integral_rho = np.zeros(self.n_ch)
        self._e_conv_prev = np.zeros(self.n_ch)
        self._e_conv_prev2 = np.zeros(self.n_ch)
        self._step_count = 0

    def update(self, e: np.ndarray, qdot: np.ndarray = None) -> float:
        """
        Compute the scalar Rho control output from current error and velocities.

        Parameters
        ----------
        e : np.ndarray, shape (n_ch,)
            System error vector: (q_ref - q).
        qdot : np.ndarray, shape (n_ch,), optional
            Generalized velocity vector. When provided, enables exact zero-lag
            damping feedback: d(e)/dt = -qdot.

        Returns
        -------
        float
            Scalar actuator output to be scaled by rho and added to the primary driver.
        """
        e = np.asarray(e, dtype=float)
        qdot = np.asarray(qdot, dtype=float) if qdot is not None else None

        # 1. Update history buffer
        self.history.appendleft(e.copy())

        # 2. Convolve history with FIR kernel
        e_conv = self._convolve()

        # 3. Multichannel predictive PID
        u_rho_vec = self._rho_pid(e_conv, qdot)

        # 4. Project channels to scalar actuator force
        u_rho_scalar = float(np.dot(self.weights, u_rho_vec))

        # 5. Advance internal step registers
        self._e_conv_prev2 = self._e_conv_prev.copy()
        self._e_conv_prev = e_conv.copy()
        self._step_count += 1

        return u_rho_scalar

    def reset(self):
        """Reset internal state registers and history buffer."""
        self.history.clear()
        self.integral_rho = np.zeros(self.n_ch)
        self._e_conv_prev = np.zeros(self.n_ch)
        self._e_conv_prev2 = np.zeros(self.n_ch)
        self._step_count = 0

    def _convolve(self) -> np.ndarray:
        """Apply normalized exponential FIR kernel to history buffer."""
        buf = list(self.history)
        n = len(buf)

        if n == 0:
            return np.zeros(self.n_ch)

        # Normalize kernel over available history during initial fill
        k_slice = self.kernel[:n]
        k_norm = k_slice / k_slice.sum()

        stack = np.stack(buf, axis=0)
        return k_norm @ stack

    def _rho_pid(self, e_conv: np.ndarray, qdot: np.ndarray = None) -> np.ndarray:
        """Evaluate multichannel predictive PID on convolved errors."""
        dt = self.dt

        # Proportional term
        P = self.kp * e_conv

        # Integral term
        self.integral_rho += e_conv * dt
        I = self.ki * self.integral_rho

        # Derivative term with predictive curvature
        if qdot is not None:
            e_dot_pred = -qdot
        elif self._step_count == 0:
            e_dot_pred = np.zeros(self.n_ch)
        elif self._step_count == 1 or self.H == 0:
            e_dot_pred = (e_conv - self._e_conv_prev) / dt
        else:
            first_order = (e_conv - self._e_conv_prev) / dt
            curvature = (e_conv - 2 * self._e_conv_prev + self._e_conv_prev2) / dt
            e_dot_pred = first_order + float(self.H) * curvature

        D = self.kd * e_dot_pred

        return P + I + D
