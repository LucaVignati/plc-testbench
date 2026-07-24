import warnings

import numpy as np
import scipy
from numba import njit


from copy import deepcopy

# TEMP
# TODO REMOVE
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except:
    print("eh ")


@njit
def _apply_prediction_filter(
    past: np.ndarray, coeff: np.ndarray, steps: int
) -> np.ndarray:
    # Initialize output vector
    prediction = np.zeros(steps)

    # Predict future steps one at a time
    for i in range(steps):
        pred = np.dot(past, coeff)

        prediction[i] = pred

        past = np.roll(past, -1)
        past[-1] = pred

    return prediction


class ARModel:
    """AR model of order p."""

    def __init__(self, p: int, diagonal_load: float = 0.0):
        """
        :param p: the order of the AR(p) model
        :param y_true: diagonal loading term
        """
        self.p = p
        self.diagonal_load = diagonal_load

        # Pre-compile Numba decorated function to expedite future calls
        _apply_prediction_filter(past=np.zeros(self.p), coeff=np.ones(self.p), steps=1)

    def _autocorrelation_method(self, valid: np.ndarray) -> np.ndarray:
        """
        Finds the AR(p) model parameters via the autocorrelation method and Levinson-Durbin recursion.
        In doing so, applies a diagonal loading term to the autocorrelation matrix to combat ill-conditioning.
        """
        # Compute the sample autocorrelation function
        acf = scipy.signal.correlate(valid, valid, mode="full", method="auto")

        # In rare cases, method='fft' appears to produce NaN. If so, use method='direct' instead.
        if np.any(np.isnan(acf)):
            acf = scipy.signal.correlate(valid, valid, mode="full", method="direct")

        # Find the zeroth lag index
        zero_lag = len(acf) // 2

        # First column of the autocorrelation matrix
        c = acf[zero_lag : zero_lag + self.p]

        # Diagonal loading to improve conditioning
        c[0] += self.diagonal_load

        # Autocorrelation vector
        b = acf[zero_lag + 1 : zero_lag + self.p + 1]

        # Solve the Toeplitz system of equations using Levinson-Durbin recursion
        ar_coeff = scipy.linalg.solve_toeplitz(c, b, check_finite=False)

        return ar_coeff

    def predict(self, valid: np.ndarray, steps: int) -> np.ndarray:
        """
        Fits the AR model from an array of valid samples before linearly predicting an arbitrary number of steps into
        the future. Uses Numba jit to accelerate sample-by-sample inference.

        As a fail-safe, returns an array of zeros if the output contains NaN or takes values outside of [-1.5, 1.5].
        This allows us to train PARCnet by sampling audio chunks at random without worrying about ill-conditioning.

            :param valid: ndarray of past samples
            :param steps: the number of samples to be predicted
            :return: ndarray of linearly predicted samples
        """
        # Find the AR model parameters
        ar_coeff = self._autocorrelation_method(valid)

        # Apply linear prediction
        pred = _apply_prediction_filter(
            past=np.ascontiguousarray(valid[-self.p :], dtype=np.float32),
            coeff=np.ascontiguousarray(
                ar_coeff[::-1], dtype=np.float32
            ),  # needed for njit
            steps=steps,
        )

        # Raise warnings; helpful in case the AR model becomes numerically unstable.
        if np.any(np.isnan(pred)):
            warnings.warn(f"AR prediction contains NaN", RuntimeWarning)
            return np.zeros_like(pred)

        elif np.any(np.abs(pred) > 1.5):
            warnings.warn(
                f"AR prediction exceeded the safety range: found [{pred.min()}, {pred.max()}]",
                RuntimeWarning,
            )
            return np.zeros_like(pred)

        return pred


"""
PARCnet implementation adapted from https://github.com/polimi-ispl/PARCnet/blob/main/parcnet/model.py
"""


class PARCnet:
    def __init__(
        self,
        model_path: str,
        packet_dim: int,
        extra_pred_dim: int,
        ar_order: int,
        ar_diagonal_load: float,
        ar_context_dim: int,
        nn_context_dim: int,
        nn_fade_dim: int,
        device: str,
    ):

        # Store arguments
        self.packet_dim = packet_dim
        self.extra_dim = extra_pred_dim
        self.device = device

        # Define the prediction length, including the extra length
        self.pred_dim = packet_dim + extra_pred_dim

        # Define the AR and neural network contexts in sample
        self.ar_context_dim = ar_context_dim * packet_dim
        self.nn_context_dim = nn_context_dim * packet_dim

        # Define fade-in modulation vector (neural network contribution only)
        self.nn_fade_dim = nn_fade_dim
        self.nn_fade = np.linspace(0.0, 1.0, nn_fade_dim)

        # Define fade-in and fade-out modulation vectors
        self.fade_in = np.linspace(0.0, 1.0, extra_pred_dim)
        self.fade_out = np.linspace(1.0, 0.0, extra_pred_dim)

        # Instantiate the linear predictor
        self.ar_model = ARModel(ar_order, ar_diagonal_load)

        self.neural_net = torch.jit.load(model_path).to(self.device)

    def __call__(self, context: np.ndarray, is_burst: bool) -> np.ndarray:
        # AR model prediction
        ar_context = np.pad(context, (self.ar_context_dim - len(ar_context), 0))
        ar_pred = self.ar_model.predict(valid=ar_context, steps=self.pred_dim)

        # NN model context
        nn_context = np.pad(
            context, (self.nn_context_dim - len(nn_context), self.pred_dim)
        )
        nn_context = torch.Tensor(nn_context[None, None, ...]).to(self.device)

        # NN model inference
        with torch.no_grad():
            nn_pred = self.neural_net(nn_context)
            nn_pred = nn_pred[..., -self.pred_dim :]
            nn_pred = nn_pred.squeeze().cpu().numpy()

        # Apply fade-in to the neural network contribution (inbound fade-in)
        nn_pred[: self.nn_fade_dim] *= self.nn_fade

        # Combine the two predictions
        prediction = ar_pred + nn_pred

        # Cross-fade the compound prediction (outbound fade-out)
        prediction[-self.extra_dim :] *= self.fade_out

        if is_burst:
            # Cross-fade the prediction in case of consecutive packet losses (inbound fade-in)
            prediction[: self.extra_dim] *= self.fade_in

        # Split prediction and extra pred
        extra_pred = prediction[-self.extra_dim :]
        prediction = prediction[: self.packet_dim]

        return prediction, extra_pred
