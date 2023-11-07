"""Statistical model for IMU."""
import pandas as pd
import numpy as np
from .util import INDEX_TO_XYZ, XYZ_TO_INDEX


class InertialSensor:
    """Inertial sensor triad description.

    Below all parameters might be floats or arrays. In the former case the parameter
    is assumed to be the same for each of 3 sensors. In the latter case a non-positive
    elements disables the effect for the corresponding axis (or axes in case of
    `scale_misal`). Setting a parameter to None completely disables the effect for all
    axes.

    Note that all parameters are measured in International System of Units.

    Generally this class is not intended for public usage except its
    construction with desired parameters and passing it to a filter class
    constructor.

    Parameters
    ----------
    bias : array_like or None
        Standard deviation of a bias, which is modeled as a random constant
        (plus an optional random walk).
    noise : array_like or None
        Strength of additive white noise. Known as an angle random walk for
        gyros and velocity random walk for accelerometers.
    bias_walk : array_like or None
        Strength of white noise which is integrated into the bias. Known as
        a rate random walk for gyros. Can be set only if `bias` is set.
    scale_misal : array_like, shape (3, 3) or None
        Standard deviations of matrix elements which represent sensor triad
        scale factor errors and misalignment. Non-positive elements will
        disable the corresponding effect estimation.
    """
    MAX_STATES = 12
    MAX_NOISES = 3
    MAX_OUTPUT_NOISES = 3

    def __init__(self, bias=None, noise=None, bias_walk=None, scale_misal=None):
        bias = self._verify_param(bias, (3,), "bias")
        noise = self._verify_param(noise, (3,), "noise")
        bias_walk = self._verify_param(bias_walk, (3,), "bias_walk")
        scale_misal = self._verify_param(scale_misal, (3, 3), "scale_misal")

        if np.any(bias_walk[bias <= 0] > 0):
            raise ValueError(
                "``bias_walk` can be enabled only for axes where `bias` enabled")

        F = np.zeros((self.MAX_STATES, self.MAX_STATES))
        G = np.zeros((self.MAX_STATES, self.MAX_NOISES))
        H = np.zeros((3, self.MAX_STATES))

        P = np.zeros((self.MAX_STATES, self.MAX_STATES))
        q = np.zeros(self.MAX_NOISES)

        J = np.zeros((3, self.MAX_OUTPUT_NOISES))
        v = np.zeros(self.MAX_OUTPUT_NOISES)

        n_states = 0
        n_noises = 0
        states = []
        for axis in range(3):
            if bias[axis] > 0:
                P[n_states, n_states] = bias[axis] ** 2
                H[axis, n_states] = 1
                states.append(f"bias_{INDEX_TO_XYZ[axis]}")

                if bias_walk[axis] > 0:
                    G[n_states, n_noises] = 1
                    q[n_noises] = bias_walk[axis]
                    n_noises += 1

                n_states += 1

        output_axes = []
        input_axes = []
        scale_misal_states = []
        if scale_misal is not None:
            for output_axis in range(3):
                for input_axis in range(3):
                    if scale_misal[output_axis, input_axis] > 0:
                        output_axes.append(output_axis)
                        input_axes.append(input_axis)
                        scale_misal_states.append(n_states)
                        P[n_states, n_states] = scale_misal[
                            output_axis, input_axis] ** 2
                        states.append(f"sm_{INDEX_TO_XYZ[output_axis]}"
                                      f"{INDEX_TO_XYZ[input_axis]}")
                        n_states += 1

        n_output_noises = 0
        for axis in range(3):
            if noise[axis] > 0:
                J[axis, n_output_noises] = 1
                v[n_output_noises] = noise[axis]
                n_output_noises += 1

        F = F[:n_states, :n_states]
        G = G[:n_states, :n_noises]
        H = H[:, :n_states]
        P = P[:n_states, :n_states]
        q = q[:n_noises]

        J = J[:, :n_output_noises]
        v = v[:n_output_noises]

        self.n_states = n_states
        self.n_noises = n_noises
        self.n_output_noises = n_output_noises
        self.states = states
        self.bias = bias
        self.noise = noise
        self.bias_walk = bias_walk
        self.readings_required = bool(output_axes)
        self.scale_misal = scale_misal
        self.scale_misal_data = output_axes, input_axes, scale_misal_states
        self.P = P
        self.q = q
        self.F = F
        self.G = G
        self.H = H
        self.J = J
        self.v = v

        self.transform = np.identity(3)
        self.bias = np.zeros(3)

    @staticmethod
    def _verify_param(param, shape, name):
        if param is None:
            return np.zeros(shape)

        param = np.asarray(param)
        if param.ndim == 0:
            param = np.resize(param, 3)
        if param.shape != shape:
            raise ValueError(f"`{name}` might be float or array with shape {shape}")

        return param

    def output_matrix(self, readings=None):
        if not self.readings_required:
            return self.H

        if readings is None:
            raise ValueError(
                "Inertial `readings` are required when " "`self.scale_misal` is set."
            )

        readings = np.asarray(readings)
        output_axes, input_axes, states = self.scale_misal_data
        if readings.ndim == 1:
            H = self.H.copy()
            H[output_axes, states] = readings[input_axes]
        else:
            H = np.zeros((len(readings), 3, self.n_states))
            H[:] = self.H
            H[:, output_axes, states] = readings[:, input_axes]
        return H

    def reset_estimates(self):
        self.transform = np.identity(3)
        self.bias = np.zeros(3)

    def update_estimates(self, x):
        if len(x) != len(self.states):
            raise ValueError("`x` must have the same length as `states` from the "
                             "constructor")
        for state, xi in zip(self.states, x):
            items = state.split("_")
            if items[0] == 'bias':
                axis = XYZ_TO_INDEX[items[1]]
                self.bias[axis] += xi
            elif items[0] == 'sm':
                axis_out = XYZ_TO_INDEX(items[1][0])
                axis_in = XYZ_TO_INDEX(items[2][1])
                self.transform[axis_out, axis_in] += xi

    def correct_increments(self, dt, increments):
        dt = np.asarray(dt).reshape(-1, 1)
        result = np.linalg.solve(
            self.transform,
            (increments.values - self.bias * dt).transpose()).transpose()
        result = np.ascontiguousarray(result)
        return pd.DataFrame(result, index=increments.index, columns=increments.columns)

    def get_estimates(self):
        states = []
        estimates = []
        for state in self.states:
            items = state.split("_")
            if items[0] == 'bias':
                axis = XYZ_TO_INDEX[items[1]]
                states.append(state)
                estimates.append(self.bias[axis])
            elif items[0] == 'sm':
                axis_out = XYZ_TO_INDEX[items[1][0]]
                axis_in = XYZ_TO_INDEX[items[1][1]]
                states.append(state)
                estimates.append(self.transform[axis_out, axis_in] -
                                 1 if axis_out == axis_in else 0)
        return pd.Series(estimates, index=states)
