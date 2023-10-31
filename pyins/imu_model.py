"""Statistical model for IMU."""
from collections import OrderedDict
import numpy as np


class InertialSensor:
    """Inertial sensor triad description.

    Below all parameters might be floats or arrays with 3 elements. In the
    former case the parameter is assumed to be the same for each of 3 sensors.
    Setting a parameter to None means that such error is not presented in
    a sensor.

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
        gyros.
    bias_walk : array_like or None
        Strength of white noise which is integrated into the bias. Known as
        a rate random walk for gyros. Can be set only if `bias` is set.
    scale_misal : array_like, shape (3, 3) or None
        Standard deviations of matrix elements which represent sensor triad
        scale factor errors and misalignment. Non-positive elements will
        disable the corresponding effect estimation.
    """
    MAX_STATES = 12
    MAX_NOISES = 6

    def __init__(self, bias=None, noise=None, bias_walk=None, scale_misal=None):
        bias = self._verify_param(bias, 'bias')
        noise = self._verify_param(noise, 'noise')
        bias_walk = self._verify_param(bias_walk, 'bias_walk')

        if scale_misal is None:
            scale_misal = np.zeros((3, 3))
        scale_misal = np.asarray(scale_misal)
        if scale_misal.shape != (3, 3):
            raise ValueError("`scale_misal` must have shape (3, 3)")

        if bias is None and bias_walk is not None:
            raise ValueError("Set `bias` if you want to use `bias_walk`.")

        F = np.zeros((self.MAX_STATES, self.MAX_STATES))
        G = np.zeros((self.MAX_STATES, self.MAX_NOISES))
        H = np.zeros((3, self.MAX_STATES))
        P = np.zeros((self.MAX_STATES, self.MAX_STATES))
        q = np.zeros(self.MAX_NOISES)
        I = np.identity(3)

        n_states = 0
        n_noises = 0
        states = OrderedDict()
        if bias is not None:
            P[:3, :3] = I * bias ** 2
            H[:, :3] = I
            states['BIAS_1'] = n_states
            states['BIAS_2'] = n_states + 1
            states['BIAS_3'] = n_states + 2
            n_states += 3

        output_axes = []
        input_axes = []
        scale_misal_states = []
        if scale_misal is not None:
            for i in range(3):
                for j in range(3):
                    if scale_misal[i, j] > 0.0:
                        output_axes.append(i)
                        input_axes.append(j)
                        scale_misal_states.append(n_states)
                        P[n_states, n_states] = scale_misal[i, j] ** 2
                        states[f'SCALE_MISAL_{i + 1}{j + 1}'] = n_states
                        n_states += 1

        if bias_walk is not None:
            G[:3, :3] = I
            q[:3] = bias_walk
            n_noises += 3

        F = F[:n_states, :n_states]
        G = G[:n_states, :n_noises]
        H = H[:, :n_states]
        P = P[:n_states, :n_states]
        q = q[:n_noises]

        self.n_states = n_states
        self.n_noises = n_noises
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
        self._H = H

    @staticmethod
    def _verify_param(param, name):
        if param is None:
            return None

        param = np.asarray(param)
        if param.ndim == 0:
            param = np.resize(param, 3)
        if param.shape != (3,):
            raise ValueError("`{}` might be float or array with "
                             "3 elements.".format(name))
        if np.any(param < 0):
            raise ValueError("`{}` must contain non-negative values."
                             .format(name))

        return param

    def output_matrix(self, readings=None):
        if not self.readings_required:
            return self._H

        if readings is None:
            raise ValueError("Inertial `readings` are required when "
                             "`self.scale_misal` is set.")

        readings = np.asarray(readings)
        output_axes, input_axes, states = self.scale_misal_data
        if readings.ndim == 1:
            H = self._H.copy()
            H[output_axes, states] = readings[input_axes]
        else:
            H = np.zeros((len(readings), 3, self.n_states))
            H[:] = self._H
            H[:, output_axes, states] = readings[:, input_axes]
        return H
