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
    scale : array_like or None
        Standard deviation of a scale factor, which is modeled as a random
        constant (plus an optional random walk).
    """
    MAX_STATES = 6
    MAX_NOISES = 6

    def __init__(self, bias=None, noise=None, bias_walk=None, scale=None):
        bias = self._verify_param(bias, 'bias')
        noise = self._verify_param(noise, 'noise')
        bias_walk = self._verify_param(bias_walk, 'bias_walk')
        scale = self._verify_param(scale, 'scale')

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
        if scale is not None:
            P[n_states: n_states + 3, n_states: n_states + 3] = I * scale ** 2
            states['SCALE_1'] = n_states
            states['SCALE_2'] = n_states + 1
            states['SCALE_3'] = n_states + 2
            n_states += 3
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
        self.scale = scale
        self.P = P
        self.q = q
        self.F = F
        self.G = G
        self._H = H

    @staticmethod
    def _verify_param(param, name, only_positive=False):
        if param is None:
            return None

        param = np.asarray(param)
        if param.ndim == 0:
            param = np.resize(param, 3)
        if param.shape != (3,):
            raise ValueError("`{}` might be float or array with "
                             "3 elements.".format(name))
        if only_positive and np.any(param <= 0):
            raise ValueError("`{}` must contain positive values.".format(name))
        elif np.any(param < 0):
            raise ValueError("`{}` must contain non-negative values."
                             .format(name))

        return param

    def output_matrix(self, readings=None):
        if self.scale is not None and readings is None:
            raise ValueError("Inertial `readings` are required when "
                             "`self.scale` is set.")

        if self.scale is not None:
            readings = np.asarray(readings)
            if readings.ndim == 1:
                H = self._H.copy()
                i1 = self.states['SCALE_1']
                i2 = self.states['SCALE_3'] + 1
                H[:, i1: i2] = np.diag(readings)
            else:
                n_readings = readings.shape[0]
                H = np.zeros((n_readings, 3, self.n_states))
                H[:] = self._H
                i1 = self.states['SCALE_1']
                i2 = self.states['SCALE_3'] + 1

                I1 = np.repeat(np.arange(n_readings), 3)
                I2 = np.tile(np.arange(3), n_readings)

                H_view = H[:, :, i1: i2]
                H_view[I1, I2, I2] = readings.ravel()
            return H
        else:
            return self._H
