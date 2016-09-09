"""Navigation Kalman filters."""
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, cho_solve, solve_triangular
from . import dcm, earth, util


N_BASE_STATES = 7
DR1 = 0
DR2 = 1
DV1 = 2
DV2 = 3
PHI1 = 4
PHI2 = 5
PSI3 = 6

DRE = 0
DRN = 1
DVE = 2
DVN = 3
DH = 4
DP = 5
DR = 6


class FiltResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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
    scale_walk : array_like or None
        Strength of white noise which is integrated into the scale factor.
        Can be set only if `scale` is set.
    corr_sd, corr_time : array_like or None
        Steady state standard deviation and correlation time for exponentially
        correlated noise. You need to set both or none of these values.
    """
    MAX_STATES = 9
    MAX_NOISES = 9

    def __init__(self, bias=None, noise=None, bias_walk=None,
                 scale=None, scale_walk=None, corr_sd=None, corr_time=None):
        bias = self._verify_param(bias, 'bias')
        noise = self._verify_param(noise, 'noise')
        bias_walk = self._verify_param(bias_walk, 'bias_walk')
        scale = self._verify_param(scale, 'scale')
        scale_walk = self._verify_param(scale_walk, 'scale_walk')
        corr_time = self._verify_param(corr_time, 'corr_time', True)
        corr_sd = self._verify_param(corr_sd, 'corr_sd')

        if (corr_sd is None) + (corr_time is None) == 1:
            raise ValueError("Set both `corr_sd` and `corr_time`.")

        if bias is None and bias_walk is not None:
            raise ValueError("Set `bias` if you want to use `bias_walk`.")

        if scale is None and scale_walk is not None:
            raise ValueError("Set `scale` if you want to use `scale_walk`.")

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
        if scale is not None or scale_walk is not None:
            P[n_states: n_states + 3, n_states: n_states + 3] = I * scale ** 2
            states['SCALE_1'] = n_states
            states['SCALE_2'] = n_states + 1
            states['SCALE_3'] = n_states + 2
            n_states += 3
        if bias_walk is not None:
            G[:3, :3] = I
            q[:3] = bias_walk
            n_noises += 3
        if scale_walk is not None:
            G[n_noises: n_noises + 3, n_noises: n_noises + 3] = I
            q[n_noises: n_noises + 3] = scale_walk
            n_noises += 3
        if corr_sd is not None:
            F[n_states:n_states + 3, n_states:n_states + 3] = -I / corr_time
            G[n_noises:n_noises + 3, n_noises:n_noises + 3] = I
            H[:, n_states: n_states + 3] = I
            P[n_states:n_states + 3, n_states:n_states + 3] = I * corr_sd ** 2
            q[n_noises:n_noises + 3] = (2 / corr_time) ** 0.5 * corr_sd
            states['CORR_1'] = n_states
            states['CORR_2'] = n_states + 1
            states['CORR_3'] = n_states + 2
            n_states += 3
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
        self.scale_walk = scale_walk
        self.corr_sd = corr_sd
        self.corr_time = corr_time
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


class Observation:
    """Base class for observation models.

    Documentation is given to explain how you can implement a new observation
    model. All you need to do is to implement `compute_obs` function. See Also
    section contains links to already implemented models.

    Other methods are generally of no interest for a user, they are called by
    other classes.

    Parameters
    ----------
    data : DataFrame
        Observed values as a DataFrame. Index must contain time stamps.
    gain_curve : None, callable or 3-tuple
        Kalman correction gain curve. It determines the proportionality of
        a state correction and a normalized measurement residual (by its
        theoretical covariance). In the standard Kalman correction it is an
        identity function. To make the filter robust to outliers a sublinear
        function can be provided. A convenient parametrization of such function
        is supported. It described by 3 numbers [L, F, C], if q is a normalized
        residual then:

            * If q < L: standard Kalman correction is used.
            * If L <= q < F: correction is kept constant on a level of q = L.
            * If F <= q < C: correction decays to 0 as ~1/q.
            * IF q >= C: the measurement is rejected completely.

        If None (default), the standard Kalman correction will be used.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.

    See Also
    --------
    LatLonObs
    VeVnObs
    """
    def __init__(self, data, gain_curve=None):
        if callable(gain_curve):
            self.gain_curve = gain_curve
        elif gain_curve is not None:
            self.gain_curve = self._create_gain_curve(gain_curve)
        else:
            self.gain_curve = None

        self.data = data

    @staticmethod
    def _create_gain_curve(params):
        L, F, C = params

        def gain_curve(q):
            if q > C:
                return 0
            if F < q <= C:
                return L * F * (C - q) / ((C - F) * q)
            elif L < q <= F:
                return L
            else:
                return q

        return gain_curve

    def compute_obs(self, stamp, traj_point):
        """Compute ingredients for a single linearized observation.

        It must compute the observation model (z, H, R) at a given time stamp.
        If the observation is not available at a given `stamp`, it must return
        None.

        Parameters
        ----------
        stamp : int
            Time stamp.
        traj_point : Series
            Point of INS trajectory at `stamp`.

        Returns
        -------
        z : ndarray, shape (n_obs,)
            Observation vector. A difference between an INS corresponding
            value and an observed value.
        H : ndarray, shape (n_obs, 7)
            Observation model matrix. It relates the vector `z` to the
            INS error states.
        R : ndarray, shape (n_obs, n_obs)
            Covariance matrix of the observation error.
        """
        raise NotImplementedError()


class LatLonObs(Observation):
    """Observation of latitude and longitude (from GPS or any other source).

    Parameters
    ----------
    data : DataFrame
        Must contain columns 'lat' and 'lon' for latitude and longitude.
        Index must contain time stamps.
    sd : float
        Measurement accuracy in meters.
    gain_curve : None, callable or 3-tuple
        Kalman correction gain curve. It determines the proportionality of
        a state correction and a normalized measurement residual (by its
        theoretical covariance). In the standard Kalman correction it is an
        identity function. To make the filter robust to outliers a sublinear
        function can be provided. A convenient parametrization of such function
        is supported. It described by 3 numbers [L, F, C], if q is a normalized
        residual then:

            * If q < L: standard Kalman correction is used.
            * If L <= q < F: correction is kept constant on a level of q = L.
            * If F <= q < C: correction decays to 0 as ~1/q.
            * IF q >= C: the measurement is rejected completely.

        If None (default), the standard Kalman correction will be used.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.
    """
    def __init__(self, data, sd, gain_curve=None):
        super(LatLonObs, self).__init__(data, gain_curve)
        self.R = np.diag([sd, sd]) ** 2
        H = np.zeros((2, N_BASE_STATES))
        H[0, DR1] = 1
        H[1, DR2] = 1
        self.H = H

    def compute_obs(self, stamp, traj_point):
        """Compute ingredients for a single observation.

        See `Observation.compute_obs`.
        """
        if stamp not in self.data.index:
            return None

        d_lat = traj_point.lat - self.data.lat.loc[stamp]
        d_lon = traj_point.lon - self.data.lon.loc[stamp]
        clat = np.cos(np.deg2rad(self.data.lat.loc[stamp]))
        z = np.array([
            np.deg2rad(d_lon) * earth.R0 * clat,
            np.deg2rad(d_lat) * earth.R0
        ])

        return z, self.H, self.R


class VeVnObs(Observation):
    """Observation of East and North velocity (from GPS or any other source).

    Parameters
    ----------
    data : DataFrame
        Must contain columns 'VE' and 'VN' for East and North velocity
        components. Index must contain time stamps.
    sd : float
        Measurement accuracy in m/s.
    gain_curve : None, callable or 3-tuple
        Kalman correction gain curve. It determines the proportionality of
        a state correction and a normalized measurement residual (by its
        theoretical covariance). In the standard Kalman correction it is an
        identity function. To make the filter robust to outliers a sublinear
        function can be provided. A convenient parametrization of such function
        is supported. It described by 3 numbers [L, F, C], if q is a normalized
        residual then:

            * If q < L: standard Kalman correction is used.
            * If L <= q < F: correction is kept constant on a level of q = L.
            * If F <= q < C: correction decays to 0 as ~1/q.
            * IF q >= C: the measurement is rejected completely.

        If None (default), the standard Kalman correction will be used.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.
    """
    def __init__(self, data, sd, gain_curve=None):
        super(VeVnObs, self).__init__(data, gain_curve)
        self.R = np.diag([sd, sd]) ** 2

    def compute_obs(self, stamp, traj_point):
        """Compute ingredients for a single observation.

        See `Observation.compute_obs`.
        """
        if stamp not in self.data.index:
            return None

        VE = self.data.VE.loc[stamp]
        VN = self.data.VN.loc[stamp]

        z = np.array([traj_point.VE - VE, traj_point.VN - VN])
        H = np.zeros((2, N_BASE_STATES))

        H[0, DV1] = 1
        H[0, PSI3] = VN
        H[1, DV2] = 1
        H[1, PSI3] = -VE

        return z, H, self.R


def _errors_transform_matrix(traj):
    lat = np.deg2rad(traj.lat)
    VE = traj.VE
    VN = traj.VN
    h = np.deg2rad(traj.h)
    p = np.deg2rad(traj.p)

    tlat = np.tan(lat)
    sh, ch = np.sin(h), np.cos(h)
    cp, tp = np.cos(p), np.tan(p)

    T = np.zeros((traj.shape[0], N_BASE_STATES, N_BASE_STATES))
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 0] = VN * tlat / earth.R0
    T[:, 2, 2] = 1
    T[:, 2, 6] = VN
    T[:, 3, 0] = -VE * tlat / earth.R0
    T[:, 3, 3] = 1
    T[:, 3, 6] = -VE
    T[:, 4, 0] = tlat / earth.R0
    T[:, 4, 4] = -sh * tp
    T[:, 4, 5] = -ch * tp
    T[:, 4, 6] = 1
    T[:, 5, 4] = -ch
    T[:, 5, 5] = sh
    T[:, 6, 4] = -sh / cp
    T[:, 6, 5] = -ch / cp

    return T


def _error_model_matrices(traj):
    n_samples = traj.shape[0]
    lat = np.deg2rad(traj.lat)
    slat, clat = np.sin(lat), np.cos(lat)
    tlat = slat / clat

    u = np.zeros((n_samples, 3))
    u[:, 1] = earth.RATE * clat
    u[:, 2] = earth.RATE * slat

    rho = np.empty((n_samples, 3))
    rho[:, 0] = -traj.VN / earth.R0
    rho[:, 1] = traj.VE / earth.R0
    rho[:, 2] = rho[:, 1] * tlat

    Cnb = dcm.from_hpr(traj.h, traj.p, traj.r)

    F = np.zeros((n_samples, N_BASE_STATES, N_BASE_STATES))

    F[:, DR1, DR2] = rho[:, 2]
    F[:, DR1, DV1] = 1
    F[:, DR1, PSI3] = traj.VN

    F[:, DR2, DR1] = -rho[:, 2]
    F[:, DR2, DV2] = 1
    F[:, DR2, PSI3] = -traj.VE

    F[:, DV1, DV2] = 2 * u[:, 2] + rho[:, 2]
    F[:, DV1, PHI2] = -earth.G0

    F[:, DV2, DV1] = -2 * u[:, 2] - rho[:, 2]
    F[:, DV2, PHI1] = earth.G0

    F[:, PHI1, DR1] = -u[:, 2] / earth.R0
    F[:, PHI1, DV2] = -1 / earth.R0
    F[:, PHI1, PHI2] = u[:, 2] + rho[:, 2]
    F[:, PHI1, PSI3] = -u[:, 1]

    F[:, PHI2, DR2] = -u[:, 2] / earth.R0
    F[:, PHI2, DV1] = 1 / earth.R0
    F[:, PHI2, PHI1] = -u[:, 2] - rho[:, 2]
    F[:, PHI2, PSI3] = u[:, 0]

    F[:, PSI3, DR1] = (u[:, 0] + rho[:, 0]) / earth.R0
    F[:, PSI3, DR2] = (u[:, 1] + rho[:, 1]) / earth.R0
    F[:, PSI3, PHI1] = u[:, 1] + rho[:, 1]
    F[:, PSI3, PHI2] = -u[:, 0] - rho[:, 0]

    B_gyro = np.zeros((n_samples, N_BASE_STATES, 3))
    B_gyro[np.ix_(np.arange(n_samples), [PHI1, PHI2, PSI3], [0, 1, 2])] = -Cnb

    B_accel = np.zeros((n_samples, N_BASE_STATES, 3))
    B_accel[np.ix_(np.arange(n_samples), [DV1, DV2], [0, 1, 2])] = Cnb[:, :2]

    return F, B_gyro, B_accel


def propagate_errors(dt, traj, d_lat=0, d_lon=0, d_VE=0, d_VN=0,
                     d_h=0, d_p=0, d_r=0, d_gyro=0, d_accel=0):
    """Deterministic linear propagation of INS errors.

    Parameters
    ----------
    dt : float
        Time step per stamp.
    traj : DataFrame
        Trajectory.
    d_lat, d_lon : float
        Initial position errors in meters.
    d_VE, d_VN : float
        Initial velocity errors.
    d_h, d_p, d_r : float
        Initial heading, pitch and roll errors.
    d_gyro, d_accel : array_like
        Gyro and accelerometer errors (in SI units). Can be constant or
        specified for each time stamp in `traj`.

    Returns
    -------
    traj_err : DataFrame
        Trajectory errors.
    """
    Fi, Fig, Fia = _error_model_matrices(traj)
    Phi = 0.5 * (Fi[1:] + Fi[:-1]) * dt
    Phi[:] += np.identity(Phi.shape[-1])

    d_gyro = np.asarray(d_gyro)
    d_accel = np.asarray(d_accel)
    if d_gyro.ndim == 0:
        d_gyro = np.resize(d_gyro, 3)
    if d_accel.ndim == 0:
        d_accel = np.resize(d_accel, 3)

    d_gyro = util.mv_prod(Fig, d_gyro)
    d_accel = util.mv_prod(Fia, d_accel)
    d_sensor = 0.5 * (d_gyro[1:] + d_gyro[:-1] + d_accel[1:] + d_accel[:-1])

    T = _errors_transform_matrix(traj)
    d_h = np.deg2rad(d_h)
    d_p = np.deg2rad(d_p)
    d_r = np.deg2rad(d_r)
    x0 = np.array([d_lon, d_lat, d_VE, d_VN, d_h, d_p, d_r])
    x0 = np.linalg.inv(T[0]).dot(x0)

    n_samples = Fi.shape[0]
    x = np.empty((n_samples, N_BASE_STATES))
    x[0] = x0
    for i in range(n_samples - 1):
        x[i + 1] = Phi[i].dot(x[i]) + d_sensor[i] * dt

    x = util.mv_prod(T, x)
    error = pd.DataFrame(index=traj.index)
    error['lat'] = x[:, DRN]
    error['lon'] = x[:, DRE]
    error['VE'] = x[:, DVE]
    error['VN'] = x[:, DVN]
    error['h'] = np.rad2deg(x[:, DH])
    error['p'] = np.rad2deg(x[:, DP])
    error['r'] = np.rad2deg(x[:, DR])

    return error


def _kalman_correct(x, P, z, H, R, gain_factor, gain_curve):
    PHT = np.dot(P, H.T)

    S = np.dot(H, PHT) + R
    e = z - H.dot(x)
    L = cholesky(S, lower=True)
    inn = solve_triangular(L, e, lower=True)

    if gain_curve is not None:
        q = (np.dot(inn, inn) / inn.shape[0]) ** 0.5
        f = gain_curve(q)
        if f == 0:
            return inn
        L *= (q / f) ** 0.5

    K = cho_solve((L, True), PHT.T, overwrite_b=True).T
    if gain_factor is not None:
        K *= gain_factor[:, None]

    U = -K.dot(H)
    U[np.diag_indices_from(U)] += 1
    x += K.dot(z - H.dot(x))
    P[:] = U.dot(P).dot(U.T) + K.dot(R).dot(K.T)

    return inn


def _refine_stamps(stamps, max_step):
    stamps = np.sort(np.unique(stamps))
    ds = np.diff(stamps)
    ds_new = []
    for d in ds:
        if d > max_step:
            repeat, left = divmod(d, max_step)
            ds_new.append([max_step] * repeat)
            if left > 0:
                ds_new.append(left)
        else:
            ds_new.append(d)
    ds_new = np.hstack(ds_new)
    stamps_new = stamps[0] + np.cumsum(ds_new)
    return np.hstack((stamps[0], stamps_new))


class FeedforwardFilter:
    """INS Kalman filter in a feedforward form.

    Parameters
    ----------
    dt : float
        Time step per stamp.
    traj_ref : DataFrame
        Trajectory which is used to propagate the error model. It should be
        reasonably accurate and must be recorded at each successive time stamp
        without skips.
    pos_sd : float
        Initial position uncertainty in meters.
    vel_sd : float
        Initial velocity uncertainty.
    azimuth_sd : float
        Initial azimuth (heading) uncertainty.
    level_sd : float
        Initial level (pitch and roll) uncertainty.
    gyro_model, accel_model : None or `InertialSensor`, optional
        Error models for gyros and accelerometers. If None (default), an empty
        model will be used.
    gyro, accel : array_like or None, optional
        Gyro and accelerometer readings, required only if a scale factor is
        modeled in `gyro_model` and `accel_model` respectively.

    Attributes
    ----------
    n_states : int
        Number of states.
    n_noises : int
        Number of noise sources.
    states : OrderedDict
        Dictionary mapping state names to their indices.
    """
    def __init__(self, dt, traj_ref, pos_sd, vel_sd, azimuth_sd, level_sd,
                 gyro_model=None, accel_model=None, gyro=None, accel=None):
        if gyro_model is None:
            gyro_model = InertialSensor()
        if accel_model is None:
            accel_model = InertialSensor()

        if gyro_model.scale is not None and gyro is None:
            raise ValueError("`gyro_model` contains scale factor errors, "
                             "thus you must provide `gyro`.")
        if accel_model.scale is not None and accel is None:
            raise ValueError("`accel_model` contains scale factor errors, "
                             "thus you must provide `accel`.")

        self.traj_ref = traj_ref

        n_points = traj_ref.shape[0]
        n_states = N_BASE_STATES + gyro_model.n_states + accel_model.n_states
        n_noises = (gyro_model.n_noises + accel_model.n_noises +
                    3 * (gyro_model.noise is not None) +
                    3 * (accel_model.noise is not None))

        F = np.zeros((n_points, n_states, n_states))
        G = np.zeros((n_points, n_states, n_noises))
        q = np.zeros(n_noises)
        P0 = np.zeros((n_states, n_states))

        n = N_BASE_STATES
        n1 = gyro_model.n_states
        n2 = accel_model.n_states

        states = OrderedDict((
            ('DR1', DR1),
            ('DR2', DR2),
            ('DV1', DV1),
            ('DV2', DV2),
            ('PHI1', PHI1),
            ('PHI2', PHI2),
            ('PSI3', PSI3)
        ))
        for name, state in gyro_model.states.items():
            states['GYRO_' + name] = n + state
        for name, state in accel_model.states.items():
            states['ACCEL_' + name] = n + n1 + state

        level_sd = np.deg2rad(level_sd)
        azimuth_sd = np.deg2rad(azimuth_sd)

        P0[DR1, DR1] = P0[DR2, DR2] = pos_sd ** 2
        P0[DV1, DV1] = P0[DV2, DV2] = vel_sd ** 2
        P0[PHI1, PHI1] = P0[PHI2, PHI2] = level_sd ** 2
        P0[PSI3, PSI3] = azimuth_sd ** 2

        P0[n: n + n1, n: n + n1] = gyro_model.P
        P0[n + n1: n + n1 + n2, n + n1: n + n1 + n2] = accel_model.P

        self.P0 = P0

        Fi, Fig, Fia = _error_model_matrices(traj_ref)
        F[:, :n, :n] = Fi
        F[:, n: n + n1, n: n + n1] = gyro_model.F
        F[:, n + n1:n + n1 + n2, n + n1: n + n1 + n2] = accel_model.F

        if gyro is not None:
            gyro = np.asarray(gyro)
            gyro = gyro / dt
            gyro = np.vstack((gyro, 2 * gyro[-1] - gyro[-2]))

        if accel is not None:
            accel = np.asarray(accel)
            accel = accel / dt
            accel = np.vstack((accel, 2 * accel[-1] - accel[-2]))

        H_gyro = gyro_model.output_matrix(gyro)
        H_accel = accel_model.output_matrix(accel)
        F[:, :n, n: n + n1] = util.mm_prod(Fig, H_gyro)
        F[:, :n, n + n1: n + n1 + n2] = util.mm_prod(Fia, H_accel)

        s = 0
        s1 = gyro_model.n_noises
        s2 = accel_model.n_noises
        if gyro_model.noise is not None:
            G[:, :n, :3] = Fig
            q[:3] = gyro_model.noise
            s += 3
        if accel_model.noise is not None:
            G[:, :n, s: s + 3] = Fia
            q[s: s + 3] = accel_model.noise
            s += 3

        G[:, n: n + n1, s: s + s1] = gyro_model.G
        q[s: s + s1] = gyro_model.q
        G[:, n + n1: n + n1 + n2, s + s1: s + s1 + s2] = accel_model.G
        q[s + s1: s + s1 + s2] = accel_model.q

        self.F = F
        self.q = q
        self.G = G

        states = OrderedDict((
            ('DR1', DR1),
            ('DR2', DR2),
            ('DV1', DV1),
            ('DV2', DV2),
            ('PHI1', PHI1),
            ('PHI2', PHI2),
            ('PSI3', PSI3)
        ))
        for name, state in gyro_model.states.items():
            states['GYRO_' + name] = n + state
        for name, state in accel_model.states.items():
            states['ACCEL_' + name] = n + n1 + state

        self.dt = dt
        self.n_points = n_points
        self.n_states = n_states
        self.n_noises = n_noises
        self.states = states

        self.gyro_model = gyro_model
        self.accel_model = accel_model

    def _validate_parameters(self, traj, observations, gain_factor,
                             max_step, record_stamps):
        if traj is None:
            traj = self.traj_ref

        if not np.all(traj.index == self.traj_ref.index):
            raise ValueError("Time stamps of reference and computed "
                             "trajectories don't match.")

        if gain_factor is not None:
            gain_factor = np.asarray(gain_factor)
            if gain_factor.shape != (self.n_states,):
                raise ValueError("`gain_factor` is expected to have shape {}, "
                                 "but actually has {}."
                                 .format((self.n_states,), gain_factor.shape))
            if np.any(gain_factor < 0):
                raise ValueError("`gain_factor` must contain positive values.")

        if observations is None:
            observations = []

        obs_stamps = pd.Index([])
        for obs in observations:
            obs_stamps = obs_stamps.union(obs.data.index)

        start, end = traj.index[0], traj.index[-1]
        stamps = pd.Index([start, end])
        stamps = stamps.union(obs_stamps)

        if record_stamps is not None:
            end = min(end, record_stamps[-1])
            record_stamps = record_stamps[(record_stamps >= start) &
                                          (record_stamps <= end)]
            stamps = stamps.union(pd.Index(record_stamps))

        stamps = stamps[(stamps >= start) & (stamps <= end)]

        max_step = max(1, int(np.floor(max_step / self.dt)))
        stamps = _refine_stamps(stamps, max_step)

        if record_stamps is None:
            record_stamps = stamps

        return traj, observations, stamps, record_stamps, obs_stamps

    def _compute_output_errors(self, traj, x, P, output_stamps):
        T = _errors_transform_matrix(traj.loc[output_stamps])
        y = util.mv_prod(T, x[:, :N_BASE_STATES])
        Py = util.mm_prod(T, P[:, :N_BASE_STATES, :N_BASE_STATES])
        Py = util.mm_prod(Py, T, bt=True)
        sd_y = np.diagonal(Py, axis1=1, axis2=2) ** 0.5

        err = pd.DataFrame(index=output_stamps)
        err['lat'] = y[:, DRN]
        err['lon'] = y[:, DRE]
        err['VE'] = y[:, DVE]
        err['VN'] = y[:, DVN]
        err['h'] = np.rad2deg(y[:, DH])
        err['p'] = np.rad2deg(y[:, DP])
        err['r'] = np.rad2deg(y[:, DR])

        sd = pd.DataFrame(index=output_stamps)
        sd['lat'] = sd_y[:, DRN]
        sd['lon'] = sd_y[:, DRE]
        sd['VE'] = sd_y[:, DVE]
        sd['VN'] = sd_y[:, DVN]
        sd['h'] = np.rad2deg(sd_y[:, DH])
        sd['p'] = np.rad2deg(sd_y[:, DP])
        sd['r'] = np.rad2deg(sd_y[:, DR])

        gyro_err = pd.DataFrame(index=output_stamps)
        gyro_sd = pd.DataFrame(index=output_stamps)
        n = N_BASE_STATES
        for i, name in enumerate(self.gyro_model.states):
            gyro_err[name] = x[:, n + i]
            gyro_sd[name] = P[:, n + i, n + i] ** 0.5

        accel_err = pd.DataFrame(index=output_stamps)
        accel_sd = pd.DataFrame(index=output_stamps)
        ng = self.gyro_model.n_states
        for i, name in enumerate(self.accel_model.states):
            accel_err[name] = x[:, n + ng + i]
            accel_sd[name] = P[:, n + ng + i, n + ng + i] ** 0.5

        return err, sd, gyro_err, gyro_sd, accel_err, accel_sd

    def _forward_pass(self, traj, observations, gain_factor, stamps,
                      record_stamps, data_for_backward=False):
        inds = stamps - stamps[0]

        if data_for_backward:
            n_stamps = stamps.shape[0]
            x = np.empty((n_stamps, self.n_states))
            P = np.empty((n_stamps, self.n_states, self.n_states))
            xa = x.copy()
            Pa = P.copy()
            Phi_arr = np.empty((n_stamps - 1, self.n_states, self.n_states))
            record_stamps = stamps
        else:
            n_stamps = record_stamps.shape[0]
            x = np.empty((n_stamps, self.n_states))
            P = np.empty((n_stamps, self.n_states, self.n_states))
            xa = None
            Pa = None
            Phi_arr = None

        xc = np.zeros(self.n_states)
        Pc = self.P0.copy()
        i_save = 0

        H_max = np.zeros((10, self.n_states))

        obs_stamps = [[] for _ in range(len(observations))]
        obs_residuals = [[] for _ in range(len(observations))]

        for i in range(stamps.shape[0] - 1):
            stamp = stamps[i]
            ind = inds[i]
            next_ind = inds[i + 1]

            if data_for_backward and record_stamps[i_save] == stamp:
                xa[i_save] = xc
                Pa[i_save] = Pc

            for i_obs, obs in enumerate(observations):
                ret = obs.compute_obs(stamp, traj.loc[stamp])
                if ret is not None:
                    z, H, R = ret
                    H_max[:H.shape[0], :N_BASE_STATES] = H
                    res = _kalman_correct(xc, Pc, z, H_max[:H.shape[0]], R,
                                          gain_factor, obs.gain_curve)
                    obs_stamps[i_obs].append(stamp)
                    obs_residuals[i_obs].append(res)

            if record_stamps[i_save] == stamp:
                x[i_save] = xc
                P[i_save] = Pc
                i_save += 1

            dt = self.dt * (next_ind - ind)
            Phi = 0.5 * (self.F[ind] + self.F[next_ind]) * dt
            Phi[np.diag_indices_from(Phi)] += 1
            Qd = 0.5 * (self.G[ind] + self.G[next_ind])
            Qd *= self.q
            Qd = np.dot(Qd, Qd.T) * dt

            if data_for_backward:
                Phi_arr[i] = Phi

            xc = Phi.dot(xc)
            Pc = Phi.dot(Pc).dot(Phi.T) + Qd

        x[-1] = xc
        P[-1] = Pc

        if data_for_backward:
            xa[-1] = xc
            Pa[-1] = Pc

        residuals = []
        for s, r in zip(obs_stamps, obs_residuals):
            residuals.append(pd.DataFrame(index=s, data=np.asarray(r)))

        return x, P, xa, Pa, Phi_arr, residuals

    def run(self, traj=None, observations=[], gain_factor=None, max_step=1,
            record_stamps=None):
        """Run the filter.

        Parameters
        ----------
        traj : DataFrame or None
            Trajectory computed by INS of which to estimate the errors.
            If None (default), use `traj_ref` from the constructor.
        observations : list of `Observation`
            Observations which will be processed. Empty by default.
        gain_factor : array_like with shape (n_states,) or None
            Factor for Kalman gain for each filter state. It might be
            beneficial in some practical situations to set factors less than 1
            in order to decrease influence of measurements on some states.
            Setting values higher than 1 is unlikely to be reasonable. If None
            (default), use standard optimal Kalman gain.
        max_step : float, optional
            Maximum allowed time step in seconds for errors propagation.
            Default is 1 second. Set to 0 if you desire the smallest possible
            step.
        record_stamps : array_like or None
            Stamps at which record estimated errors. If None (default), errors
            will be saved at each stamp used internally in the filter.

        Returns
        -------
        Bunch object with the fields listed below. Note that all data frames
        contain stamps only presented in `record_stamps`.
        traj : DataFrame
            Trajectory corrected by estimated errors.
        err, sd : DataFrame
            Estimated trajectory errors and their standard deviations.
        gyro_err, gyro_sd : DataFrame
            Estimated gyro error and their standard deviations.
        accel_err, accel_sd : DataFrame
            Estimated accelerometer errors and their standard deviations.
        x : ndarray, shape (n_points, n_states)
            History of the filter states.
        P : ndarray, shape (n_points, n_states, n_states)
            History of the filter covariance.
        """
        traj, observations, stamps, record_stamps, _ = \
            self._validate_parameters(traj, observations, gain_factor,
                                      max_step, record_stamps)

        x, P, _, _, _, residuals = self._forward_pass(
            traj, observations, gain_factor, stamps, record_stamps)

        err, sd, gyro_err, gyro_sd, accel_err, accel_sd = \
            self._compute_output_errors(self.traj_ref, x, P, record_stamps)

        traj_corr = correct_traj(traj, err)

        return FiltResult(traj=traj_corr, err=err, sd=sd, gyro_err=gyro_err,
                          gyro_sd=gyro_sd, accel_err=accel_err,
                          accel_sd=accel_sd, residuals=residuals)

    def run_smoother(self, traj=None, observations=[], gain_factor=None,
                     max_step=1, record_stamps=None):
        """Run the smoother.

        It means that observations during the whole time is used to estimate
        the errors at each moment of time (i.e. it is not real time). The
        Rauch-Tung-Striebel two pass recursion is used [1]_.

        Parameters
        ----------
        traj : DataFrame or None
            Trajectory computed by INS of which to estimate the errors.
            If None (default), use `traj_ref` from the constructor.
        observations : list of `Observation`
            Observations which will be processed. Empty by default.
        gain_factor : array_like with shape (n_states,) or None
            Factor for Kalman gain for each filter state. It might be
            beneficial in some practical situations to set factors less than 1
            in order to decrease influence of measurements on some states.
            Setting values higher than 1 is unlikely to be reasonable. If None
            (default), use standard optimal Kalman gain.
        max_step : float, optional
            Maximum allowed time step in seconds for errors propagation.
            Default is 1 second. Set to 0 if you desire the smallest possible
            step.
        record_stamps : array_like or None
            Stamps at which record estimated errors. If None (default), errors
            will be saved at each stamp used internally in the filter.

        Returns
        -------
        Bunch object with the fields listed below. Note that all data frames
        contain stamps only presented in `record_stamps`.
        traj : DataFrame
            Trajectory corrected by estimated errors. It will only contain
            stamps presented in `record_stamps`.
        err, sd : DataFrame
            Estimated trajectory errors and their standard deviations.
        gyro_err, gyro_sd : DataFrame
            Estimated gyro error and their standard deviations.
        accel_err, accel_sd : DataFrame
            Estimated accelerometer errors and their standard deviations.
        x : ndarray, shape (n_points, n_states)
            History of the filter states.
        P : ndarray, shape (n_points, n_states, n_states)
            History of the filter covariance.

        References
        ----------
        .. [1] H. E. Rauch, F. Tung and C.T. Striebel, "Maximum Likelihood
               Estimates of Linear Dynamic Systems", AIAA Journal, Vol. 3,
               No. 8, August 1965.
        """
        traj, observations, stamps, record_stamps, _ = \
            self._validate_parameters(traj, observations, gain_factor,
                                      max_step, record_stamps)

        x, P, xa, Pa, Phi_arr, residuals = self._forward_pass(
            traj, observations, gain_factor, stamps, record_stamps,
            data_for_backward=True)

        I = np.identity(self.n_states)
        for i in reversed(range(x.shape[0] - 1)):
            L = cholesky(Pa[i + 1], check_finite=False)
            Pa_inv = cho_solve((L, False), I, check_finite=False)

            C = P[i].dot(Phi_arr[i].T).dot(Pa_inv)

            x[i] += C.dot(x[i + 1] - xa[i + 1])
            P[i] += C.dot(P[i + 1] - Pa[i + 1]).dot(C.T)

        ind = np.searchsorted(stamps, record_stamps)
        x = x[ind]
        P = P[ind]

        err, sd, gyro_err, gyro_sd, accel_err, accel_sd = \
            self._compute_output_errors(self.traj_ref, x, P, record_stamps)

        traj_corr = correct_traj(traj, err)

        return FiltResult(traj=traj_corr, err=err, sd=sd, gyro_err=gyro_err,
                          gyro_sd=gyro_sd, accel_err=accel_err,
                          accel_sd=accel_sd, residuals=residuals)


class FeedbackFilter:
    """INS Kalman filter with feedback corrections.

    Parameters
    ----------
    dt : float
        Time step per stamp.
    pos_sd : float
        Initial position uncertainty in meters.
    vel_sd : float
        Initial velocity uncertainty.
    azimuth_sd : float
        Initial azimuth (heading) uncertainty.
    level_sd : float
        Initial level (pitch and roll) uncertainty.
    gyro_model, accel_model : None or `InertialSensor`, optional
        Error models for gyros and accelerometers. If None (default), an empty
        model will be used.

    Attributes
    ----------
    n_states : int
        Number of states.
    n_noises : int
        Number of noise sources.
    states : OrderedDict
        Dictionary mapping state names to their indices.
    """
    def __init__(self, dt, pos_sd, vel_sd, azimuth_sd, level_sd,
                 gyro_model=None, accel_model=None):
        if gyro_model is None:
            gyro_model = InertialSensor()
        if accel_model is None:
            accel_model = InertialSensor()

        n_states = N_BASE_STATES + gyro_model.n_states + accel_model.n_states
        n_noises = (gyro_model.n_noises + accel_model.n_noises +
                    3 * (gyro_model.noise is not None) +
                    3 * (accel_model.noise is not None))

        q = np.zeros(n_noises)
        P0 = np.zeros((n_states, n_states))

        n = N_BASE_STATES
        n1 = gyro_model.n_states
        n2 = accel_model.n_states

        level_sd = np.deg2rad(level_sd)
        azimuth_sd = np.deg2rad(azimuth_sd)

        P0[DR1, DR1] = P0[DR2, DR2] = pos_sd ** 2
        P0[DV1, DV1] = P0[DV2, DV2] = vel_sd ** 2
        P0[PHI1, PHI1] = P0[PHI2, PHI2] = level_sd ** 2
        P0[PSI3, PSI3] = azimuth_sd ** 2

        P0[n: n + n1, n: n + n1] = gyro_model.P
        P0[n + n1: n + n1 + n2, n + n1: n + n1 + n2] = accel_model.P
        self.P0 = P0

        s = 0
        s1 = gyro_model.n_noises
        s2 = accel_model.n_noises
        if gyro_model.noise is not None:
            q[:3] = gyro_model.noise
            s += 3
        if accel_model.noise is not None:
            q[s: s + 3] = accel_model.noise
            s += 3
        q[s: s + s1] = gyro_model.q
        q[s + s1: s + s1 + s2] = accel_model.q

        self.q = q

        states = OrderedDict((
            ('DR1', DR1),
            ('DR2', DR2),
            ('DV1', DV1),
            ('DV2', DV2),
            ('PHI1', PHI1),
            ('PHI2', PHI2),
            ('PSI3', PSI3)
        ))
        for name, state in gyro_model.states.items():
            states['GYRO_' + name] = n + state
        for name, state in accel_model.states.items():
            states['ACCEL_' + name] = n + n1 + state

        self.dt = dt
        self.n_states = n_states
        self.n_noises = n_noises
        self.states = states

        self.gyro_model = gyro_model
        self.accel_model = accel_model

    def run(self, integrator, theta, dv, observations=[], gain_factor=None,
            max_step=1, feedback_period=500, record_stamps=None):
        """Run the filter.

        Parameters
        ----------
        integrator : `pyins.integrate.Integrator` instance
            Integrator to use for INS state propagation. It must be
            initialized.
        theta, dv : ndarray, shape (n_readings, 3)
            Rotation vectors and velocity increments computed from gyro and
            accelerometer readings after applying coning and sculling
            corrections.
        observations : list of `Observation`
            Measurements which will be processed. Empty by default.
        gain_factor : array_like with shape (n_states,) or None, optional
            Factor for Kalman gain for each filter's state. It might be
            beneficial in some practical situations to set factors less than 1
            in order to decrease influence of measurements on some states.
            Setting values higher than 1 is unlikely to be reasonable. If None
            (default), use standard optimal Kalman gain.
        max_step : float, optional
            Maximum allowed time step. Default is 1 second. Set to 0 if you
            desire the smallest possible step.
        feedback_period : float
            Time after which INS state will be corrected by the estimated
            errors. Default is 500 seconds.
        record_stamps : array_like or None
            At which stamps record estimated errors. If None (default), errors
            will be saved at each stamp used internally in the filter.

        Returns
        -------
        Bunch object with the fields listed below. Note that all data frames
        contain stamps only presented in `record_stamps`.
        traj : DataFrame
            Trajectory corrected by estimated errors. It will only contain
            stamps presented in `record_stamps`.
        err, sd : DataFrame
            Estimated trajectory errors and their standard deviations.
        gyro_err, gyro_sd : DataFrame
            Estimated gyro error and their standard deviations.
        accel_err, accel_sd : DataFrame
            Estimated accelerometer errors and their standard deviations.
        x : ndarray, shape (n_points, n_states)
            History of the filter states.
        P : ndarray, shape (n_points, n_states, n_states)
            History of the filter covariance.
        residuals : list of DataFrame
            Each element is DataFrame with index being measurement time stamps
            and columns containing normalized measurement residuals.
        """
        if gain_factor is not None:
            gain_factor = np.asarray(gain_factor)
            if gain_factor.shape != (self.n_states,):
                raise ValueError("`gain_factor` is expected to have shape {}, "
                                 "but actually has {}."
                                 .format((self.n_states,), gain_factor.shape))
            if np.any(gain_factor < 0):
                raise ValueError("`gain_factor` must contain positive values.")

        stamps = pd.Index([])
        for obs in observations:
            stamps = stamps.union(obs.data.index)

        integrator.reset()

        n_readings = theta.shape[0]
        initial_stamp = integrator.traj.index[-1]

        start = initial_stamp
        end = start + n_readings
        if record_stamps is not None:
            end = min(end, record_stamps[-1])
            n_readings = end - start
            record_stamps = record_stamps[(record_stamps >= start) &
                                          (record_stamps <= end)]
            theta = theta[:n_readings]
            dv = dv[:n_readings]

        stamps = stamps.union(pd.Index([start, end]))

        feedback_period = max(1, int(np.floor(feedback_period / self.dt)))
        stamps = stamps.union(
            pd.Index(np.arange(0, n_readings, feedback_period) +
                     initial_stamp))

        if record_stamps is not None:
            stamps = stamps.union(pd.Index(record_stamps))

        stamps = stamps[(stamps >= start) & (stamps <= end)]

        max_step = max(1, int(np.floor(max_step / self.dt)))
        stamps = _refine_stamps(stamps, max_step)

        if record_stamps is None:
            record_stamps = stamps

        n_stamps = record_stamps.shape[0]

        x = np.empty((n_stamps, self.n_states))
        P = np.empty((n_stamps, self.n_states, self.n_states))

        xc = np.zeros(self.n_states)
        Pc = self.P0.copy()

        H_max = np.zeros((10, self.n_states))

        i_reading = 0  # Number of processed readings.
        i_stamp = 0  # Index of current stamp in stamps array.
        # Index of current position in x and P arrays for saving xc and Pc.
        i_save = 0

        n = N_BASE_STATES
        n1 = self.gyro_model.n_states
        n2 = self.accel_model.n_states

        if self.gyro_model.scale is not None:
            gyro = theta / self.dt
            gyro = np.vstack((gyro, 2 * gyro[-1] - gyro[-2]))
        else:
            gyro = None

        if self.accel_model.scale is not None:
            accel = dv / self.dt
            accel = np.vstack((accel, 2 * accel[-1] - accel[-2]))
        else:
            accel = None

        H_gyro = np.atleast_2d(self.gyro_model.output_matrix(gyro))
        H_accel = np.atleast_2d(self.accel_model.output_matrix(accel))

        F = np.zeros((self.n_states, self.n_states))
        F[n: n + n1, n: n + n1] = self.gyro_model.F
        F[n + n1:n + n1 + n2, n + n1: n + n1 + n2] = self.accel_model.F
        F1 = F
        F2 = F.copy()

        s = 0
        s1 = self.gyro_model.n_noises
        s2 = self.accel_model.n_noises
        if self.gyro_model.noise is not None:
            s += 3
        if self.accel_model.noise is not None:
            s += 3

        G = np.zeros((self.n_states, self.n_noises))
        G[n: n + n1, s: s + s1] = self.gyro_model.G
        G[n + n1: n + n1 + n2, s + s1: s + s1 + s2] = self.accel_model.G
        G1 = G
        G2 = G.copy()

        obs_stamps = [[] for _ in range(len(observations))]
        obs_residuals = [[] for _ in range(len(observations))]

        while i_reading < n_readings:
            theta_b = theta[i_reading: i_reading + feedback_period]
            dv_b = dv[i_reading: i_reading + feedback_period]

            traj_b = integrator.integrate(theta_b, dv_b)
            Fi, Fig, Fia = _error_model_matrices(traj_b)
            i = 0

            while i < theta_b.shape[0]:
                stamp = stamps[i_stamp]
                stamp_next = stamps[i_stamp + 1]
                delta_i = stamp_next - stamp
                i_next = i + delta_i
                for i_obs, obs in enumerate(observations):
                    ret = obs.compute_obs(stamp, traj_b.iloc[i])
                    if ret is not None:
                        z, H, R = ret
                        H_max[:H.shape[0], :N_BASE_STATES] = H
                        res = _kalman_correct(xc, Pc, z,
                                              H_max[:H.shape[0]], R,
                                              gain_factor, obs.gain_curve)
                        obs_stamps[i_obs].append(stamp)
                        obs_residuals[i_obs].append(res)

                if record_stamps[i_save] == stamp:
                    x[i_save] = xc.copy()
                    P[i_save] = Pc.copy()
                    i_save += 1

                dt = self.dt * delta_i
                F1[:n, :n] = Fi[i]
                F2[:n, :n] = Fi[i_next]

                if H_gyro.ndim == 2:
                    H_gyro_i = H_gyro
                    H_gyro_i_next = H_gyro
                else:
                    H_gyro_i = H_gyro[stamp - start]
                    H_gyro_i_next = H_gyro[stamp_next - start]

                if H_accel.ndim == 2:
                    H_accel_i = H_accel
                    H_accel_i_next = H_accel
                else:
                    H_accel_i = H_accel[stamp - start]
                    H_accel_i_next = H_accel[stamp_next - start]

                F1[:n, n: n + n1] = Fig[i].dot(H_gyro_i)
                F2[:n, n: n + n1] = Fig[i_next].dot(H_gyro_i_next)
                F1[:n, n + n1: n + n1 + n2] = Fia[i].dot(H_accel_i)
                F2[:n, n + n1: n + n1 + n2] = Fia[i_next].dot(H_accel_i_next)

                s = 0
                if self.gyro_model.noise is not None:
                    G1[:n, :3] = Fig[i]
                    G2[:n, :3] = Fig[i_next]
                    s += 3
                if self.accel_model.noise is not None:
                    G1[:n, s: s + 3] = Fia[i]
                    G2[:n, s: s + 3] = Fia[i_next]

                Phi = 0.5 * (F1 + F2) * dt
                Phi[np.diag_indices_from(Phi)] += 1

                Qd = 0.5 * (G1 + G2)
                Qd *= self.q
                Qd = np.dot(Qd, Qd.T) * dt

                xc = Phi.dot(xc)
                Pc = Phi.dot(Pc).dot(Phi.T) + Qd

                i = i_next
                i_stamp += 1

            i_reading += feedback_period
            integrator._correct(xc[:N_BASE_STATES])
            xc[:N_BASE_STATES] = 0

        if record_stamps[i_save] == stamps[i_stamp]:
            x[i_save] = xc
            P[i_save] = Pc
    
        T = _errors_transform_matrix(integrator.traj.loc[record_stamps])
        y = util.mv_prod(T, x[:, :N_BASE_STATES])
        Py = util.mm_prod(T, P[:, :N_BASE_STATES, :N_BASE_STATES])
        Py = util.mm_prod(Py, T, bt=True)
        sd_y = np.diagonal(Py, axis1=1, axis2=2) ** 0.5

        err = pd.DataFrame(index=record_stamps)
        err['lat'] = y[:, DRN]
        err['lon'] = y[:, DRE]
        err['VE'] = y[:, DVE]
        err['VN'] = y[:, DVN]
        err['h'] = np.rad2deg(y[:, DH])
        err['p'] = np.rad2deg(y[:, DP])
        err['r'] = np.rad2deg(y[:, DR])

        sd = pd.DataFrame(index=record_stamps)
        sd['lat'] = sd_y[:, DRN]
        sd['lon'] = sd_y[:, DRE]
        sd['VE'] = sd_y[:, DVE]
        sd['VN'] = sd_y[:, DVN]
        sd['h'] = np.rad2deg(sd_y[:, DH])
        sd['p'] = np.rad2deg(sd_y[:, DP])
        sd['r'] = np.rad2deg(sd_y[:, DR])

        gyro_err = pd.DataFrame(index=record_stamps)
        gyro_sd = pd.DataFrame(index=record_stamps)
        n = N_BASE_STATES
        for i, name in enumerate(self.gyro_model.states):
            gyro_err[name] = x[:, n + i]
            gyro_sd[name] = P[:, n + i, n + i] ** 0.5

        accel_err = pd.DataFrame(index=record_stamps)
        accel_sd = pd.DataFrame(index=record_stamps)
        ng = self.gyro_model.n_states
        for i, name in enumerate(self.accel_model.states):
            accel_err[name] = x[:, n + ng + i]
            accel_sd[name] = P[:, n + ng + i, n + ng + i] ** 0.5

        traj_corr = correct_traj(integrator.traj, err)

        residuals = []
        for s, r in zip(obs_stamps, obs_residuals):
            residuals.append(pd.DataFrame(index=s, data=np.asarray(r)))

        return FiltResult(traj=traj_corr, err=err, sd=sd, gyro_err=gyro_err,
                          gyro_sd=gyro_sd, accel_err=accel_err,
                          accel_sd=accel_sd, residuals=residuals)


def traj_diff(t1, t2):
    """Compute trajectory difference.

    Parameters
    ----------
    t1, t2 : DataFrame
        Trajectories.

    Returns
    -------
    diff : DataFrame
        Trajectory difference. It can be interpreted as errors in `t1` relative
        to `t2`.
    """
    diff = t1 - t2
    diff['lat'] *= np.deg2rad(earth.R0)
    diff['lon'] *= np.deg2rad(earth.R0) * np.cos(0.5 *
                                                 np.deg2rad(t1.lat + t2.lat))
    diff['h'] %= 360
    diff.h[diff.h < -180] += 360
    diff.h[diff.h > 180] -= 360

    return diff.dropna()


def correct_traj(traj, error):
    """Correct trajectory by estimated errors.

    Note that it means subtracting errors from the trajectory.

    Parameters
    ----------
    traj : DataFrame
        Trajectory.
    error : DataFrame
        Estimated errors.

    Returns
    -------
    traj_corr : DataFrame
        Corrected trajectory.
    """
    traj_corr = traj.copy()
    traj_corr['lat'] -= np.rad2deg(error.lat / earth.R0)
    traj_corr['lon'] -= np.rad2deg(error.lon / (earth.R0 *
                                   np.cos(np.deg2rad(traj_corr['lat']))))
    traj_corr['VE'] -= error.VE
    traj_corr['VN'] -= error.VN
    traj_corr['h'] -= error.h
    traj_corr['p'] -= error.p
    traj_corr['r'] -= error.r

    return traj_corr.dropna()
