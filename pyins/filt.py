"""Navigation Kalman filters."""
from collections import OrderedDict
import numpy as np
import pandas as pd
from . import dcm, error_models, kalman, util, transform
from .transform import correct_trajectory


class FiltResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = list(self.__dict__.keys())
        items = self.__dict__.items()
        if keys:
            m = max(map(len, keys)) + 1
            return '\n'.join(["{} : {}".format(k.rjust(m), type(v))
                              for k, v in sorted(items)])
        else:
            return self.__class__.__name__ + "()"


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

    Parameters
    ----------
    data : DataFrame
        Observed values as a DataFrame. Index must contain time stamps.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.

    See Also
    --------
    LatLonObs
    VeVnObs
    """
    def __init__(self, data):
        self.data = data

    def compute_obs(self, stamp, trajectory_point, error_model):
        """Compute ingredients for a single linearized observation.

        It must compute the observation model (z, H, R) at a given time stamp.
        If the observation is not available at a given `stamp`, it must return
        None.

        Parameters
        ----------
        stamp : int
            Time stamp.
        trajectory_point : Series
            Point of INS trajectory at `stamp`.
        error_model : ErrorModel
            Error model object.

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
        raise NotImplementedError


class PositionObs(Observation):
    """Observation of latitude and longitude (from GPS or any other source).

    Parameters
    ----------
    data : DataFrame
        Must contain columns 'lat', 'lon' and `alt` columns for latitude,
        longitude and altitude. Index must contain time stamps.
    sd : float
        Measurement accuracy in meters.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.
    """
    def __init__(self, data, sd):
        super(PositionObs, self).__init__(data)
        self.R = sd**2 * np.eye(3)

    def compute_obs(self, stamp, trajectory_point, error_model):
        if stamp not in self.data.index:
            return None

        z = transform.difference_lla(trajectory_point[['lat', 'lon', 'alt']],
                                     self.data.loc[stamp,
                                                   ['lat', 'lon', 'alt']])
        H = error_model.position_error_jacobian(trajectory_point)

        return z, H, self.R


class EnuVelocityObs(Observation):
    """Observation of velocity resolved in ENU frame.

    Parameters
    ----------
    data : DataFrame
        Must contain columns 'VE', 'VN' and 'VU' columns.
        Index must contain time stamps.
    sd : float
        Measurement accuracy in m/s.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.
    """
    def __init__(self, data, sd):
        super(EnuVelocityObs, self).__init__(data)
        self.R = sd**2 * np.eye(3)

    def compute_obs(self, stamp, trajectory_point, error_model):
        if stamp not in self.data.index:
            return None

        z = trajectory_point[['VE', 'VN', 'VU']] - \
            self.data.loc[stamp,  ['VE', 'VN', 'VU']]
        H = error_model.enu_velocity_error_jacobian(trajectory_point)

        return z, H, self.R


class BodyVelocityObs(Observation):
    """Observation of velocity resolved in body frame.

    Parameters
    ----------
    data : DataFrame
        Must contain columns 'VX', 'VY' and 'VZ' columns.
        Index must contain time stamps.
    sd : float
        Measurement accuracy in m/s.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.
    """
    def __init__(self, data, sd):
        super(BodyVelocityObs, self).__init__(data)
        self.R = sd**2 * np.eye(3)

    def compute_obs(self, stamp, trajectory_point, error_model):
        if stamp not in self.data.index:
            return None

        Cnb = dcm.from_rph(trajectory_point[['roll', 'pitch', 'heading']])
        z = Cnb.transpose() @ trajectory_point[['VE', 'VN', 'VU']] - \
            self.data.loc[stamp, ['VX', 'VY', 'VZ']]
        H = error_model.body_velocity_error_jacobian(trajectory_point)
        return z, H, self.R


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


def _compute_output_errors(traj, x, P, output_stamps,
                           error_model, gyro_model, accel_model):
    T = error_model.transform_to_output(traj.loc[output_stamps])
    y = util.mv_prod(T, x[:, :error_model.N_STATES])
    Py = util.mm_prod(T, P[:, :error_model.N_STATES, :error_model.N_STATES])
    Py = util.mm_prod(Py, T, bt=True)
    sd_y = np.diagonal(Py, axis1=1, axis2=2) ** 0.5

    err = pd.DataFrame(index=output_stamps)
    err['east'] = y[:, error_model.DRE]
    err['north'] = y[:, error_model.DRN]
    err['up'] = y[:, error_model.DRU]
    err['VE'] = y[:, error_model.DVE]
    err['VN'] = y[:, error_model.DVN]
    err['VU'] = y[:, error_model.DVU]
    err['roll'] = np.rad2deg(y[:, error_model.DROLL])
    err['pitch'] = np.rad2deg(y[:, error_model.DPITCH])
    err['heading'] = np.rad2deg(y[:, error_model.DHEADING])

    sd = pd.DataFrame(index=output_stamps)
    sd['east'] = sd_y[:, error_model.DRE]
    sd['north'] = sd_y[:, error_model.DRN]
    sd['up'] = sd_y[:, error_model.DRU]
    sd['VE'] = sd_y[:, error_model.DVE]
    sd['VN'] = sd_y[:, error_model.DVN]
    sd['VU'] = sd_y[:, error_model.DVU]
    sd['roll'] = np.rad2deg(sd_y[:, error_model.DROLL])
    sd['pitch'] = np.rad2deg(sd_y[:, error_model.DPITCH])
    sd['heading'] = np.rad2deg(sd_y[:, error_model.DHEADING])

    gyro_err = pd.DataFrame(index=output_stamps)
    gyro_sd = pd.DataFrame(index=output_stamps)
    n = error_model.N_STATES
    for i, name in enumerate(gyro_model.states):
        gyro_err[name] = x[:, n + i]
        gyro_sd[name] = P[:, n + i, n + i] ** 0.5

    accel_err = pd.DataFrame(index=output_stamps)
    accel_sd = pd.DataFrame(index=output_stamps)
    ng = gyro_model.n_states
    for i, name in enumerate(accel_model.states):
        accel_err[name] = x[:, n + ng + i]
        accel_sd[name] = P[:, n + ng + i, n + ng + i] ** 0.5

    return err, sd, gyro_err, gyro_sd, accel_err, accel_sd


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
                 error_model=error_models.ModifiedPhiModel(),
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
        n_states = error_model.N_STATES + gyro_model.n_states + \
                   accel_model.n_states
        n_noises = (gyro_model.n_noises + accel_model.n_noises +
                    3 * (gyro_model.noise is not None) +
                    3 * (accel_model.noise is not None))

        F = np.zeros((n_points, n_states, n_states))
        G = np.zeros((n_points, n_states, n_noises))
        q = np.zeros(n_noises)
        P0 = np.zeros((n_states, n_states))

        n = error_model.N_STATES
        n1 = gyro_model.n_states
        n2 = accel_model.n_states

        states = error_model.STATES
        for name, state in gyro_model.states.items():
            states['GYRO_' + name] = n + state
        for name, state in accel_model.states.items():
            states['ACCEL_' + name] = n + n1 + state

        level_sd = np.deg2rad(level_sd)
        azimuth_sd = np.deg2rad(azimuth_sd)

        T = np.linalg.inv(error_model.transform_to_output(traj_ref.iloc[0]))
        P_nav = np.zeros((error_model.N_STATES, error_model.N_STATES))
        P_nav[error_model.DRE, error_model.DRE] = pos_sd ** 2
        P_nav[error_model.DRN, error_model.DRN] = pos_sd ** 2
        P_nav[error_model.DRU, error_model.DRU] = pos_sd ** 2
        P_nav[error_model.DVE, error_model.DVE] = vel_sd ** 2
        P_nav[error_model.DVN, error_model.DVN] = vel_sd ** 2
        P_nav[error_model.DVU, error_model.DVU] = vel_sd ** 2
        P_nav[error_model.DROLL, error_model.DROLL] = level_sd ** 2
        P_nav[error_model.DPITCH, error_model.DPITCH] = level_sd ** 2
        P_nav[error_model.DHEADING, error_model.DHEADING] = azimuth_sd ** 2

        P0[:n, :n] = T @ P_nav @ T.transpose()
        P0[n: n + n1, n: n + n1] = gyro_model.P
        P0[n + n1: n + n1 + n2, n + n1: n + n1 + n2] = accel_model.P

        self.P0 = P0

        Fi, Fig, Fia = error_model.system_matrix(traj_ref)
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

        self.dt = dt
        self.n_points = n_points
        self.n_states = n_states
        self.n_noises = n_noises
        self.states = states

        self.error_model = error_model
        self.gyro_model = gyro_model
        self.accel_model = accel_model

    def _validate_parameters(self, traj, observations, max_step, record_stamps):
        if traj is None:
            traj = self.traj_ref

        if not np.all(traj.index == self.traj_ref.index):
            raise ValueError("Time stamps of reference and computed "
                             "trajectories don't match.")

        if observations is None:
            observations = []

        stamps = pd.Index([])
        for obs in observations:
            stamps = stamps.union(obs.data.index)

        start, end = traj.index[0], traj.index[-1]
        stamps = stamps.union(pd.Index([start, end]))

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

        return traj, observations, stamps, record_stamps

    def _forward_pass(self, traj, observations, stamps,
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
                ret = obs.compute_obs(stamp, traj.loc[stamp], self.error_model)
                if ret is not None:
                    z, H, R = ret
                    H_max[:H.shape[0], :self.error_model.N_STATES] = H
                    res = kalman.correct(xc, Pc, z, H_max[:H.shape[0]], R)
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

    def run(self, traj=None, observations=[], max_step=1, record_stamps=None):
        """Run the filter.

        Parameters
        ----------
        traj : DataFrame or None
            Trajectory computed by INS of which to estimate the errors.
            If None (default), use `traj_ref` from the constructor.
        observations : list of `Observation`
            Observations which will be processed. Empty by default.
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
            Estimated gyro error states and their standard deviations.
        accel_err, accel_sd : DataFrame
            Estimated accelerometer error states and their standard deviations.
        x : ndarray, shape (n_points, n_states)
            History of the filter states.
        P : ndarray, shape (n_points, n_states, n_states)
            History of the filter covariance.
        residuals : list of DataFrame
            Each DataFrame corresponds to an observation from `observations`.
            Its index is observation time stamps and columns contain normalized
            observations residuals for each component of the observation
            vector `z`.
        """
        traj, observations, stamps, record_stamps = \
            self._validate_parameters(traj, observations, max_step,
                                      record_stamps)

        x, P, _, _, _, residuals = self._forward_pass(traj, observations,
                                                      stamps, record_stamps)

        err, sd, gyro_err, gyro_sd, accel_err, accel_sd = \
            _compute_output_errors(self.traj_ref, x, P, record_stamps,
                                   self.error_model, self.gyro_model,
                                   self.accel_model)

        traj_corr = correct_trajectory(traj, err)

        return FiltResult(traj=traj_corr, err=err, sd=sd, gyro_err=gyro_err,
                          gyro_sd=gyro_sd, accel_err=accel_err,
                          accel_sd=accel_sd, x=x, P=P, residuals=residuals)

    def run_smoother(self, traj=None, observations=[], max_step=1,
                     record_stamps=None):
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
            Estimated gyro error states and their standard deviations.
        accel_err, accel_sd : DataFrame
            Estimated accelerometer error states and their standard deviations.
        x : ndarray, shape (n_points, n_states)
            History of the filter states.
        P : ndarray, shape (n_points, n_states, n_states)
            History of the filter covariance.
        residuals : list of DataFrame
            Each DataFrame corresponds to an observation from `observations`.
            Its index is observation time stamps and columns contain normalized
            observations residuals for each component of the observation
            vector `z`.

        References
        ----------
        .. [1] H. E. Rauch, F. Tung and C.T. Striebel, "Maximum Likelihood
               Estimates of Linear Dynamic Systems", AIAA Journal, Vol. 3,
               No. 8, August 1965.
        """
        traj, observations, stamps, record_stamps = \
            self._validate_parameters(traj, observations, max_step,
                                      record_stamps)

        x, P, xa, Pa, Phi_arr, residuals = self._forward_pass(
            traj, observations, stamps, record_stamps, data_for_backward=True)

        kalman.smooth_rts(x, P, xa, Pa, Phi_arr)

        ind = np.searchsorted(stamps, record_stamps)
        x = x[ind]
        P = P[ind]

        err, sd, gyro_err, gyro_sd, accel_err, accel_sd = \
            _compute_output_errors(self.traj_ref, x, P, record_stamps,
                                   self.error_model, self.gyro_model,
                                   self.accel_model)

        traj_corr = correct_trajectory(traj, err)

        return FiltResult(traj=traj_corr, err=err, sd=sd, gyro_err=gyro_err,
                          gyro_sd=gyro_sd, accel_err=accel_err,
                          accel_sd=accel_sd, x=x, P=P, residuals=residuals)


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
                 error_model=error_models.ModifiedPhiModel(),
                 gyro_model=None, accel_model=None):
        if gyro_model is None:
            gyro_model = InertialSensor()
        if accel_model is None:
            accel_model = InertialSensor()

        n_states = error_model.N_STATES + gyro_model.n_states + \
                   accel_model.n_states
        n_noises = (gyro_model.n_noises + accel_model.n_noises +
                    3 * (gyro_model.noise is not None) +
                    3 * (accel_model.noise is not None))

        q = np.zeros(n_noises)
        P0 = np.zeros((n_states, n_states))

        n = error_model.N_STATES
        n1 = gyro_model.n_states
        n2 = accel_model.n_states

        level_sd = np.deg2rad(level_sd)
        azimuth_sd = np.deg2rad(azimuth_sd)

        P0_nav = np.zeros((error_model.N_STATES, error_model.N_STATES))
        P0_nav[error_model.DRE, error_model.DRE] = pos_sd ** 2
        P0_nav[error_model.DRN, error_model.DRN] = pos_sd ** 2
        P0_nav[error_model.DRU, error_model.DRU] = pos_sd ** 2
        P0_nav[error_model.DVE, error_model.DVE] = vel_sd ** 2
        P0_nav[error_model.DVN, error_model.DVN] = vel_sd ** 2
        P0_nav[error_model.DVU, error_model.DVU] = vel_sd ** 2
        P0_nav[error_model.DROLL, error_model.DROLL] = level_sd ** 2
        P0_nav[error_model.DPITCH, error_model.DPITCH] = level_sd ** 2
        P0_nav[error_model.DHEADING, error_model.DHEADING] = azimuth_sd ** 2

        P0[n: n + n1, n: n + n1] = gyro_model.P
        P0[n + n1: n + n1 + n2, n + n1: n + n1 + n2] = accel_model.P
        self.P0_nav = P0_nav
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

        states = error_model.STATES
        for name, state in gyro_model.states.items():
            states['GYRO_' + name] = n + state
        for name, state in accel_model.states.items():
            states['ACCEL_' + name] = n + n1 + state

        self.dt = dt
        self.n_states = n_states
        self.n_noises = n_noises
        self.states = states

        self.error_model = error_model
        self.gyro_model = gyro_model
        self.accel_model = accel_model

    def _validate_parameters(self, integrator, theta, dv, observations,
                             max_step, record_stamps, feedback_period):
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

        return theta, dv, observations, stamps, record_stamps, feedback_period

    def _forward_pass(self, integrator, theta, dv, observations, stamps,
                      record_stamps, feedback_period, data_for_backward=False):
        start = integrator.traj.index[0]

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
        T = np.linalg.inv(
            self.error_model.transform_to_output(integrator.get_state()))
        Pc[:self.error_model.N_STATES, :self.error_model.N_STATES] = \
            T @ self.P0_nav @ T.transpose()

        H_max = np.zeros((10, self.n_states))

        i_reading = 0  # Number of processed readings.
        i_stamp = 0  # Index of current stamp in stamps array.
        # Index of current position in x and P arrays for saving xc and Pc.
        i_save = 0

        n = self.error_model.N_STATES
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

        n_readings = theta.shape[0]
        while i_reading < n_readings:
            theta_b = theta[i_reading: i_reading + feedback_period]
            dv_b = dv[i_reading: i_reading + feedback_period]

            traj_b = integrator.integrate(theta_b, dv_b)
            Fi, Fig, Fia = self.error_model.system_matrix(traj_b)
            i = 0

            while i < theta_b.shape[0]:
                stamp = stamps[i_stamp]
                stamp_next = stamps[i_stamp + 1]
                delta_i = stamp_next - stamp
                i_next = i + delta_i

                if data_for_backward and record_stamps[i_save] == stamp:
                    xa[i_save] = xc
                    Pa[i_save] = Pc

                for i_obs, obs in enumerate(observations):
                    ret = obs.compute_obs(stamp, traj_b.iloc[i],
                                          self.error_model)
                    if ret is not None:
                        z, H, R = ret
                        H_max[:H.shape[0], :self.error_model.N_STATES] = H
                        res = kalman.correct(xc, Pc, z, H_max[:H.shape[0]], R)
                        obs_stamps[i_obs].append(stamp)
                        obs_residuals[i_obs].append(res)

                if record_stamps[i_save] == stamp:
                    x[i_save] = xc
                    P[i_save] = Pc
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

                if data_for_backward:
                    Phi_arr[i_save - 1] = Phi

                i = i_next
                i_stamp += 1

            i_reading += feedback_period

            integrator.set_state(self.error_model.correct_state(
                integrator.get_state(), xc))

            xc[:self.error_model.N_STATES] = 0

        if record_stamps[i_save] == stamps[i_stamp]:
            x[i_save] = xc
            P[i_save] = Pc

            if data_for_backward:
                xa[i_save] = xc
                Pa[i_save] = Pc

        residuals = []
        for s, r in zip(obs_stamps, obs_residuals):
            residuals.append(pd.DataFrame(index=s, data=np.asarray(r)))

        return x, P, xa, Pa, Phi_arr, residuals

    def run(self, integrator, theta, dv, observations=[], max_step=1,
            feedback_period=500, record_stamps=None):
        """Run the filter.

        Parameters
        ----------
        integrator : `pyins.integrate.Integrator` instance
            Integrator to use for INS state propagation. It will be reset
            before the filter start.
        theta, dv : ndarray, shape (n_readings, 3)
            Rotation vectors and velocity increments computed from gyro and
            accelerometer readings after applying coning and sculling
            corrections.
        observations : list of `Observation`
            Measurements which will be processed. Empty by default.
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
        sd : DataFrame
            Estimated standard deviations of trajectory errors.
        gyro_err, gyro_sd : DataFrame
            Estimated gyro error states and their standard deviations.
        accel_err, accel_sd : DataFrame
            Estimated accelerometer error states and their standard deviations.
        P : ndarray, shape (n_points, n_states, n_states)
            History of the filter covariance.
        residuals : list of DataFrame
            Each DataFrame corresponds to an observation from `observations`.
            Its index is observation time stamps and columns contain normalized
            observations residuals for each component of the observation
            vector `z`.

        Notes
        -----
        Estimated trajectory errors and a history of the filter states are not
        returned because they are computed relative to partially corrected
        trajectory and are not useful for interpretation.
        """
        theta, dv, observations, stamps, record_stamps, feedback_period = \
            self._validate_parameters(integrator, theta, dv, observations,
                                      max_step, record_stamps, feedback_period)

        x, P, _, _, _, residuals = self._forward_pass(integrator, theta, dv,
                                                      observations, stamps,
                                                      record_stamps,
                                                      feedback_period)

        traj = integrator.traj.loc[record_stamps]
        err, sd, gyro_err, gyro_sd, accel_err, accel_sd = \
            _compute_output_errors(traj, x, P, record_stamps, self.error_model,
                                   self.gyro_model, self.accel_model)

        traj_corr = correct_trajectory(integrator.traj, err)

        return FiltResult(traj=traj_corr, sd=sd, gyro_err=gyro_err,
                          gyro_sd=gyro_sd, accel_err=accel_err,
                          accel_sd=accel_sd, P=P, residuals=residuals)

    def run_smoother(self, integrator, theta, dv, observations=[], max_step=1,
                     feedback_period=500, record_stamps=None):
        """Run the smoother.

        It means that observations during the whole time is used to estimate
        the errors at each moment of time (i.e. it is not real time). The
        Rauch-Tung-Striebel two pass recursion is used [1]_.

        Parameters
        ----------
        integrator : `pyins.integrate.Integrator` instance
            Integrator to use for INS state propagation. It will be reset
            before the filter start.
        theta, dv : ndarray, shape (n_readings, 3)
            Rotation vectors and velocity increments computed from gyro and
            accelerometer readings after applying coning and sculling
            corrections.
        observations : list of `Observation`
            Measurements which will be processed. Empty by default.
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
        sd : DataFrame
            Estimated trajectory errors and their standard deviations.
        gyro_err, gyro_sd : DataFrame
            Estimated gyro error states and their standard deviations.
        accel_err, accel_sd : DataFrame
            Estimated accelerometer error states and their standard deviations.
        P : ndarray, shape (n_points, n_states, n_states)
            History of the filter covariance.
        residuals : list of DataFrame
            Each DataFrame corresponds to an observation from `observations`.
            Its index is observation time stamps and columns contain normalized
            observations residuals for each component of the observation
            vector `z`.

        Notes
        -----
        Estimated trajectory errors and a history of the filter states are not
        returned because they are computed relative to partially corrected
        trajectory and are not useful for interpretation.
        """
        theta, dv, observations, stamps, record_stamps, feedback_period = \
            self._validate_parameters(integrator, theta, dv, observations,
                                      max_step, record_stamps, feedback_period)

        x, P, xa, Pa, Phi_arr, residuals = self._forward_pass(
            integrator, theta, dv, observations, stamps, record_stamps,
            feedback_period, data_for_backward=True)

        traj = integrator.traj.loc[record_stamps]
        err, sd, gyro_err, gyro_sd, accel_err, accel_sd = \
            _compute_output_errors(traj, x, P, record_stamps, self.error_model,
                                   self.gyro_model, self.accel_model)

        traj = correct_trajectory(traj, err)
        xa[:, :self.error_model.N_STATES] -= x[:, :self.error_model.N_STATES]
        x[:, :self.error_model.N_STATES] = 0

        kalman.smooth_rts(x, P, xa, Pa, Phi_arr)

        ind = np.searchsorted(stamps, record_stamps)
        x = x[ind]
        P = P[ind]
        traj = traj.iloc[ind]

        err, sd, gyro_err, gyro_sd, accel_err, accel_sd = \
            _compute_output_errors(traj, x, P, record_stamps[ind],
                                   self.error_model, self.gyro_model,
                                   self.accel_model)

        traj = correct_trajectory(traj, err)

        return FiltResult(traj=traj, sd=sd, gyro_err=gyro_err,
                          gyro_sd=gyro_sd, accel_err=accel_err,
                          accel_sd=accel_sd, P=P, residuals=residuals)
