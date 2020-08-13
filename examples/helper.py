"""Helper Functions"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy import interpolate


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_nb(Cnb, plotNframe=True):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.grid(False)
    lim = [-0.75, 0.75]
    ax.set_xlim3d(lim)
    ax.set_xticks([])
    ax.set_ylim3d(lim)
    ax.set_yticks([])
    ax.set_zlim3d(lim)
    ax.set_zticks([])
    plt.axis('off')

    # plot the navigation frame N in black
    if plotNframe:
        z = [0, 0]
        xn = Arrow3D([0, 1], z, z, mutation_scale=20,
                     lw=1, arrowstyle="-|>", color='k')
        ax.add_artist(xn)
        yn = Arrow3D(z, [0, 1], z, mutation_scale=20,
                     lw=1, arrowstyle="-|>", color='k')
        ax.add_artist(yn)
        zn = Arrow3D(z, z, [0, 1], mutation_scale=20,
                     lw=1, arrowstyle="-|>", color='k')
        ax.add_artist(zn)

    # plot the body frame B
    xb = Arrow3D([0, Cnb[0, 0]], [0, Cnb[1, 0]], [0, Cnb[2, 0]], mutation_scale=20,
                 lw=1, arrowstyle="-|>", color='b', label='$x_b$')
    ax.add_artist(xb)
    yb = Arrow3D([0, Cnb[0, 1]], [0, Cnb[1, 1]], [0, Cnb[2, 1]], mutation_scale=20,
                 lw=1, arrowstyle="-|>", color='orange', label='$y_b$')
    ax.add_artist(yb)
    zb = Arrow3D([0, Cnb[0, 2]], [0, Cnb[1, 2]], [0, Cnb[2, 2]], mutation_scale=20,
                 lw=1, arrowstyle="-|>", color='g', label='$z_b$')
    ax.add_artist(zb)

    plt.legend(handles=[xb, yb, zb])
    plt.show()


def plot_inertial_readings(dt, gyros, accels, step=1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    n = accels.shape[0]
    t = dt * np.arange(n)

    ax1.plot(t[::step], accels[::step])
    ax1.legend(['X', 'Y', 'Z'], loc='center right')
    ax1.set_xlabel('time, s')
    ax1.set_title('accels readings')
    ax1.set_ylabel('m/s')

    ax2.plot(t[::step], gyros[::step])
    ax2.legend(['X', 'Y', 'Z'], loc='center right')
    ax2.set_xlabel('time, s')
    ax2.set_title('gyros readings')
    ax2.set_ylabel('rad')

    plt.show()


def plot_traj(dt, traj, step=1, coord_unit="deg", time_unit="s"):

    traj = traj.iloc[::step]
    xlabel = "time, " + time_unit

    if time_unit == "s":
        t = traj.index * dt
    elif time_unit == "h":
        t = traj.index * dt / 3600
    else:
        raise ValueError("Time unit must be s (second) or h (heure)")

    plt.figure(figsize=(15, 10))

    plt.subplot(331)
    plt.plot(t, traj.lat, label='lat')
    plt.xlabel(xlabel)
    plt.ylabel(coord_unit)
    plt.legend(loc='best')

    plt.subplot(334)
    plt.plot(t, traj.lon, label='lon')
    plt.xlabel(xlabel)
    plt.ylabel(coord_unit)
    plt.legend(loc='best')

    if 'alt' in traj:
        plt.subplot(337)
        plt.plot(t, traj.alt, label='alt')
        plt.xlabel(xlabel)
        plt.ylabel("m")
        plt.legend(loc='best')

    plt.subplot(332)
    plt.plot(t, traj.VE, label='VE')
    plt.xlabel(xlabel)
    plt.ylabel("m/s")
    plt.legend(loc='best')

    plt.subplot(335)
    plt.plot(t, traj.VN, label='VN')
    plt.xlabel(xlabel)
    plt.ylabel("m/s")
    plt.legend(loc='best')

    if 'VU' in traj:
        plt.subplot(338)
        plt.plot(t, traj.VU, label='VU')
        plt.xlabel(xlabel)
        plt.ylabel("m/s")
        plt.legend(loc='best')

    plt.subplot(333)
    plt.plot(t, traj.h, label='heading')
    plt.xlabel(xlabel)
    plt.ylabel("deg")
    plt.legend(loc='best')

    plt.subplot(336)
    plt.plot(t, traj.p, label='pitch')
    plt.xlabel(xlabel)
    plt.ylabel("deg")
    plt.legend(loc='best')

    plt.subplot(339)
    plt.plot(t, traj.r, label='roll')
    plt.xlabel(xlabel)
    plt.ylabel("deg")
    plt.legend(loc='best')

    plt.tight_layout()


def generate_WP(n_move, step, angle_spread, random_state=0):

    rng = np.random.RandomState(random_state)
    angle_spread = np.deg2rad(angle_spread)
    angle = rng.uniform(2 * np.pi)
    WP = [np.zeros(2)]

    f, ax = plt.subplots()

    for i in range(n_move):
        ax.quiver(WP[i][0], WP[i][1], np.cos(angle), np.sin(angle), color='k')
        ax.scatter(WP[i][0], WP[i][1], color='k')

        WP.append(WP[-1] + step * np.array([np.cos(angle), np.sin(angle)]))
        angle += rng.uniform(-angle_spread, angle_spread)

    ax.scatter(WP[-1][0], WP[-1][1], color='k')
    WP = np.asarray(WP)

    return WP, ax


def generate_trajectory(n_samples, WP):
    tck, u = interpolate.splprep([WP[:, 0], WP[:, 1]], s=0)
    x, y = interpolate.splev(np.linspace(0, 1, n_samples), tck)

    h = np.unwrap(np.pi / 2 - np.arctan2(np.diff(y), np.diff(x)))

    r = 200 * np.diff(h)
    h = np.hstack((h, h[-1]))
    r = np.hstack((r, r[-1], r[-1]))

    return x, y, np.rad2deg(h), np.rad2deg(r)
