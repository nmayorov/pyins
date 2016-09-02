.. _design:

Design Principles and Conventions
==================================

The package is aimed to allow convenient computations according to a classic
aided INS scheme: 

1. Inertial readings are integrated with a high frequency.
2. INS errors are propagated in a Kalman filter in parallel.
3. External measurements are used to estimate the errors.
4. The INS output is corrected by the estimated errors. Optionally the
   errors can be fed back into the INS periodically to keep them small and
   establish the error equations linearity.

The idea is to focus on this simple formulation, but to make it flexible and 
easy to use.

To allow modeling without data from actual sensors the package contains
simulation routines. But currently this part is not developed enough.

Conventions
-----------
The following conventions were adopted:

1. The variables describing a vehicle trajectory are chosen to be:

    - latitude and longitude for a position
    - East and North velocity components
    - heading, pitch and roll for an attitude  

   Note that a vertical velocity and an altitude are not considered. The are two
   reasons for that. On the one hand, the vertical channel is exponentially
   divergent without aiding, thus an aiding source for an altitude is absolutely
   necessary if we want to compute it. On the other hand, it doesn't
   significantly influence horizontal variables. It all means that including
   vertical channel would complicate the package design without strong benefits.
   Later the vertical channel computations might be added to the package.

2. Latitude and longitude as well as attitude angles (heading, pitch and roll)
   are measured in *degrees*. *All* other quantities are measured in
   International System of Units.

3. The INS errors are computed for the same variables of which a trajectory is
   composed. The only subtlety is that latitude and longitude errors are
   computed in meters for an ease of interpretation.

4. Trajectory and its errors are stored in pandas DataFrame. The index of this
   DataFrame is integer time stamps at which variables are measured. The step
   of one integer time stamp corresponds to the sampling period of inertial
   sensors. The DataFrame contains the following columns (the meaning 
   is clear): lat, lon, VE, VN, h, p, r.

5. Measurements by external sensors are also stored in pandas DataFrame with an
   index being integer time stamps at which measurements are taken. Using time
   stamps for an index provides convenient data synchronization --- obviously
   important thing in aided INS.

6. The local-level North-pointing frame (namely ENU frame) is used for velocity
   integration as opposed to the wander azimuth frame. While the wander azimuth
   mechanization is more robust and recommended for an on-board INS
   implementation, it brings more complications than advantages for our
   simulation and postprocessing package.
