PyINS
=====

PyINS is a Python package for data processing commonly done in Strapdown
Inertial Navigation Systems integrated with other aiding sensors.

Existing MATLAB codes for INS modeling and analysis are either proprietary or
unstructured and poorly documented. This package is intended to be open-source
and follow established conventions in scientific Python, including code quality,
documentation and testing. It is hoped that PyINS will offer an educational as
well as a practical value.

The main features currently available:
    
    - State of the art algorithm for integrating inertial readings.
    - Linear model for INS errors propagation.
    - Feedforward and feedback Kalman filters on top of the INS error model. 
      Any number of external aiding sources can be considered and users
      can easily define their own observation models.
    - Convenient class for representing errors of an Inertial Measurement Unit 
      in the Kalman filter. It includes bias, scale factor, random walk,
      white and time correlated noises.
    - Simulator of strapdown inertial readings given a vehicle trajectory. 
      The algorithm doesn't rely on reversing the strapdown integration
      algorithm, but provides accurate and realistic measurements nevertheless.

It is highly recommended to start with reading :ref:`design`.

.. toctree::
   :caption: Context
   :maxdepth: 1

   design
   tutorial
   examples
   api
   todo
