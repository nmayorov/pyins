=====
pyins
=====

pyins is a Python package which provides common basic algorithms used in Inertial Navigation Systems (INS) aided by external sensors like GNSS.
The package has rather limited scope and purposes, it may serve as:

- Reasonably well written and tested code of basic algorithms in aided INS for people studying the subject
- Modelling toolkit to verify achievable performance depending on IMU specification, test observability properties, etc.
- For *basic* processing of real sensor data in case there are no better alternatives
- As a starting reference point for developing practical aided INS algorithms

The package is not meant to achieve robust state of the art performance on real sensor data.
Also it is not planned to further extend its functionality besides what's currently implemented.

The following core features are available:

- Synthesis of incremental and rate IMU sensors for modelling purposes
- Strapdown INS integration algorithm
- INS error model
- Feedforward and feedback INS Kalman filters with customizable IMU model and set of measurements
- Trajectory interpolation, smoothing and comparison functions

Tutorial
========

There is no extensive dedicated tutorial for the package.
The :doc:`api` describes available modules with some additional context information.
The examples of usage in jupyter notebooks are available in :doc:`examples`.

The following books are recommended to understand pyins functionality and the field in general:

- P\. D\. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems" 2nd edition
- P\. G\. Savage, "Strapdown analytics"
- P\. S\. Maybeck, "Stochastic Models, Estimation and Control", volumes 1 and 2

.. toctree::
    :caption: Context
    :maxdepth: 1

    api
    examples
