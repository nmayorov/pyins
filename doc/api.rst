API Reference
=============

Direction Cosine Matrices
-------------------------

.. automodule:: pyins.dcm
    :no-members:
    :no-inherited-members:

.. currentmodule:: pyins

.. autosummary::
    :toctree: generated/

    dcm.from_rv
    dcm.to_rv
    dcm.from_hpr
    dcm.to_hpr
    dcm.from_llw
    dcm.to_llw


Earth Surface and Gravity Models
--------------------------------

.. automodule:: pyins.earth
    :no-members:
    :no-inherited-members:

.. currentmodule:: pyins

.. autosummary::
    earth.RATE
    earth.SF
    earth.G0
    earth.R0

.. autosummary::
    :toctree: generated/
            
    earth.set_model
    earth.principal_radii
    earth.gravity
    earth.gravitation_ecef


Coordinate Transformations
--------------------------

.. automodule:: pyins.coord
    :no-members:
    :no-inherited-members:

.. currentmodule:: pyins

.. autosummary::
    :toctree: generated/

    coord.lla_to_ecef
    coord.perturb_ll


INS Self Alignment
------------------

.. automodule:: pyins.align
    :no-members:
    :no-inherited-members:

.. currentmodule:: pyins

.. autosummary::
    :toctree: generated/

        align.align_wahba



Strapdown Integration
---------------------

.. automodule:: pyins.integrate
    :no-members:
    :no-inherited-members:

.. currentmodule:: pyins

.. autosummary::
    :toctree: generated/

    integrate.coning_sculling
    integrate.integrate
    integrate.Integrator


Navigation Kalman Filters
-------------------------

.. automodule:: pyins.filt
    :no-members:
    :no-inherited-members:

.. currentmodule:: pyins

.. autosummary::
    :toctree: generated/
    
    filt.InertialSensor
    filt.Observation
    filt.LatLonObs
    filt.VeVnObs
    filt.FeedforwardFilter
    filt.FeedbackFilter


Strapdown Sensors Simulator
------------------------

.. automodule:: pyins.sim
    :no-members:
    :no-inherited-members:

.. currentmodule:: pyins

.. autosummary::
    :toctree: generated/

    sim.from_position


Utility Functions
-----------------

.. automodule:: pyins.util
    :no-members:
    :no-inherited-members:

.. currentmodule:: pyins

.. autosummary::
    :toctree: generated/
    
    util.mm_prod
    util.mv_prod
