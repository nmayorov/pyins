pyins
=====

pyins is a Python package for data processing commonly done in Strapdown
Inertial Navigation Systems integrated with other aiding sensors.


Installation
************

Dependencies
------------

The package is developed for Python 3 only. It has the following dependencies:

1. numpy
2. scipy >= 0.18
3. pandas
4. Cython >= 0.22

Installation
------------

Currenly you need to install from source, you will need a C compiler suiable
for your version of Python. To install::

    python setup.py install

If you want to use pyins from its source directory without installing, then
you need to compile extensions::

    python setup.py build_ext -i

Then append the path to the directory to PYTHONPATH.

Documentation
*************

Documentation is available `here <pyins.readthedocs.io>`_.
