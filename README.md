![tests status](https://github.com/nmayorov/pyins/actions/workflows/build_and_test.yaml/badge.svg)
![documentation status](https://readthedocs.org/projects/pyins/badge/?version=latest)

# pyins

pyins is a Python package which provides common basic algorithms used in Inertial Navigation Systems (INS) aided by external sensors like GNSS.

Version 1.0 has refined and improved functionality, consistent docstrings and extended examples.
Refer to [version_1.0.md](./version_1.0.md) for the list of main changes compared to earlier versions.

## Installation

To install the package to the user directory execute in the package directory:
```shell
pip install . --user
```
To perform editable (inplace) install:
```shell
pip install -e .
```

Installation requires building Cython extension for which you need Cython, scipy and a C compiler.
On Linux and Mac it works seamlessly with system compilers (gcc or clang), for building on Windows refer to https://docs.python.org/3/extending/windows.html#building-on-windows.
Add option ``--no-build-isolation`` if you want to use already installed Cython and scipy during the installation.

Runtime dependencies include (versions in parentheses were used for the latest development):

* numpy (1.24.3)
* scipy (1.11.3)
* pandas (2.0.3)
 
## Running tests

To run all the supplied tests with `pytest` execute: 
```shell
pytest pyins
```
Due to some `pytest` limitations it won't work for editable installation. 
In this case supply path to `tests` directory:
```shell
pytest /your_dev_dir/pyins/pyins/tests
```

## Documentation

Documentation is available here https://pyins.readthedocs.io.
