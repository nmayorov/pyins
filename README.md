![tests status](https://github.com/nmayorov/pyins/actions/workflows/build_and_test.yaml/badge.svg)
![documentation status](https://readthedocs.org/projects/pyins/badge/?version=latest)

# pyins

pyins is a Python package which provides common basic algorithms used in Inertial Navigation Systems (INS) aided by external sensors like GNSS.

Version 1.0 has refined and improved functionality, consistent docstrings and extended examples.
Refer to [version_1.0.md](./version_1.0.md) for the list of main changes compared to earlier versions.

## Installation

The package now is pure Python and thus can be easily installed on any platform.

To install execute in the package directory: 
```shell
pip install .
```
To perform editable (inplace) install:
```shell
pip install -e .
```

## Dependencies

Runtime dependencies include (versions in parentheses were used for the latest development):

* numba (0.58.0)
* numpy (1.25.2)
* scipy (1.11.3)
* pandas (2.1.1)

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
