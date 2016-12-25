from distutils.core import setup
from Cython.Build import cythonize


setup_options = dict(
    name="pyins",
    version="0.1",
    description="Inertial navigation system toolkit",
    maintainer="Nikolay Mayorov",
    maintainer_email="nikolay.mayorov@zoho.com",
    license="MIT",
    packages=["pyins"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    requires=["numpy", "scipy", "pandas", "Cython"],
    ext_modules=cythonize(["pyins/_integrate.pyx",
                           "pyins/_dcm_spline_solver.pyx"])
)

setup(**setup_options)
