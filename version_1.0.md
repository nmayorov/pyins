# Changes for version 1.0

1. Code related to "smoothing" is removed.
   The implemented "EKF smoother" is a half-baked approach which doesn't correspond to a well define optimal algorithm.
2. Filter classes were replaced by more straightforward plain functions.
3. All data is now indexed by time in seconds, not IMU stamp. 
   This is a more intuitive and convenient approach which directly corresponds to practical algorithms.
   This also allows to more optimally process measurements not precisely synchronous with IMU samples, which again corresponds to real-world data.
4. All data now is stored in pandas DataFrame with consistent column naming.
5. Transform utilities were extended and improved.
6. Everything is designed to work better together.
7. Better test coverage.
8. Cython was replaced by numba to avoid the need of compilation.
9. HTML documentation and examples are reworked and cleaned up.
