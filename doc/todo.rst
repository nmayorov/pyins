To Do List
==========

The following features are considered important, but not yet implemented:

1. Optimal trajectory smoothing on the interval in 
   :class:`pyins.filt.FeedforwardFilter` and :class:`pyins.filt.FeedbackFilter` 
   (as opposed to real time filtering).

2. Support for error states in observation models in 
   :class:`pyins.filt.Observation`.

3. Improvements to simulation routines in :mod:`pyins.sim`:
   
   - Figure out logical and convenient way to specify a vehichle trajectory for
     strapdown sensors simulation (most likely several approaches are
     necessary).
   - Ability to conveniently apply errors to generated ideal readings,
     including advanced effects as quantization.
   - Realistic simulation of other sesnors, the first candidate being GPS 
     receiver.
