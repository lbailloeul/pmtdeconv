"""DAQ -> ROOT data-prep for the PMT deconvolution pipeline.

A SEPARATE package from ``pmtdeconv`` because it depends on PyROOT
(``import ROOT``), which is not pip-installable. It owns the two pre-deconv
steps:

  1. DRS binary -> ``Run<N>.root``  (mcitron's drsProcessing, wrapped via subprocess)
  2. ``Run<N>.root`` -> ``flattened_Run<N>.root``  (``flatten_waveforms``)

Both outputs land side-by-side under one data dir — the layout ``pmtdeconv``
expects (``<data-root>/<dataset>/`` holding both ``Run<N>.root`` and
``flattened_Run<N>.root``). See ``README.md``.

Imports are kept lazy (no eager ``import ROOT`` here) so ``dataprep --help``
works without ROOT installed.
"""

__version__ = "0.1.0"
