"""Monte-Carlo waveform simulation (0–20 PE re-scope, 2026-07-28).

``pmtdeconv.sim.tang2024`` reimplements the simulation of Tang et al.
2411.07620 ("Investigation and optimization of the deconvolution method for
PMT waveform reconstruction", JINST) — the paper that established the 0–20 PE
applicability domain. Its per-hit SPE shape fluctuation gives every hit a
slightly different kernel than the calibrated template the deconvolution
divides by: kernel mismatch is BUILT IN, so unlike the regime fixtures this
family is not an inverse crime.
"""
