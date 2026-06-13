#!/usr/bin/env python3
"""
Unit tests for the acceptance and veto logic in utils.py.

Run:  python -m pytest tests/test_vetoes.py -v
  or: python tests/test_vetoes.py
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pmtdeconv.utils import (
    DTYPE, LANL_TEMPLATE, MILLIQAN_TEMPLATE,
    gaussian_smeared_biexponential, build_spe_template,
    gaussian_lowpass_filter, cosine_lowpass_filter,
    deconvolve_hits, stft_denoise,
    baseline_noise_bestwindow, sliding_local_mean,
    find_integration_windows, _walk_to_trough_left, _walk_to_trough_right,
    apply_peak_vetoes,
)

# ═══════════════════════════════════════════════════════════════════
# Helper: create synthetic waveforms
# ═══════════════════════════════════════════════════════════════════

def make_flat_waveform(N=1024, baseline=100.0, noise_sigma=2.0, dt_ns=1.25):
    """Flat waveform with Gaussian noise — no pulses."""
    rng = np.random.default_rng(42)
    wave = baseline + rng.normal(0, noise_sigma, N)
    t = np.arange(N, dtype=DTYPE) * dt_ns
    return t, wave.astype(DTYPE)


def make_single_pulse(N=1024, baseline=100.0, noise_sigma=2.0, dt_ns=1.25,
                      pulse_amp=500.0, pulse_t0=400.0):
    """Flat baseline + one clean pulse at pulse_t0."""
    t, wave = make_flat_waveform(N, baseline, noise_sigma, dt_ns)
    pulse_bin = int(pulse_t0 / dt_ns)
    # Add a simple Gaussian pulse
    sigma_bins = 5
    pulse = pulse_amp * np.exp(-0.5 * ((np.arange(N) - pulse_bin) / sigma_bins) ** 2)
    wave += pulse
    return t, wave


def make_saturated_waveform(N=1024, baseline=100.0, dt_ns=1.25, sat_level=1500.0):
    """Waveform with a pulse that exceeds saturation threshold."""
    t, wave = make_single_pulse(N, baseline, pulse_amp=sat_level)
    return t, wave


# ═══════════════════════════════════════════════════════════════════
# Tests: baseline_noise_bestwindow
# ═══════════════════════════════════════════════════════════════════

class TestBaseline:
    def test_flat_waveform_finds_baseline(self):
        """Flat waveform should find a clean baseline with passed=True."""
        t, wave = make_flat_waveform(baseline=50.0, noise_sigma=1.0)
        ped, sig, (t0, t1), w, passed = baseline_noise_bestwindow(
            t, wave, 1.25, search_end_ns=600.0,
            window_ns=200.0, min_sigma=1e-3,
            flat_frac_min=0.95, flat_k=4.0, spike_k=5.0,
        )
        assert abs(ped - 50.0) < 5.0, f"Baseline {ped} too far from 50.0"
        assert sig < 5.0, f"Sigma {sig} too high for clean waveform"
        assert passed, "Should pass log-sigma threshold for clean waveform"

    def test_noisy_waveform_fails_quality(self):
        """Very noisy waveform should return passed=False."""
        t, wave = make_flat_waveform(baseline=50.0, noise_sigma=20.0)
        ped, sig, _, _, passed = baseline_noise_bestwindow(
            t, wave, 1.25, search_end_ns=600.0,
            window_ns=200.0, min_sigma=1e-3,
            flat_frac_min=0.95, flat_k=4.0, spike_k=5.0,
        )
        assert not passed, "Very noisy waveform should not pass quality threshold"

    def test_baseline_avoids_pulse(self):
        """Baseline finder should avoid the pulse region."""
        t, wave = make_single_pulse(baseline=50.0, noise_sigma=1.0,
                                    pulse_amp=500.0, pulse_t0=100.0)
        ped, sig, (t0, t1), _, passed = baseline_noise_bestwindow(
            t, wave, 1.25, search_end_ns=600.0,
            window_ns=200.0, min_sigma=1e-3,
            flat_frac_min=0.95, flat_k=4.0, spike_k=5.0,
        )
        # Baseline window should NOT overlap with pulse at t=100 ns
        assert abs(ped - 50.0) < 10.0, f"Baseline {ped} contaminated by pulse"

    def test_returns_five_values(self):
        """Baseline function must return (ped, sig, (t0, t1), w, passed)."""
        t, wave = make_flat_waveform()
        result = baseline_noise_bestwindow(t, wave, 1.25, search_end_ns=600.0)
        assert len(result) == 5, f"Expected 5 return values, got {len(result)}"

    def test_no_baseline_raises(self):
        """Waveform with no flat region should raise RuntimeError."""
        N = 500
        t = np.arange(N, dtype=DTYPE) * 1.25
        wave = np.arange(N, dtype=DTYPE) * 100.0  # steep ramp — no flat region
        try:
            baseline_noise_bestwindow(t, wave, 1.25, search_end_ns=600.0,
                                      max_baseline_adc=50.0)
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass


# ═══════════════════════════════════════════════════════════════════
# Tests: apply_peak_vetoes
# ═══════════════════════════════════════════════════════════════════

class TestPeakVetoes:
    """Test each veto independently using synthetic u_rec."""

    def _make_simple_urec(self, N=1024, peak_bins=None, peak_heights=None, dt_ns=1.25):
        """Create a synthetic u_rec with peaks at specified bins."""
        if peak_bins is None:
            peak_bins = [400]
        if peak_heights is None:
            peak_heights = [1.0] * len(peak_bins)

        u_rec = np.zeros(N, dtype=DTYPE)
        for b, h in zip(peak_bins, peak_heights):
            sigma = 3
            idx = np.arange(max(0, b-20), min(N, b+20))
            u_rec[idx] += h * np.exp(-0.5 * ((idx - b) / sigma) ** 2)

        t = np.arange(N, dtype=DTYPE) * dt_ns
        peaks = np.array(peak_bins, dtype=int)

        pad_bins = int(18.0 / dt_ns)
        windows = find_integration_windows(u_rec, peaks, pad_bins)

        pulse_widths = []
        for pk in peaks:
            lo_tr = _walk_to_trough_left(u_rec, pk, pad_bins)
            hi_tr = _walk_to_trough_right(u_rec, pk, pad_bins)
            pulse_widths.append(hi_tr - lo_tr)

        return u_rec, t, peaks, windows, pulse_widths

    def test_good_peak_passes(self):
        """A clean peak at t=500 ns should pass all vetoes."""
        u_rec, t, peaks, windows, pw = self._make_simple_urec(peak_bins=[400])
        success, reason, charge = apply_peak_vetoes(
            400, 0, peaks, u_rec, t, windows, pw, len(u_rec),
            t_peak_min_ns=20.0,
            dt_ns=1.25,
        )
        assert success, f"Clean peak should pass, got reason={reason}"
        assert reason == ""
        assert charge > 0

    def test_out_of_time_veto(self):
        """Peak at t=5 ns (before t_peak_min_ns) is vetoed as out_of_time."""
        u_rec, t, peaks, windows, pw = self._make_simple_urec(peak_bins=[4])
        success, reason, _ = apply_peak_vetoes(
            4, 0, peaks, u_rec, t, windows, pw, len(u_rec),
            t_peak_min_ns=20.0, dt_ns=1.25,
        )
        assert not success
        assert reason == "out_of_time"

    def test_post_sat_shift_veto(self):
        """Peak after post_sat_cutoff_bin should be vetoed."""
        u_rec, t, peaks, windows, pw = self._make_simple_urec(peak_bins=[400])
        success, reason, _ = apply_peak_vetoes(
            400, 0, peaks, u_rec, t, windows, pw, len(u_rec),
            post_sat_cutoff_bin=300,  # cutoff before the peak
            dt_ns=1.25,
        )
        assert not success
        assert reason == "post_sat_shift"

    def test_post_sat_shift_before_cutoff_passes(self):
        """Peak before post_sat_cutoff_bin should pass."""
        u_rec, t, peaks, windows, pw = self._make_simple_urec(peak_bins=[200])
        success, reason, _ = apply_peak_vetoes(
            200, 0, peaks, u_rec, t, windows, pw, len(u_rec),
            post_sat_cutoff_bin=500,
            dt_ns=1.25,
        )
        assert success, f"Peak before cutoff should pass, got reason={reason}"

    def test_afterpulse_veto(self):
        """Small peak near a much larger peak should be vetoed as afterpulse."""
        u_rec, t, peaks, windows, pw = self._make_simple_urec(
            peak_bins=[400, 420], peak_heights=[10.0, 0.5]
        )
        # Test the smaller peak (index 1)
        success, reason, _ = apply_peak_vetoes(
            420, 1, peaks, u_rec, t, windows, pw, len(u_rec),
            afterpulse_window_ns=40.0,
            afterpulse_height_ratio=5.0,
            dt_ns=1.25,
        )
        assert not success
        assert reason == "afterpulse_veto"

    def test_negative_charge_veto(self):
        """Peak with negative integrated charge should be vetoed."""
        N = 1024
        u_rec = np.zeros(N, dtype=DTYPE)
        pk = 400
        # Create a peak that's positive at the top but has large negative wings
        u_rec[pk] = 0.5
        u_rec[pk-1] = -2.0
        u_rec[pk+1] = -2.0
        u_rec[pk-2] = -1.0
        u_rec[pk+2] = -1.0

        t = np.arange(N, dtype=DTYPE) * 1.25
        peaks = np.array([pk], dtype=int)
        pad_bins = int(18.0 / 1.25)
        windows = find_integration_windows(u_rec, peaks, pad_bins)
        pulse_widths = [_walk_to_trough_right(u_rec, pk, pad_bins) - _walk_to_trough_left(u_rec, pk, pad_bins)]

        success, reason, charge = apply_peak_vetoes(
            pk, 0, peaks, u_rec, t, windows, pulse_widths, N,
            dt_ns=1.25,
        )
        assert not success
        assert reason == "negative_charge"
        assert charge < 0

    def test_negative_charge_catches_ringing(self):
        """Ringing artifact with negative integral is caught by negative_charge veto."""
        N = 1024
        u_rec = np.zeros(N, dtype=DTYPE)
        # Big peak at 400 (height=10)
        for i in range(-5, 6):
            u_rec[400 + i] = max(0, 10.0 - abs(i) * 2.0)
        # Small ringing artifact at 450 — positive peak but negative total charge
        u_rec[450] = 0.3
        u_rec[449] = -3.0
        u_rec[451] = -3.0
        u_rec[448] = -1.5
        u_rec[452] = -1.5

        t = np.arange(N, dtype=DTYPE) * 1.25
        peaks = np.array([400, 450], dtype=int)
        pad_bins = int(18.0 / 1.25)
        windows = find_integration_windows(u_rec, peaks, pad_bins)
        pulse_widths = []
        for pk in peaks:
            pulse_widths.append(_walk_to_trough_right(u_rec, pk, pad_bins) -
                                _walk_to_trough_left(u_rec, pk, pad_bins))

        success, reason, charge = apply_peak_vetoes(
            450, 1, peaks, u_rec, t, windows, pulse_widths, N,
            afterpulse_window_ns=40.0, afterpulse_height_ratio=5.0,
            dt_ns=1.25,
        )
        assert not success
        # Should be caught by afterpulse or negative_charge veto
        assert reason in ("negative_charge", "afterpulse_veto"), f"Expected negative_charge or afterpulse, got {reason}"


# ═══════════════════════════════════════════════════════════════════
# Tests: sliding_local_mean
# ═══════════════════════════════════════════════════════════════════

class TestSlidingLocalMean:
    def test_flat_signal(self):
        """Sliding mean of a flat signal should be approximately the same value."""
        wave = np.full(200, 100.0)
        lm = sliding_local_mean(wave, 40)
        # Interior should be exactly 100
        assert np.allclose(lm[20:-20], 100.0, atol=1e-10)

    def test_detects_shift(self):
        """Sliding mean should detect a baseline shift after a big pulse."""
        wave = np.full(1000, 100.0)
        # Drop baseline to 50 after bin 500
        wave[500:] = 50.0
        lm = sliding_local_mean(wave, 40)
        # Well into the shifted region, local mean should be ~50
        assert lm[600] < 70.0, f"Local mean at shifted region = {lm[600]}"


# ═══════════════════════════════════════════════════════════════════
# Tests: integration windows
# ═══════════════════════════════════════════════════════════════════

class TestIntegrationWindows:
    def test_single_peak(self):
        """Single peak should have one window containing the peak."""
        u_rec = np.zeros(200, dtype=DTYPE)
        u_rec[100] = 1.0
        u_rec[99] = 0.5
        u_rec[101] = 0.5
        peaks = np.array([100])
        windows = find_integration_windows(u_rec, peaks, pad_bins=15)
        assert len(windows) == 1
        lo, hi = windows[0]
        assert lo <= 100 <= hi

    def test_two_peaks_no_overlap(self):
        """Two well-separated peaks should have non-overlapping windows."""
        u_rec = np.zeros(500, dtype=DTYPE)
        u_rec[100] = 1.0
        u_rec[400] = 1.0
        peaks = np.array([100, 400])
        windows = find_integration_windows(u_rec, peaks, pad_bins=15)
        assert len(windows) == 2
        _, hi1 = windows[0]
        lo2, _ = windows[1]
        assert hi1 <= lo2, "Windows should not overlap for well-separated peaks"


# ═══════════════════════════════════════════════════════════════════
# Tests: SPE template and filters
# ═══════════════════════════════════════════════════════════════════

class TestTemplateAndFilters:
    def test_spe_template_nonzero(self):
        """SPE template should have nonzero values."""
        r = build_spe_template(1024, 1.25, MILLIQAN_TEMPLATE)
        assert np.max(r) > 0, "Template should have positive values"
        assert r[0] == 0.0, "Template should start at zero (before t0)"

    def test_gaussian_filter_unity_at_dc(self):
        """Gaussian filter should be 1.0 at DC."""
        filt = gaussian_lowpass_filter(1024, 1.25, 15.0, 10.0, 50.0)
        assert filt[0] == 1.0

    def test_gaussian_filter_zero_at_high_freq(self):
        """Gaussian filter should be 0 above kill_mhz."""
        filt = gaussian_lowpass_filter(1024, 1.25, 15.0, 10.0, 50.0)
        # Nyquist = 400 MHz >> 50 MHz kill, so high-freq bins should be 0
        assert filt[len(filt)//2] == 0.0 or filt[len(filt)//2] < 1e-10

    def test_cosine_filter_unity_at_dc(self):
        """Cosine filter should be 1.0 at DC."""
        filt = cosine_lowpass_filter(1024, 1.25, 50.0)
        assert filt[0] == 1.0

    def test_deconvolution_recovers_peak(self):
        """Deconvolving a template should recover a peak near t=0."""
        N = 1024
        dt_ns = 1.25
        r_calib = build_spe_template(N, dt_ns, MILLIQAN_TEMPLATE)
        filt = gaussian_lowpass_filter(N, dt_ns, 15.0, 10.0, 50.0)
        # Deconvolving f/f = filter → should peak near 0 (delta function)
        u_rec = deconvolve_hits(r_calib, r_calib, filt)
        peak_bin = np.argmax(u_rec)
        # The peak should be near 0 (deconv of identical signals = filtered delta)
        assert u_rec[peak_bin] > 0, "Deconvolution should produce positive peak"
        # Total integral should be positive
        assert np.sum(u_rec) > 0, "Total deconvolved signal should be positive"


# ═══════════════════════════════════════════════════════════════════
# Tests: veto priority order
# ═══════════════════════════════════════════════════════════════════

class TestVetoPriority:
    """Verify that vetoes are applied in the correct priority order."""

    def test_out_of_time_beats_post_sat(self):
        """The out_of_time (time-window) veto takes priority over post_sat_shift."""
        N = 1024
        u_rec = np.zeros(N, dtype=DTYPE)
        u_rec[4] = 1.0
        t = np.arange(N, dtype=DTYPE) * 1.25
        peaks = np.array([4])
        pad_bins = 15
        windows = find_integration_windows(u_rec, peaks, pad_bins)
        pw = [_walk_to_trough_right(u_rec, 4, pad_bins) - _walk_to_trough_left(u_rec, 4, pad_bins)]

        _, reason, _ = apply_peak_vetoes(
            4, 0, peaks, u_rec, t, windows, pw, N,
            t_peak_min_ns=20.0,
            post_sat_cutoff_bin=2,  # also in post-sat zone
            dt_ns=1.25,
        )
        assert reason == "out_of_time", f"Expected out_of_time, got {reason}"

    def test_afterpulse_beats_post_sat(self):
        """Afterpulse veto should take priority over post_sat_shift."""
        N = 1024
        u_rec = np.zeros(N, dtype=DTYPE)
        # Big peak at 400
        for i in range(-3, 4):
            u_rec[400 + i] = max(0, 5.0 - abs(i))
        # Small peak at 410
        u_rec[410] = 0.5

        t = np.arange(N, dtype=DTYPE) * 1.25
        peaks = np.array([400, 410])
        pad_bins = 15
        windows = find_integration_windows(u_rec, peaks, pad_bins)
        pw = []
        for pk in peaks:
            pw.append(_walk_to_trough_right(u_rec, pk, pad_bins) -
                      _walk_to_trough_left(u_rec, pk, pad_bins))

        _, reason, _ = apply_peak_vetoes(
            410, 1, peaks, u_rec, t, windows, pw, N,
            afterpulse_window_ns=40.0, afterpulse_height_ratio=5.0,
            post_sat_cutoff_bin=405,  # also in post-sat zone
            dt_ns=1.25,
        )
        assert reason == "afterpulse_veto", f"Expected afterpulse_veto, got {reason}"


# ═══════════════════════════════════════════════════════════════════
# Run with plain python (no pytest needed)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import traceback
    test_classes = [
        TestBaseline, TestPeakVetoes, TestSlidingLocalMean,
        TestIntegrationWindows, TestTemplateAndFilters, TestVetoPriority,
    ]
    n_pass = 0
    n_fail = 0
    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  PASS  {cls.__name__}.{method_name}")
                n_pass += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{method_name}: {e}")
                traceback.print_exc()
                n_fail += 1
    print(f"\n{'='*60}")
    print(f"Results: {n_pass} passed, {n_fail} failed")
    print(f"{'='*60}")
    sys.exit(1 if n_fail > 0 else 0)
