"""Signal-processing feature extraction for EEG and ECG.

EEG: baseline-corrected band power (theta/alpha/beta/gamma) per channel,
via Welch PSD -- a standard feature for EEG affect recognition.

ECG: heart-rate and HRV features derived from R-peak detection --
mean heart rate, SDNN, RMSSD, and LF/HF power ratio of the RR tachogram.
"""
import numpy as np
from scipy.signal import welch, find_peaks

EEG_BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def _band_power(signal, fs, band):
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs * 2))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapz(psd[mask], freqs[mask])


def eeg_band_power_features(stimuli, baseline, fs, bands=EEG_BANDS):
    """stimuli, baseline: (n_samples, n_channels) arrays for the same channels.

    Returns a dict of log-power(stimuli) - log-power(baseline) per channel per band,
    which removes each subject's resting-state EEG offset before comparing conditions.
    """
    n_channels = stimuli.shape[1]
    feats = {}
    for ch in range(n_channels):
        for band_name, band in bands.items():
            stim_power = _band_power(stimuli[:, ch], fs, band)
            base_power = _band_power(baseline[:, ch], fs, band)
            feats[f"ch{ch}_{band_name}"] = np.log1p(stim_power) - np.log1p(base_power)
    return feats


def ecg_hrv_features(stimuli, fs, lead=0):
    """stimuli: (n_samples, n_leads). Returns heart-rate/HRV features from one lead."""
    signal = stimuli[:, lead]
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    min_distance = int(fs * 0.4)  # refractory period, caps detectable HR at ~150 bpm
    peaks, _ = find_peaks(signal, distance=min_distance, height=0.5)

    if len(peaks) < 3:
        return {"hr_mean": np.nan, "sdnn": np.nan, "rmssd": np.nan, "lf_hf_ratio": np.nan}

    rr = np.diff(peaks) / fs  # seconds between heartbeats
    hr_mean = np.mean(60.0 / rr)
    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else np.nan
    lf_hf_ratio = _lf_hf_ratio(rr)

    return {"hr_mean": hr_mean, "sdnn": sdnn, "rmssd": rmssd, "lf_hf_ratio": lf_hf_ratio}


def _lf_hf_ratio(rr, resample_hz=4.0):
    """LF/HF power ratio from the RR-interval tachogram, resampled to a uniform grid
    (standard frequency-domain HRV analysis). Returns NaN if there's too little data
    to resolve the LF/HF bands reliably.
    """
    if len(rr) <= 4:
        return np.nan
    t_rr = np.cumsum(rr)
    t_uniform = np.arange(t_rr[0], t_rr[-1], 1 / resample_hz)
    if len(t_uniform) <= 8:
        return np.nan
    rr_interp = np.interp(t_uniform, t_rr, rr)
    freqs, psd = welch(rr_interp, fs=resample_hz, nperseg=min(len(rr_interp), 32))
    lf = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)], freqs[(freqs >= 0.04) & (freqs < 0.15)])
    hf = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)], freqs[(freqs >= 0.15) & (freqs < 0.4)])
    return lf / hf if hf > 0 else np.nan
