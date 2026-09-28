"""Loads the DREAMER dataset (.mat) and builds a per-(subject, clip) feature table."""
import pandas as pd
import scipy.io as sio

from .features import eeg_band_power_features, ecg_hrv_features


def load_dreamer(mat_path):
    """Returns the DREAMER struct: .Data (23 subjects), .EEG_SamplingRate, .ECG_SamplingRate."""
    raw = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    return raw["DREAMER"]


def build_feature_table(dreamer, n_clips=18):
    """Extracts EEG + ECG features for every (subject, clip) pair.

    Returns a DataFrame with one row per trial: feature columns, plus
    subject, clip, and the raw valence/arousal/dominance self-report scores (1-5).
    """
    eeg_fs = int(dreamer.EEG_SamplingRate)
    ecg_fs = int(dreamer.ECG_SamplingRate)

    rows = []
    for subject_idx, participant in enumerate(dreamer.Data):
        for clip_idx in range(n_clips):
            feats = {}
            feats.update(eeg_band_power_features(
                participant.EEG.stimuli[clip_idx], participant.EEG.baseline[clip_idx], fs=eeg_fs
            ))
            feats.update(ecg_hrv_features(participant.ECG.stimuli[clip_idx], fs=ecg_fs))
            feats["subject"] = subject_idx
            feats["clip"] = clip_idx
            feats["valence"] = participant.ScoreValence[clip_idx]
            feats["arousal"] = participant.ScoreArousal[clip_idx]
            feats["dominance"] = participant.ScoreDominance[clip_idx]
            rows.append(feats)

    return pd.DataFrame(rows)
