"""Build the DREAMER feature table (or load a cached one) and evaluate binary
high/low classifiers for valence, arousal, and dominance using subject-independent
(GroupKFold by subject) cross-validation -- so no subject's data leaks between
train and test, which is what makes the reported accuracy meaningful.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from eeg_ecg_emotion import build_feature_table, load_dreamer

TARGETS = ["valence", "arousal", "dominance"]


def evaluate(df, n_splits=5, random_state=42):
    feature_cols = [c for c in df.columns if c not in ("subject", "clip", *TARGETS)]
    X = df[feature_cols].values
    groups = df["subject"].values

    results = {}
    for target in TARGETS:
        y = (df[target] > 3).astype(int).values  # standard DREAMER high/low split at the midpoint
        gkf = GroupKFold(n_splits=n_splits)
        accs, f1s, baseline_accs = [], [], []

        for train_idx, test_idx in gkf.split(X, y, groups):
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=random_state, class_weight="balanced"
            )
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], pred))
            f1s.append(f1_score(y[test_idx], pred, average="macro"))

            baseline = DummyClassifier(strategy="most_frequent")
            baseline.fit(X[train_idx], y[train_idx])
            baseline_accs.append(accuracy_score(y[test_idx], baseline.predict(X[test_idx])))

        results[target] = {
            "accuracy": (np.mean(accs), np.std(accs)),
            "macro_f1": np.mean(f1s),
            "baseline_accuracy": np.mean(baseline_accs),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate EEG+ECG emotion classifiers on DREAMER")
    parser.add_argument("--dreamer-mat", default="data/DREAMER.mat", help="Path to DREAMER.mat")
    parser.add_argument("--cache-csv", default="data/dreamer_features.csv",
                         help="Where to cache the extracted feature table (skips re-extraction if present)")
    parser.add_argument("--rebuild", action="store_true", help="Force re-extraction even if cache exists")
    args = parser.parse_args()

    import os
    if not args.rebuild and os.path.exists(args.cache_csv):
        df = pd.read_csv(args.cache_csv)
        print(f"Loaded cached features from {args.cache_csv} ({len(df)} rows)")
    else:
        print(f"Loading {args.dreamer_mat} and extracting features (this takes a few minutes)...")
        dreamer = load_dreamer(args.dreamer_mat)
        df = build_feature_table(dreamer)
        df.to_csv(args.cache_csv, index=False)
        print(f"Extracted {len(df)} rows, cached to {args.cache_csv}")

    results = evaluate(df)
    print("\nSubject-independent 5-fold cross-validation (GroupKFold by subject):")
    for target, r in results.items():
        acc_mean, acc_std = r["accuracy"]
        print(
            f"  {target:10s}: accuracy={acc_mean:.3f} (+/-{acc_std:.3f})  "
            f"macro-F1={r['macro_f1']:.3f}  majority-class baseline={r['baseline_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
