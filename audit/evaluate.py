"""Build a test set (half clean, half corrupted with a known type/severity) and
score every detector: AUROC of 'corrupted vs clean', broken down by corruption
type and severity. Reference cases for calibration are disjoint from test cases.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .corrupt import CORRUPTIONS
from .detectors import Calibrated

DETECTORS = ["blind", "edge_signed", "edge", "components", "rules", "combined"]


def split_cases(cases, n_ref):
    return cases[:n_ref], cases[n_ref:]


def run(cases, n_ref=60, severities=(1, 2, 3), seed=0, extra_detectors=None):
    """cases: list of (img, mask). extra_detectors: {name: fn(img, mask) -> score}."""
    rng = np.random.default_rng(seed)
    ref, test = split_cases(cases, n_ref)
    cal = Calibrated(ref)
    fns = {d: getattr(cal, d) for d in DETECTORS}
    fns.update(extra_detectors or {})

    # clean test scores (shared negative class)
    clean = {d: np.array([f(i, m) for i, m in test]) for d, f in fns.items()}

    rows = []
    for ctype, corrupt in CORRUPTIONS.items():
        for sev in severities:
            bad = {d: [] for d in fns}
            for i, m in test:
                m2 = corrupt(m, sev, rng)
                for d, f in fns.items():
                    bad[d].append(f(i, m2))
            for d in fns:
                y = np.r_[np.zeros(len(test)), np.ones(len(test))]
                s = np.r_[clean[d], np.array(bad[d])]
                rows.append(dict(corruption=ctype, severity=sev, detector=d,
                                 auroc=roc_auc_score(y, s)))
    return pd.DataFrame(rows)
