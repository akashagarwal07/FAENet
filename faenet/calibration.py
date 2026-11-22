"""Post-hoc probability calibration (fit on validation only).

Platt scaling on the evidential log-odds log(alpha_1/alpha_0):  p_cal = sigmoid(a * logodds + b).
A pure temperature (a only) cannot undo the base-rate shift introduced by class weighting,
so an intercept b is included. The evidential uncertainty score is NOT recalibrated.
"""
from __future__ import annotations

import joblib
import numpy as np
from scipy.optimize import minimize


class PlattCalibrator:
    def __init__(self):
        self.a, self.b = 1.0, 0.0

    def fit(self, logodds: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        z = logodds.astype(np.float64); y = y.astype(np.float64)

        def nll(w):
            s = w[0] * z + w[1]
            return np.mean(np.logaddexp(0, s) - y * s)

        res = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
        self.a, self.b = float(res.x[0]), float(res.x[1])
        return self

    def transform(self, logodds: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * logodds + self.b)))

    def save(self, path):
        joblib.dump({"a": self.a, "b": self.b}, path)

    @classmethod
    def load(cls, path):
        c = cls(); d = joblib.load(path); c.a, c.b = d["a"], d["b"]; return c
