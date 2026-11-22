"""Predictive, calibration, fairness and selective-prediction metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def predictive(y, p, threshold=0.5) -> dict:
    yhat = (p >= threshold).astype(int)
    out = {
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "accuracy": accuracy_score(y, yhat),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y, p) if len(set(y)) > 1 else float("nan"),
        "brier": brier_score(y, p),
        "ece": ece(y, p),
        "threshold": threshold,
        "positive_rate_pred": float(yhat.mean()),
    }
    return {k: float(v) for k, v in out.items()}


def brier_score(y, p) -> float:
    return float(np.mean((p - y) ** 2))


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-mass (quantile) bins."""
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    n = len(p)
    return float(sum(len(b) / n * abs(p[b].mean() - y[b].mean()) for b in bins if len(b)))


def reliability_table(y, p, n_bins: int = 10):
    order = np.argsort(p)
    rows = []
    for b in np.array_split(order, n_bins):
        if len(b):
            rows.append({"mean_pred": float(p[b].mean()), "obs_rate": float(y[b].mean()), "n": int(len(b))})
    return rows


def group_metrics(y, yhat, groups: np.ndarray) -> dict:
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        yt, yp = y[m], yhat[m]
        tp = int(((yt == 1) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum()); tn = int(((yt == 0) & (yp == 0)).sum())
        out[g] = {"n": int(m.sum()), "positives": tp + fn, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                  "TPR": tp / (tp + fn) if tp + fn else float("nan"),
                  "FPR": fp / (fp + tn) if fp + tn else float("nan"),
                  "precision": tp / (tp + fp) if tp + fp else float("nan"),
                  "error_rate": (fp + fn) / m.sum() if m.sum() else float("nan")}
    return out


def eo_gap(gm: dict, key: str = "TPR") -> float:
    """Equal-opportunity gap = max - min group rate over groups where the rate is defined."""
    vals = [v[key] for v in gm.values() if not np.isnan(v[key])]
    return float(max(vals) - min(vals)) if len(vals) > 1 else float("nan")


def selective(y, yhat, score, tau) -> dict:
    """Review rule: flag score > tau. Coverage = fraction NOT flagged; risk = error among retained."""
    keep = score <= tau
    err = (y != yhat)
    out = {"tau": float(tau), "review_rate": float(1 - keep.mean()), "coverage": float(keep.mean()),
           "risk_retained": float(err[keep].mean()) if keep.any() else float("nan"),
           "risk_flagged": float(err[~keep].mean()) if (~keep).any() else float("nan"),
           "overall_error": float(err.mean())}
    if keep.any() and y[keep].sum():
        out["recall_retained"] = float(((yhat == 1) & (y == 1))[keep].sum() / y[keep].sum())
    return out


def coverage_risk_curve(y, yhat, score, n_points: int = 50):
    order = np.argsort(score)  # most certain first
    err = (y != yhat)[order]
    n = len(y)
    cov, risk = [], []
    for k in np.linspace(max(1, n // n_points), n, n_points).astype(int):
        cov.append(k / n); risk.append(float(err[:k].mean()))
    aurc = float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(np.trapz(risk, cov))
    return np.array(cov), np.array(risk), aurc


def error_detection_auc(y, yhat, score) -> float:
    """AUROC of a review score for detecting misclassified cases."""
    err = (y != yhat).astype(int)
    return float(roc_auc_score(err, score)) if 0 < err.sum() < len(err) else float("nan")
