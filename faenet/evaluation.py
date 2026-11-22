"""v3 evaluation: pooled out-of-fold test predictions per repeat (every employee scored once per
repeat), validation-frozen thresholds/calibration/review rules, employee-level paired bootstrap CIs.

  python scripts/03_evaluate.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from faenet.calibration import PlattCalibrator
from faenet.data import ROOT, get_data
from faenet.core import N_FOLDS, N_REPEATS, f1_threshold

PRED = os.path.join(ROOT, "results", "predictions")
MET = os.path.join(ROOT, "results", "metrics"); os.makedirs(MET, exist_ok=True)
D = get_data("ibm"); Y = D.y; G = D.groups; N = len(Y)
RNG = np.random.default_rng(0); B = int(os.environ.get("FAENET_BOOT", 500))  # bootstrap resamples
RATE = 0.10; C_FN = 5.0
SCORES = ["uncertainty", "margin", "entropy", "disagreement", "learned"]
AGES = ["<30", "30-39", "40+"]; GENDERS = ["Female", "Male"]


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def H2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def review_scores(d, ty):
    s = {"uncertainty": d["u"], "margin": 1 - np.abs(d["p"] - ty) / max(ty, 1 - ty), "entropy": H2(d["p"]),
         "disagreement": d["dis"]}
    if "defer" in d:
        s["learned"] = d["defer"]
    return s


def group_thresholds(yv, pv, gv, target):
    out = {}
    grid = np.linspace(0.01, 0.99, 197)
    for g in np.unique(gv):
        m = (gv == g) & (yv == 1)
        ok = [t for t in grid if m.any() and (pv[m] >= t).mean() >= target]
        out[g] = max(ok) if ok else 0.01
    return out


def pool(tag):
    """Returns per-repeat arrays over all N employees (each employee in exactly one test fold)."""
    R = []
    for r in range(N_REPEATS):
        A = {k: np.full(N, np.nan) for k in ("p", "pcal", "yhat", "yhat_cost", "yhat_G", "u", "dis", "defer")}
        S = {k: np.full(N, np.nan) for k in SCORES}
        flag = {(sc, rule): np.zeros(N, bool) for sc in SCORES for rule in ("global", "group", "shrunk")}
        for f in range(N_FOLDS):
            z = np.load(os.path.join(PRED, f"{tag}_r{r}_f{f}.npz"))
            v = {k[4:]: z[k] for k in z.files if k.startswith("val_")}
            t = {k[5:]: z[k] for k in z.files if k.startswith("test_")}
            te, va = t["idx"], v["idx"]
            ty = f1_threshold(v["y"], v["p"])
            grid = np.linspace(0.01, 0.99, 197)
            tc = grid[int(np.argmin([(C_FN * ((v["y"] == 1) & (v["p"] < x)).sum() + ((v["y"] == 0) & (v["p"] >= x)).sum()) for x in grid]))]
            cal = PlattCalibrator().fit(lg(v["p"]), v["y"])
            A["p"][te] = t["p"]; A["pcal"][te] = cal.transform(lg(t["p"]))
            A["yhat"][te] = t["p"] >= ty; A["yhat_cost"][te] = t["p"] >= tc
            A["u"][te] = t["u"]; A["dis"][te] = t["dis"]
            if "defer" in t:
                A["defer"][te] = t["defer"]
            # post-hoc per-age-band thresholds (needs age at decision time)
            target = float((v["p"][v["y"] == 1] >= ty).mean())
            th = group_thresholds(v["y"], v["p"], G["age_band"][va], target)
            A["yhat_G"][te] = t["p"] >= np.array([th.get(g, ty) for g in G["age_band"][te]])
            sv, st = review_scores(v, ty), review_scores(t, ty)
            for sc in sv:
                S[sc][te] = st[sc]
                tau = np.quantile(sv[sc], 1 - RATE)
                flag[(sc, "global")][te] = st[sc] > tau
                # group-aware (per age band) and shrunk toward global (n / (n + 50) weight)
                for g in AGES:
                    mv, mt = G["age_band"][va] == g, G["age_band"][te] == g
                    tg = np.quantile(sv[sc][mv], 1 - RATE) if mv.sum() >= 5 else tau
                    w = mv.sum() / (mv.sum() + 50.0)
                    flag[(sc, "group")][te[mt]] = st[sc][mt] > tg
                    flag[(sc, "shrunk")][te[mt]] = st[sc][mt] > (w * tg + (1 - w) * tau)
        A["S"] = S; A["flag"] = flag
        for k in ("yhat", "yhat_cost", "yhat_G"):
            A[k] = A[k].astype(bool)
        R.append(A)
    return R


def tpr(y, yh, m):
    m = m & (y == 1); return yh[m].mean() if m.any() else np.nan


def fpr(y, yh, m):
    m = m & (y == 0); return yh[m].mean() if m.any() else np.nan


def ece(y, p, n=10):
    o = np.argsort(p); return float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in np.array_split(o, n)))


def metrics_one(A, ix):
    """All scalar metrics for one repeat on employee subset ix (bootstrap resample)."""
    y = Y[ix]; p = A["p"][ix]; yh = A["yhat"][ix]; ga = G["age_band"][ix]; gg = G["gender"][ix]
    out = {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
           "precision": yh[y == 1].sum() / max(yh.sum(), 1), "recall": yh[y == 1].mean()}
    out["f1"] = 2 * out["precision"] * out["recall"] / max(out["precision"] + out["recall"], 1e-12)
    out["brier"] = np.mean((p - y) ** 2); out["ece"] = ece(y, p)
    out["brier_cal"] = np.mean((A["pcal"][ix] - y) ** 2); out["ece_cal"] = ece(y, A["pcal"][ix])
    for name, yhat in (("", yh), ("G_", A["yhat_G"][ix])):
        ta = [tpr(y, yhat, ga == g) for g in AGES]; tg = [tpr(y, yhat, gg == g) for g in GENDERS]
        fa = [fpr(y, yhat, ga == g) for g in AGES]
        out[f"{name}tpr_gap_age"] = np.nanmax(ta) - np.nanmin(ta); out[f"{name}tpr_gap_gender"] = np.nanmax(tg) - np.nanmin(tg)
        out[f"{name}worst_tpr_age"] = np.nanmin(ta); out[f"{name}fpr_gap_age"] = np.nanmax(fa) - np.nanmin(fa)
        out[f"{name}tpr_40plus"] = ta[2]; out[f"{name}fpr_40plus"] = fa[2]
        out[f"{name}recall"] = yhat[y == 1].mean(); out[f"{name}precision"] = yhat[y == 1].sum() / max(yhat.sum(), 1)
    out["auc_40plus"] = roc_auc_score(y[ga == "40+"], p[ga == "40+"])
    out["auc_under30"] = roc_auc_score(y[ga == "<30"], p[ga == "<30"])
    err = (yh != y)
    for sc in SCORES:
        s = A["S"][sc][ix]
        if np.isnan(s).all():
            continue
        out[f"errdet_{sc}"] = roc_auc_score(err, s)
        fl = A["flag"][(sc, "global")][ix]
        out[f"risk_retained_{sc}"] = err[~fl].mean(); out[f"review_rate_{sc}"] = fl.mean()
        # cost per employee: cost-threshold decisions, flagged cases resolved by reviewers of accuracy rho
        yc = A["yhat_cost"][ix]; base_cost = (C_FN * ((y == 1) & ~yc).sum() + ((y == 0) & yc).sum()) / len(y)
        fn_r = ((y == 1) & ~yc & ~fl).sum(); fp_r = ((y == 0) & yc & ~fl).sum()
        saved = base_cost - (C_FN * fn_r + fp_r) / len(y)
        out[f"breakeven_r_{sc}"] = saved / max(fl.mean(), 1e-9)   # review pays off if cost per review < this
        for rule in ("global", "group", "shrunk"):
            f2 = A["flag"][(sc, rule)][ix]
            rr = [f2[ga == g].mean() for g in AGES]
            out[f"review_gap_age_{rule}_{sc}"] = max(rr) - min(rr)
            ret = [err[(ga == g) & ~f2].mean() for g in AGES]
            out[f"retained_err_gap_age_{rule}_{sc}"] = np.nanmax(ret) - np.nanmin(ret)
    return out


def boot_summary(tags, cache):
    """mean over repeats of each metric on the full sample, and employee-level bootstrap CIs."""
    full = {t: pd.DataFrame([metrics_one(A, np.arange(N)) for A in cache[t]]).mean() for t in tags}
    idxs = [RNG.integers(0, N, N) for _ in range(B)]
    boots = {t: [] for t in tags}
    for ix in idxs:
        for t in tags:
            boots[t].append(pd.DataFrame([metrics_one(A, ix) for A in cache[t]]).mean())
    boots = {t: pd.DataFrame(v) for t, v in boots.items()}
    return full, boots


def main():
    variants = json.load(open(os.path.join(ROOT, "results", "variants.json")))
    tags = [t for t in variants if os.path.exists(os.path.join(PRED, f"{t}_r4_f4.npz"))]
    cache = {t: pool(t) for t in tags}
    full, boots = boot_summary(tags, cache)
    rows = []
    for t in tags:
        for m in full[t].index:
            lo, hi = np.nanpercentile(boots[t][m], [2.5, 97.5])
            sd = pd.DataFrame([metrics_one(A, np.arange(N)) for A in cache[t]])[m].std()
            rows.append({"variant": t, "metric": m, "mean": full[t][m], "sd_repeats": sd, "ci_lo": lo, "ci_hi": hi})
    summ = pd.DataFrame(rows); summ.to_csv(os.path.join(MET, "summary.csv"), index=False)

    # paired comparisons (same bootstrap resamples => paired at employee level)
    pairs = [("baseline", "faenet")]
    pairs += [("faenet", t) for t in tags if t in ("faenet_smote", "faenet_cellsmote")]
    pairs += [("faenet", t) for t in tags if t in ("faenet_softtpr", "faenet_dro", "faenet_fairsel", "faenet_dro_fairsel")]
    comp = []
    for a, b in pairs:
        if a not in tags or b not in tags:
            continue
        for m in full[a].index:
            if m not in full[b].index:
                continue
            d = boots[b][m] - boots[a][m]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            comp.append({"comparison": f"{b} - {a}", "metric": m, "diff": full[b][m] - full[a][m], "ci_lo": lo, "ci_hi": hi,
                         "p_two_sided": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
    # within-model score comparisons (learned / disagreement vs margin)
    for t in tags:
        for sc in ("uncertainty", "learned", "disagreement", "entropy"):
            k1, k2 = f"errdet_{sc}", "errdet_margin"
            if k1 in boots[t] and k2 in boots[t] and sc != "margin":
                d = boots[t][k1] - boots[t][k2]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                comp.append({"comparison": f"{t}: {sc} - margin", "metric": "errdet", "diff": full[t][k1] - full[t][k2],
                             "ci_lo": lo, "ci_hi": hi, "p_two_sided": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
        for rule in ("group", "shrunk"):
            for sc in SCORES:
                k1, k2 = f"review_gap_age_{rule}_{sc}", f"review_gap_age_global_{sc}"
                if k1 in boots[t]:
                    d = boots[t][k1] - boots[t][k2]; lo, hi = np.nanpercentile(d, [2.5, 97.5])
                    comp.append({"comparison": f"{t}: {rule} - global ({sc})", "metric": "review_gap_age",
                                 "diff": full[t][k1] - full[t][k2], "ci_lo": lo, "ci_hi": hi,
                                 "p_two_sided": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
    comp = pd.DataFrame(comp); comp.to_csv(os.path.join(MET, "paired.csv"), index=False)
    # per-repeat pooled predictions for figures
    np.savez_compressed(os.path.join(MET, "pooled_main.npz"),
                        **{f"{t}_{k}": np.array([A[k] for A in cache[t]]) for t in tags for k in ("p", "pcal", "yhat", "yhat_G", "u", "dis", "defer")},
                        y=Y, age=G["age_band"], gender=G["gender"])
    key = ["roc_auc", "pr_auc", "f1", "recall", "precision", "ece", "ece_cal", "tpr_gap_age", "tpr_gap_gender",
           "worst_tpr_age", "fpr_gap_age", "auc_40plus", "errdet_uncertainty", "errdet_margin", "errdet_learned",
           "errdet_disagreement"]
    piv = summ[summ.metric.isin(key)].pivot(index="variant", columns="metric", values="mean")[[k for k in key if k in set(summ.metric)]]
    print(piv.round(3).to_string())
    show = comp[comp.metric.isin(["roc_auc", "pr_auc", "tpr_gap_age", "worst_tpr_age", "recall", "errdet", "review_gap_age"])]
    print(show.round(3).to_string())


if __name__ == "__main__":
    main()
