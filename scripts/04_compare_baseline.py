"""Baseline (single evidential network) vs final FAENet: paired employee-level bootstrap,
comparison table and the main figures for the final report.

Final FAENet = cross-fitted 5-member evidential ensemble (faenet) with age-aware decision
thresholds for the fairness rows and group-balanced (per-age-band) review thresholds."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from faenet.evaluation import AGES, G, N, Y, metrics_one, pool
from faenet.plotting import INK2, SERIES, plt, save, setup
from faenet.metrics import coverage_risk_curve
from faenet.data import ROOT

OUT = os.path.join(ROOT, "results", "metrics"); FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
TAB = os.path.join(ROOT, "report", "tables"); os.makedirs(TAB, exist_ok=True)
RNG = np.random.default_rng(0); B = int(os.environ.get("FAENET_BOOT", 500))  # bootstrap resamples

ROWS = [  # label, baseline key, final key, decimals, better ("up"/"down")
    ("ROC-AUC", "roc_auc", "roc_auc", "up"),
    ("PR-AUC", "pr_auc", "pr_auc", "up"),
    ("F1-score", "f1", "f1", "up"),
    ("Precision", "precision", "precision", "up"),
    ("Recall", "recall", "recall", "up"),
    ("ECE (calibrated)", "ece_cal", "ece_cal", "down"),
    ("Brier score (calibrated)", "brier_cal", "brier_cal", "down"),
    ("Age-band TPR gap", "tpr_gap_age", "G_tpr_gap_age", "down"),
    ("TPR, age 40+", "tpr_40plus", "G_tpr_40plus", "up"),
    ("Worst age-band TPR", "worst_tpr_age", "G_worst_tpr_age", "up"),
    ("Error-detection AUROC of $u$", "errdet_uncertainty", "errdet_uncertainty", "up"),
    ("Error after reviewing 10\\%", "risk_retained_uncertainty", "risk_retained_uncertainty", "down"),
]


def main():
    setup()
    cb, cf = pool("baseline"), pool("faenet")
    full_b = pd.DataFrame([metrics_one(A, np.arange(N)) for A in cb]).mean()
    full_f = pd.DataFrame([metrics_one(A, np.arange(N)) for A in cf]).mean()
    diffs = {r[0]: [] for r in ROWS}
    for _ in range(B):
        ix = RNG.integers(0, N, N)
        mb = pd.DataFrame([metrics_one(A, ix) for A in cb]).mean()
        mf = pd.DataFrame([metrics_one(A, ix) for A in cf]).mean()
        for lab, kb, kf, _ in ROWS:
            diffs[lab].append(mf[kf] - mb[kb])
    res = []
    for lab, kb, kf, better in ROWS:
        d = np.array(diffs[lab]); lo, hi = np.nanpercentile(d, [2.5, 97.5])
        res.append({"metric": lab, "baseline": float(full_b[kb]), "faenet": float(full_f[kf]),
                    "diff": float(full_f[kf] - full_b[kb]), "ci_lo": float(lo), "ci_hi": float(hi), "better": better})
    df = pd.DataFrame(res); df.to_csv(os.path.join(OUT, "baseline_vs_final.csv"), index=False)
    print(df.round(3).to_string())

    L = ["\\begin{table}[t]\\centering",
         "\\caption{Baseline (single evidential network) versus final FAENet. Mean over 5 repeats of pooled out-of-fold predictions; "
         "$\\Delta$ with 95\\% paired employee-level bootstrap CI. Fairness rows use FAENet's age-aware thresholds; "
         "the review row refers the 10\\% most uncertain employees.}\\label{tab:main}",
         "\\scriptsize\\setlength{\\tabcolsep}{3pt}\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lccc}\\toprule",
         "Metric & Baseline & \\textbf{FAENet} & $\\Delta$ [95\\% CI]\\\\\\midrule"]
    groups = {"ROC-AUC": "\\multicolumn{4}{l}{\\textit{Prediction}}\\\\",
              "ECE (calibrated)": "\\midrule\\multicolumn{4}{l}{\\textit{Calibration}}\\\\",
              "Age-band TPR gap": "\\midrule\\multicolumn{4}{l}{\\textit{Fairness}}\\\\",
              "Error-detection AUROC of $u$": "\\midrule\\multicolumn{4}{l}{\\textit{Uncertainty and review}}\\\\"}
    for r in res:
        if r["metric"] in groups:
            L.append(groups[r["metric"]])
        L.append(f"{r['metric']} & {r['baseline']:.3f} & \\textbf{{{r['faenet']:.3f}}} & "
                 f"{r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]\\\\")
    L += ["\\bottomrule\\end{tabular}}\\end{table}"]
    open(os.path.join(TAB, "baseline_vs_final.tex"), "w").write("\n".join(L))

    # ---- Fig: TPR by age band, baseline vs final (age-aware) ----
    npz = np.load(os.path.join(ROOT, "results", "metrics", "pooled_main.npz"), allow_pickle=True)
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(4.4, 2.7), layout="constrained"); w = 0.36
    for i, (key, lab) in enumerate((("baseline_yhat", "Baseline"), ("faenet_yhat_G", "FAENet"))):
        yh = npz[key]; vals, los, his = [], [], []
        for g in AGES:
            m = (G["age_band"] == g) & (Y == 1); per = yh[:, m].mean(0)
            bs = [per[rng.integers(0, m.sum(), m.sum())].mean() for _ in range(1000)]
            vals.append(per.mean()); los.append(np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5))
        x = np.arange(3) + (i - 0.5) * w
        ax.bar(x, vals, w * 0.92, color=SERIES[i], label=lab)
        ax.errorbar(x, vals, yerr=[np.array(vals) - los, np.array(his) - vals], fmt="none", ecolor=INK2, lw=0.8)
    ax.set_xticks(range(3), [f"{g}\n({int(((G['age_band'] == g) & (Y == 1)).sum())} leavers)" for g in AGES])
    ax.set_ylabel("True-positive rate"); ax.set_ylim(0, 1); ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
    save(fig, os.path.join(FIG, "age_tpr_final"))

    # ---- Fig: uncertainty (coverage-risk) and review balance ----
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    grid = np.linspace(0.5, 1.0, 40)
    for i, (cache, lab) in enumerate(((cb, "Baseline"), (cf, "FAENet"))):
        cur = []
        for A in cache:
            c, r, _ = coverage_risk_curve(Y, A["yhat"].astype(int), A["S"]["uncertainty"], 80); cur.append(np.interp(grid, c, r))
        axes[0].plot(grid, np.mean(cur, 0), color=SERIES[i], label=lab)
    axes[0].set_xlabel("Coverage (share not referred)"); axes[0].set_ylabel("Error among retained")
    axes[0].set_title("Referring by evidential uncertainty"); axes[0].legend(fontsize=7)
    w = 0.36
    for i, (cache, rule, lab) in enumerate(((cf, "global", "one global threshold"), (cf, "group", "FAENet group-balanced"))):
        rates = [np.mean([A["flag"][("uncertainty", rule)][G["age_band"] == g].mean() for A in cache]) for g in AGES]
        axes[1].bar(np.arange(3) + (i - 0.5) * w, rates, w * 0.92, color=SERIES[i], label=lab)
    axes[1].axhline(0.10, color=INK2, lw=1, ls="--"); axes[1].set_xticks(range(3), AGES)
    axes[1].set_ylabel("Share referred to review"); axes[1].set_title("FAENet: 10% review budget by age band"); axes[1].legend(fontsize=7)
    axes[1].grid(axis="x", visible=False); axes[1].set_axisbelow(True)
    save(fig, os.path.join(FIG, "uncertainty_review_final"))
    json.dump(res, open(os.path.join(OUT, "baseline_vs_final.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
