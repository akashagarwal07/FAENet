"""Exploratory analysis of the IBM dataset: schema, imbalance, group distributions, proxy check."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from faenet.data import load_dataset
from faenet.plotting import INK2, SERIES, plt, save, setup
from faenet.preprocessing import make_split, prepare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {"ibm": "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"}
ORDER = {"age_band": ["<30", "30-39", "40+"], "salary": ["low", "medium", "high"]}


def group_table(data):
    rows = []
    for a, v in data.groups.items():
        for g in sorted(set(v)):
            m = v == g
            rows.append({"attribute": a, "group": g, "n": int(m.sum()), "leavers": int(data.y[m].sum()),
                         "attrition_rate": float(data.y[m].mean())})
    return pd.DataFrame(rows)


def plot_rates(tab, attrs, overall, title, path):
    fig, axes = plt.subplots(1, len(attrs), figsize=(3.6 * len(attrs), 2.8), squeeze=False, layout="constrained")
    fig.get_layout_engine().set(wspace=0.08)
    for ax, a in zip(axes[0], attrs):
        t = tab[tab.attribute == a].copy()
        if a in ORDER:
            t["k"] = t.group.map({g: i for i, g in enumerate(ORDER[a])}); t = t.sort_values("k")
        else:
            t = t.sort_values("attrition_rate")
        labels = [g.replace("Research & Development", "R&D").replace("Human Resources", "HR") for g in t.group]
        ax.set_axisbelow(True)
        ax.barh(labels, t.attrition_rate, color=SERIES[0], height=0.6)
        ax.axvline(overall, color=INK2, lw=1, ls="--")
        for i, (r, n, l) in enumerate(zip(t.attrition_rate, t.n, t.leavers)):
            ax.text(r + 0.005, i, f"{r:.0%}  ({l}/{n})", va="center", fontsize=7, color=INK2,
                    bbox=dict(fc="#fcfcfb", ec="none", pad=0.5))
        ax.set_xlim(0, max(t.attrition_rate) * 1.6); ax.set_title(a); ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    fig.suptitle(title + f"  (dashed line = overall {overall:.1%}; labels = leavers/n)", fontsize=9, color=INK2)
    save(fig, path)


def proxy_check(data, seed=42):
    """How well do the predictive features X recover each sensitive/fairness attribute? (val split)"""
    sp = make_split(data, seed); prep = prepare(data, sp)
    out = {}
    for a in data.spec.fairness_attrs:
        if a in data.spec.categorical:  # attribute itself is an input; trivially recoverable
            out[a] = "attribute is a predictive input"
            continue
        ytr, yva = prep.groups["train"][a], prep.groups["val"][a]
        clf = LogisticRegression(max_iter=2000).fit(prep.X["train"], ytr)
        pr = clf.predict_proba(prep.X["val"])
        auc = roc_auc_score(yva, pr[:, 1]) if pr.shape[1] == 2 else roc_auc_score(yva, pr, multi_class="ovr")
        out[a] = {"balanced_accuracy": float(balanced_accuracy_score(yva, clf.predict(prep.X["val"]))),
                  "auroc": float(auc), "chance_balanced_accuracy": 1 / pr.shape[1]}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "results", "eda"))
    args = ap.parse_args(); setup()
    summary = {}
    for name, rel in PATHS.items():
        data = load_dataset(name, os.path.join(ROOT, rel))
        tab = group_table(data)
        tab.to_csv(os.path.join(args.out, f"eda_groups_{name}.csv"), index=False)
        attrs = list(data.spec.fairness_attrs) + [a for a in data.spec.audit_attrs if a != "JobRole"]
        plot_rates(tab, attrs, data.y.mean(), f"{name}: attrition rate by group",
                   os.path.join(args.out, f"eda_attrition_by_group_{name}"))
        sp = make_split(data, 42)
        split_counts = {k: {"n": int(len(ix)), "leavers": int(data.y[ix].sum())}
                        for k, ix in (("train", sp.train), ("val", sp.val), ("test", sp.test))}
        test_group_pos = {a: {g: int(data.y[sp.test][data.groups[a][sp.test] == g].sum())
                              for g in sorted(set(data.groups[a]))} for a in data.spec.fairness_attrs}
        prep = prepare(data, sp)
        summary[name] = {**data.report, "split_seed42": split_counts, "test_leavers_by_group": test_group_pos,
                         "n_features_after_encoding": len(prep.feature_names),
                         "class_weights_seed42": prep.class_weights.tolist(),
                         "proxy_check": proxy_check(data)}
        for k in ("raw_columns", "dtypes"):
            summary[name].pop(k)
    with open(os.path.join(args.out, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({n: {k: s[k] for k in ("final_shape", "positive_rate", "split_seed42",
                                            "test_leavers_by_group", "n_features_after_encoding",
                                            "class_weights_seed42", "proxy_check")}
                      for n, s in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
