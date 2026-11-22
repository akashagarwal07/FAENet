"""H5 (v3): do review scores rise for employees from a group never seen in training?

Held-out groups: each Department with n >= 100 (Sales, R&D) and each JobRole with n >= 100.
The group is removed from train/val/test; the main v3 model is trained on the rest (3 repeats);
we compare scores on held-out employees with the in-distribution test fold.
Metric: AUROC of each score for separating held-out from familiar employees (0.5 = no signal).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from faenet.core import cfg_from
from faenet.data import ROOT, get_data
from faenet.core import add_engineered, f1_threshold, run_split

MET = os.path.join(ROOT, "results", "metrics"); os.makedirs(MET, exist_ok=True)


def H2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main():
    D = get_data("ibm"); FR = add_engineered(D.frame)
    cfg = cfg_from(json.load(open(os.path.join(ROOT, "results", "variants.json")))["faenet"])
    rows = []
    for attr in ("Department", "JobRole"):
        vals = D.frame[attr].astype(str).to_numpy()
        for g in sorted(set(vals)):
            m = vals == g
            if m.sum() < 100:
                continue
            for r in range(3):
                res = run_split(D, FR, r, 0, cfg, holdout_mask=m)
                v, t, h = res["val"], res["test"], res["heldout"]
                ty = f1_threshold(v["y"], v["p"])
                def sc(d):
                    return {"uncertainty": d["u"], "margin": 1 - np.abs(d["p"] - ty) / max(ty, 1 - ty),
                            "entropy": H2(d["p"]), "disagreement": d["dis"], "learned": d["defer"]}
                st, sh = sc(t), sc(h)
                lab = np.r_[np.zeros(len(t["p"])), np.ones(len(h["p"]))]
                row = {"attr": attr, "heldout": g, "repeat": r, "n_heldout": int(m.sum()),
                       "err_in": float(((t["p"] >= ty) != t["y"]).mean()), "err_heldout": float(((h["p"] >= ty) != h["y"]).mean())}
                for k in st:
                    row[f"auroc_{k}"] = float(roc_auc_score(lab, np.r_[st[k], sh[k]]))
                    row[f"mean_{k}_in"] = float(st[k].mean()); row[f"mean_{k}_heldout"] = float(sh[k].mean())
                if 0 < h["y"].sum() < len(h["y"]):
                    row["auc_heldout"] = float(roc_auc_score(h["y"], h["p"]))
                rows.append(row)
                print(attr, g, r, {k: round(v, 3) for k, v in row.items() if k.startswith(("auroc", "err", "auc_"))}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(MET, "holdout.csv"), index=False)
    print(df.groupby("attr")[[c for c in df.columns if c.startswith("auroc")]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
