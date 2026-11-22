"""v3 model selection on VALIDATION data only (repeat 0, its 5 CV splits; test folds never read).

  python scripts/01_select.py --stage base     # architecture / optimiser / features / class weights
  python scripts/01_select.py --stage aug      # synthetic augmentation on top of the chosen base
  python scripts/01_select.py --stage fair     # fairness families on top of base(+aug)

Selection rules (fixed in advance):
  base, aug : maximise pooled validation PR-AUC.
  fair      : among settings passing the floor (val PR-AUC >= base - 0.02) AND the collapse guard
              (val ROC-AUC >= 0.65 and predicted-positive rate <= 0.6), choose the smallest pooled
              validation TPR gap (age-band + gender, hard, at each split's F1 threshold); ties -> PR-AUC.
"""
import argparse, itertools, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import asdict, replace
from sklearn.metrics import average_precision_score, roc_auc_score

from faenet.data import ROOT, get_data
from faenet.core import Cfg, add_engineered, f1_threshold, run_split

OUT = os.path.join(ROOT, "results", "selection"); os.makedirs(OUT, exist_ok=True)
D = get_data("ibm"); FR = add_engineered(D.frame)


def tpr_gap(y, yh, g):
    r = [yh[(g == k) & (y == 1)].mean() for k in np.unique(g) if ((g == k) & (y == 1)).any()]
    return max(r) - min(r)


def evaluate(cfg):
    ys, ps, yhs, gg, ga = [], [], [], [], []
    for fold in range(5):
        r = run_split(D, FR, 0, fold, cfg, with_deferral=False)
        v = r["val"]; t = f1_threshold(v["y"], v["p"])
        ys.append(v["y"]); ps.append(v["p"]); yhs.append((v["p"] >= t).astype(int))
        gg.append(v["groups"]["gender"]); ga.append(v["groups"]["age_band"])
    y, p, yh, g1, g2 = map(np.concatenate, (ys, ps, yhs, gg, ga))
    return {"val_pr_auc": float(np.mean([average_precision_score(a, b) for a, b in zip(ys, ps)])),
            "val_roc_auc": float(np.mean([roc_auc_score(a, b) for a, b in zip(ys, ps)])),
            "val_gap_age": float(tpr_gap(y, yh, g2)), "val_gap_gender": float(tpr_gap(y, yh, g1)),
            "val_pos_rate": float(yh.mean()), "val_recall": float(yh[y == 1].mean())}


def cfg_json(c):
    d = asdict(c); d["hidden_dims"] = list(d["hidden_dims"]); return d


def load(stage):
    return json.load(open(os.path.join(OUT, f"{stage}.json")))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["base", "aug", "fair"])
    a = ap.parse_args(); rows = []; t0 = time.time()
    if a.stage == "base":
        grid = itertools.product([False, True], [1.0, 0.5], [0.0, 1e-3, 1e-2], [1e-3, 3e-3], [(32, 16), (64, 32)])
        cands = [Cfg(engineered=e, cw_power=c, weight_decay=w, lr=lr, hidden_dims=h) for e, c, w, lr, h in grid]
    elif a.stage == "aug":
        base = Cfg(**{**load("base")["best"]["cfg"], "hidden_dims": tuple(load("base")["best"]["cfg"]["hidden_dims"])})
        cands = [base] + [replace(base, aug_mode=m, aug_ratio=r) for m in ("smote", "cell_smote") for r in (0.5, 1.0)]
    else:
        src = load("aug")["best"]["cfg"]
        base = Cfg(**{**src, "hidden_dims": tuple(src["hidden_dims"])})
        cands = [base]
        cands += [replace(base, lambda_f=l) for l in (1.0, 3.0, 10.0, 30.0)]
        cands += [replace(base, dro_eta=e, dro_adj=adj, cw_power=0.0) for e in (0.01, 0.05, 0.2) for adj in (0.0, 1.0)]
        cands += [replace(base, sel_kappa=k) for k in (0.25, 0.5, 1.0, 2.0)]
    for i, c in enumerate(cands):
        m = evaluate(c)
        rows.append({"cfg": cfg_json(c), **m})
        print(f"[{a.stage} {i + 1}/{len(cands)} {time.time() - t0:.0f}s] " +
              " ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
    if a.stage in ("base", "aug"):
        best = max(rows, key=lambda r: r["val_pr_auc"])
        out = {"rule": "max pooled validation PR-AUC", "best": best, "all": rows}
    else:
        b = rows[0]
        def fam(r):
            c = r["cfg"]
            return "soft_tpr" if c["lambda_f"] > 0 else "group_dro" if c["dro_eta"] > 0 else "fair_select" if c["sel_kappa"] > 0 else "base"
        out = {"rule": "floor PR>=base-0.02, guard ROC>=0.65 & pos-rate<=0.6, then min val gap (age+gender)",
               "base": b, "families": {}, "all": rows}
        for f in ("soft_tpr", "group_dro", "fair_select"):
            fr = [r for r in rows if fam(r) == f]
            ok = [r for r in fr if r["val_pr_auc"] >= b["val_pr_auc"] - 0.02 and r["val_roc_auc"] >= 0.65 and r["val_pos_rate"] <= 0.6]
            pick = min(ok, key=lambda r: (r["val_gap_age"] + r["val_gap_gender"], -r["val_pr_auc"])) if ok else None
            out["families"][f] = {"chosen": pick, "n_passing": len(ok), "n": len(fr)}
    json.dump(out, open(os.path.join(OUT, f"{a.stage}.json"), "w"), indent=2)
    print("DONE", json.dumps(out.get("best", out.get("families")), indent=1)[:2500])


if __name__ == "__main__":
    main()
