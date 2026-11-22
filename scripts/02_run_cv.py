"""v3 full runs: every variant on all 25 CV splits (5 repeats x 5 folds), settings frozen from
validation-only selection (results/selection/*.json). Saves per-split predictions.

  python scripts/02_run_cv.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import asdict, replace

from faenet.data import ROOT, get_data
from faenet.core import N_FOLDS, N_REPEATS, Cfg, add_engineered, run_split

SEL = os.path.join(ROOT, "results", "selection")
PRED = os.path.join(ROOT, "results", "predictions"); os.makedirs(PRED, exist_ok=True)


def cfg_from(d):
    return Cfg(**{**d, "hidden_dims": tuple(d["hidden_dims"])})


def variants():
    base = cfg_from(json.load(open(os.path.join(SEL, "base.json")))["best"]["cfg"])
    aug_sel = json.load(open(os.path.join(SEL, "aug.json")))
    aug = cfg_from(aug_sel["best"]["cfg"])
    fair = json.load(open(os.path.join(SEL, "fair.json")))
    V = {
        "baseline": Cfg(n_members=1),                      # baseline: single evidential network, original features and full class weights
    }
    # every augmentation option at its best validated ratio (reported for the ablation)
    for mode in ("smote", "cell_smote"):
        rs = [r for r in aug_sel["all"] if r["cfg"]["aug_mode"] == mode]
        if rs:
            V[f"faenet_{mode.replace('_', '')}"] = cfg_from(max(rs, key=lambda r: r["val_pr_auc"])["cfg"])
    V["faenet"] = aug   # selected base (+augmentation if it won on validation): the FAENet-v3 main model
    for fam, tag in (("soft_tpr", "faenet_softtpr"), ("group_dro", "faenet_dro"), ("fair_select", "faenet_fairsel")):
        ch = fair["families"][fam]["chosen"]
        if ch:
            V[tag] = cfg_from(ch["cfg"])
    if "faenet_dro" in V and "faenet_fairsel" in V:  # combination (not separately tuned)
        V["faenet_dro_fairsel"] = replace(V["faenet_dro"], sel_kappa=V["faenet_fairsel"].sel_kappa)
    return V


def main():
    D = get_data("ibm"); FR = add_engineered(D.frame)
    V = variants()
    json.dump({k: {**asdict(v), "hidden_dims": list(v.hidden_dims)} for k, v in V.items()},
              open(os.path.join(ROOT, "results", "variants.json"), "w"), indent=2)
    t0 = time.time()
    for tag, cfg in V.items():
        for r in range(N_REPEATS):
            for f in range(N_FOLDS):
                path = os.path.join(PRED, f"{tag}_r{r}_f{f}.npz")
                if os.path.exists(path):
                    continue
                res = run_split(D, FR, r, f, cfg, with_deferral=cfg.n_members > 1)
                arr = {"epochs": np.array(res["epochs"]), "member_test_p": res["member_test_p"]}
                for s in ("val", "test"):
                    arr[f"{s}_idx"] = res["idx"][s]
                    for k, v in res[s].items():
                        if k != "groups":
                            arr[f"{s}_{k}"] = v
                np.savez_compressed(path, **arr)
        print(f"{tag} done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
