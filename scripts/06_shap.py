"""SHAP for the v3 main model: the 5-member cross-fitted ensemble trained on repeat 0 / fold 0.
Explains the ensemble's attrition probability and its mean evidential uncertainty on the test fold.
SHAP describes the trained model, not causal effects."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.model_selection import StratifiedKFold

from faenet.core import cfg_from
from faenet.plotting import INK2, SERIES, plt, save, setup
from faenet.preprocessing import build_preprocessor
from faenet.data import ROOT, get_data
from faenet.core import add_engineered, augment, cell_labels, cv_split, feature_columns, train_member

OUT = os.path.join(ROOT, "results", "explanations"); os.makedirs(OUT, exist_ok=True)


def main():
    setup()
    D = get_data("ibm"); FR = add_engineered(D.frame)
    cfg = cfg_from(json.load(open(os.path.join(ROOT, "results", "variants.json")))["faenet"])
    tr, va, te = cv_split(D.y, D.groups, 0, 0)
    num, cat = feature_columns(cfg.engineered)
    pre = build_preprocessor(num, cat).fit(FR.iloc[tr][num + cat])
    T = lambda ix: pre.transform(FR.iloc[ix][num + cat]).astype(np.float32)
    Xtr, Xva, Xte = T(tr), T(va), T(te)
    names = list(pre.get_feature_names_out())
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    fam_map = {f"cat__{c}_{v}": c for c, cats in zip(cat, ohe.categories_) for v in cats}
    fams = [n.split("__", 1)[1] if n.startswith("num__") else fam_map[n] for n in names]
    cells = cell_labels(D.y, D.groups)
    gva = {a: D.groups[a][va] for a in ("gender", "age_band")}
    models = []
    for k, (mi, _) in enumerate(StratifiedKFold(cfg.n_members, shuffle=True, random_state=7).split(Xtr, cells[tr])):
        Xa, ya, ca, _ = augment(Xtr[mi], D.y[tr][mi], cells[tr][mi], cfg.aug_mode, cfg.aug_ratio, k)
        m, _ = train_member(Xa, ya, ca, Xva, D.y[va], gva, cfg, seed=k); models.append(m)

    def f(key):
        def g(X):
            with torch.no_grad():
                outs = [m(torch.from_numpy(np.asarray(X, dtype=np.float32))) for m in models]
            return np.mean([(o["probabilities"][:, 1] if key == "p" else o["uncertainty"]).numpy() for o in outs], 0)
        return g

    bg = shap.kmeans(Xtr, 50).data
    res = {}
    for key in ("p", "u"):
        ex = shap.PermutationExplainer(f(key), shap.maskers.Independent(bg), seed=0)
        e = ex(Xte, max_evals=2 * Xte.shape[1] + 1, silent=True)
        uf = list(dict.fromkeys(fams))
        vals = np.stack([e.values[:, [i for i, x in enumerate(fams) if x == u]].sum(1) for u in uf], 1)
        imp = pd.Series(np.abs(vals).mean(0), index=uf).sort_values(ascending=False)
        imp.to_csv(os.path.join(OUT, f"shap_importance_{key}.csv"), header=["mean_abs_shap"])
        res[key] = imp
        res[f"{key}_additivity"] = float(np.abs(e.values.sum(1) + e.base_values - f(key)(Xte)).max())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), layout="constrained")
    for ax, key, lab, c in ((axes[0], "p", "Predicted attrition risk  p", SERIES[0]),
                            (axes[1], "u", "Evidential uncertainty  u", SERIES[1])):
        t = res[key].iloc[:12][::-1]
        ax.barh(t.index, t.values, color=c, height=0.6); ax.set_title(lab); ax.set_xlabel("mean |SHAP|")
        ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    fig.suptitle(f"FAENet-v3 ensemble: drivers of risk vs. uncertainty (test fold, {len(te)} employees)", fontsize=8.5, color=INK2)
    save(fig, os.path.join(OUT, "shap_global"))
    summ = {"top_risk": res["p"].head(10).round(4).to_dict(), "top_uncertainty": res["u"].head(10).round(4).to_dict(),
            "additivity_max_err": {"p": res["p_additivity"], "u": res["u_additivity"]}, "n_explained": int(len(te))}
    json.dump(summ, open(os.path.join(OUT, "shap_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
