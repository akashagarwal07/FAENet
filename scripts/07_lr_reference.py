"""Reference point only (not part of FAENet): logistic regression on exactly the v3 CV splits,
same features (engineered, no Age/Gender), pooled out-of-fold scoring per repeat."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from faenet.preprocessing import build_preprocessor
from faenet.data import ROOT, get_data
from faenet.core import N_FOLDS, N_REPEATS, add_engineered, cv_split, feature_columns

D = get_data("ibm"); FR = add_engineered(D.frame); num, cat = feature_columns(True)
out = {}
for C in (0.03, 0.1, 0.3, 1.0):
    roc, pr = [], []
    for r in range(N_REPEATS):
        p = np.full(len(D.y), np.nan)
        for f in range(N_FOLDS):
            tr, va, te = cv_split(D.y, D.groups, r, f)
            fit = np.r_[tr, va]   # LR needs no early stopping: train on train+val
            pre = build_preprocessor(num, cat).fit(FR.iloc[fit][num + cat])
            m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(pre.transform(FR.iloc[fit][num + cat]), D.y[fit])
            p[te] = m.predict_proba(pre.transform(FR.iloc[te][num + cat]))[:, 1]
        roc.append(roc_auc_score(D.y, p)); pr.append(average_precision_score(D.y, p))
    out[str(C)] = {"roc_auc": float(np.mean(roc)), "pr_auc": float(np.mean(pr))}
json.dump(out, open(os.path.join(ROOT, "results", "metrics", "lr_reference.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
