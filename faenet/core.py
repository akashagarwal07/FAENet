"""FAENet core (IBM): repeated-CV protocol, cross-fitted evidential ensemble, train-only synthetic
augmentation, Group DRO, fairness-aware checkpoint selection, and a learned review (deferral) head.

Leakage rules enforced here:
  * preprocessing, augmentation, class weights are fitted on the member's TRAINING rows only;
  * synthetic rows are never placed in validation or test;
  * thresholds / calibrators / selection use validation predictions only.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import SPECS
from .evidential_loss import adjusted_alpha, edl_mse_fit, kl_coefficient, kl_to_uniform_dirichlet
from .model import FAENet
from .preprocessing import build_preprocessor

N_FOLDS, N_REPEATS = 5, 5

# ----------------------------------------------------------------------------- features
ENGINEERED = ["log_MonthlyIncome", "tenure_ratio", "role_ratio", "manager_ratio", "promo_ratio",
              "income_per_level", "companies_per_year", "satisfaction_sum", "years_before_company"]


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic row-wise features (no fitted statistics, so no leakage)."""
    d = df.copy()
    tw = d["TotalWorkingYears"].clip(lower=0); yc = d["YearsAtCompany"].clip(lower=0)
    d["log_MonthlyIncome"] = np.log1p(d["MonthlyIncome"])
    d["tenure_ratio"] = yc / (tw + 1)
    d["role_ratio"] = d["YearsInCurrentRole"] / (yc + 1)
    d["manager_ratio"] = d["YearsWithCurrManager"] / (yc + 1)
    d["promo_ratio"] = d["YearsSinceLastPromotion"] / (yc + 1)
    d["income_per_level"] = d["MonthlyIncome"] / d["JobLevel"]
    d["companies_per_year"] = d["NumCompaniesWorked"] / (tw + 1)
    d["satisfaction_sum"] = (d["EnvironmentSatisfaction"] + d["JobSatisfaction"] + d["RelationshipSatisfaction"]
                             + d["WorkLifeBalance"] + d["JobInvolvement"])
    d["years_before_company"] = (tw - yc).clip(lower=0)
    return d


def feature_columns(engineered: bool, include_sensitive: bool = False):
    spec = SPECS["ibm"]
    num = list(spec.numeric) + (ENGINEERED if engineered else []) + (["Age"] if include_sensitive else [])
    cat = list(spec.categorical) + (["Gender"] if include_sensitive else [])
    return num, cat


# ----------------------------------------------------------------------------- splits
def cell_labels(y, groups):
    return np.array([f"{a}|{b}|{c}" for a, b, c in zip(y, groups["gender"], groups["age_band"])])


def cv_split(y, groups, repeat, fold, val_frac=0.1875):
    """Test = one stratified fold of repeat `repeat`; validation = 18.75% of the rest (15% of all)."""
    strata = cell_labels(y, groups)
    idx = np.arange(len(y))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=1000 + repeat)
    tr_all, te = list(skf.split(idx, strata))[fold]
    tr, va = train_test_split(tr_all, test_size=val_frac, stratify=strata[tr_all], random_state=100 * repeat + fold)
    assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)
    return np.sort(tr), np.sort(va), np.sort(te)


# ----------------------------------------------------------------------------- augmentation
def augment(X, y, cells_arr, mode: str, ratio: float, seed: int):
    """Train-only synthetic minority generation with SMOTE.
    mode 'smote': leavers oversampled to `ratio` x stayers overall.
    mode 'cell_smote': within every gender x age-band cell, leavers oversampled to `ratio` x that cell's
                       stayers (gives the small 40+ / female cells as much synthetic support as the rest).
    Returns X_aug, y_aug, cell labels (synthetic rows inherit their cell), and an is_synthetic mask."""
    from imblearn.over_sampling import SMOTE
    if mode == "none":
        return X, y, cells_arr, np.zeros(len(y), bool)
    newX, newy, newc = [X], [y], [cells_arr]
    rng_seed = seed
    if mode == "smote":
        n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
        target = int(ratio * n_neg)
        if target > n_pos:
            sm = SMOTE(sampling_strategy={1: target}, k_neighbors=min(5, n_pos - 1), random_state=rng_seed)
            Xs, ys = sm.fit_resample(X, y)
            Xn = Xs[len(X):]; cn = np.empty(len(Xn), dtype=object)
            # cell of a synthetic row = cell of its nearest real leaver
            pos = np.where(y == 1)[0]
            nn_idx = pos[np.argmin(((Xn[:, None, :] - X[pos][None]) ** 2).sum(-1), axis=1)]
            cn[:] = cells_arr[nn_idx]
            newX.append(Xn); newy.append(ys[len(X):]); newc.append(cn.astype(cells_arr.dtype))
    elif mode == "cell_smote":
        base = np.array([c.split("|", 1)[1] for c in cells_arr])  # gender|age
        for cell in sorted(set(base)):
            m = base == cell
            Xc, yc = X[m], y[m]
            n_pos, n_neg = (yc == 1).sum(), (yc == 0).sum()
            target = int(ratio * n_neg)
            if n_pos < 2 or target <= n_pos:
                continue
            sm = SMOTE(sampling_strategy={1: target}, k_neighbors=min(5, n_pos - 1), random_state=rng_seed)
            Xs, ys = sm.fit_resample(Xc, yc)
            Xn = Xs[len(Xc):]
            newX.append(Xn); newy.append(ys[len(Xc):]); newc.append(np.array([f"1|{cell}"] * len(Xn)))
    else:
        raise ValueError(mode)
    Xa = np.concatenate(newX).astype(np.float32); ya = np.concatenate(newy).astype(int)
    ca = np.concatenate(newc)
    syn = np.r_[np.zeros(len(y), bool), np.ones(len(ya) - len(y), bool)]
    return Xa, ya, ca, syn


# ----------------------------------------------------------------------------- training
@dataclass
class Cfg:
    hidden_dims: tuple = (32, 16)
    dropout: float = 0.1
    lr: float = 3e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 15
    anneal_epochs: int = 10
    lambda_r: float = 0.5
    cw_power: float = 1.0           # class weight = (N / 2N_c) ** cw_power ; 0 = none
    engineered: bool = False
    aug_mode: str = "none"          # none | smote | cell_smote
    aug_ratio: float = 0.5
    lambda_f: float = 0.0           # soft-TPR equal-opportunity penalty (full training positives)
    dro_eta: float = 0.0            # >0 enables Group DRO over gender x age x class cells
    dro_adj: float = 0.0            # Sagawa et al. generalisation adjustment C / sqrt(n_g)
    sel_kappa: float = 0.0          # >0: checkpoint score = val PR-AUC - kappa * val soft-TPR gap
    n_members: int = 5
    min_pos: int = 3


def cfg_from(d: dict) -> "Cfg":
    """Build a Cfg from a JSON dict (as stored in results/variants.json)."""
    return Cfg(**{**d, "hidden_dims": tuple(d["hidden_dims"])})


def _soft_gap(p1, y, gid_list):
    """Summed (over attributes) max-min soft TPR on positives; numpy, for selection."""
    total = 0.0
    for gid in gid_list:
        rates = [p1[(gid == g) & (y == 1)].mean() for g in np.unique(gid) if ((gid == g) & (y == 1)).sum() >= 3]
        if len(rates) > 1:
            total += max(rates) - min(rates)
    return total


def _soft_tpr_penalty(p_pos, gids_pos, n_groups, refs, min_pos):
    loss = torch.zeros(())
    for gid, ng, ref in zip(gids_pos, n_groups, refs):
        rates, ok = [], []
        for g in range(ng):
            m = gid == g
            ok.append(int(m.sum()) >= min_pos)
            rates.append(p_pos[m].mean() if m.any() else torch.zeros(()))
        if not ok[ref]:
            continue
        for g in range(ng):
            if g != ref and ok[g]:
                loss = loss + (rates[g] - rates[ref]) ** 2
    return loss


def train_member(Xtr, ytr, cells_tr, Xva, yva, gva, cfg: Cfg, seed: int):
    """One evidential network. cells_tr: 'y|gender|age' per training row (synthetic rows included)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    X = torch.from_numpy(Xtr); Y = torch.from_numpy(ytr).long()
    counts = np.bincount(ytr, minlength=2)
    cw = torch.tensor((len(ytr) / (2 * counts)) ** cfg.cw_power, dtype=torch.float32)

    gender = np.array([c.split("|")[1] for c in cells_tr]); age = np.array([c.split("|")[2] for c in cells_tr])
    g_names = sorted(set(gender)); a_names = sorted(set(age))
    gid = torch.tensor([g_names.index(v) for v in gender]); aid = torch.tensor([a_names.index(v) for v in age])
    refs = [g_names.index("Male") if "Male" in g_names else 0, a_names.index("30-39") if "30-39" in a_names else 0]
    pos = Y == 1
    Xpos, gpos, apos = X[pos], gid[pos], aid[pos]

    cell_names = sorted(set(cells_tr)); cell_id = torch.tensor([cell_names.index(c) for c in cells_tr])
    n_cells = len(cell_names)
    cell_n = torch.bincount(cell_id, minlength=n_cells).float()
    q = torch.ones(n_cells) / n_cells

    gva_ids = [np.array(gva["gender"]), np.array(gva["age_band"])]
    model = FAENet(X.shape[1], cfg.hidden_dims, cfg.dropout, "evidential")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(seed)
    best, best_score, bad, best_ep = None, -np.inf, 0, -1
    Xva_t = torch.from_numpy(Xva)
    for ep in range(cfg.max_epochs):
        model.train()
        lam_r = kl_coefficient(ep, cfg.anneal_epochs, cfg.lambda_r)
        for idx in torch.randperm(len(Y), generator=gen).split(cfg.batch_size):
            out = model(X[idx]); yb = Y[idx]
            y1h = F.one_hot(yb, 2).float()
            per = edl_mse_fit(out["alpha"], y1h) + lam_r * kl_to_uniform_dirichlet(adjusted_alpha(out["alpha"], y1h))
            if cfg.dro_eta > 0:
                cb = cell_id[idx]
                gl = torch.zeros(n_cells); present = torch.zeros(n_cells, dtype=torch.bool)
                for c in cb.unique():
                    m = cb == c; gl[c] = per[m].mean(); present[c] = True
                adj = gl.detach() + cfg.dro_adj / torch.sqrt(cell_n)
                q = q * torch.exp(cfg.dro_eta * adj * present.float()); q = q / q.sum()
                loss = (q * gl).sum()
            else:
                loss = (per * cw[yb]).mean()
            if cfg.lambda_f > 0:
                pp = model(Xpos)["probabilities"][:, 1]
                loss = loss + cfg.lambda_f * _soft_tpr_penalty(pp, [gpos, apos], [len(g_names), len(a_names)], refs, cfg.min_pos)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep + 1 < cfg.anneal_epochs:
            continue
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t)["probabilities"][:, 1].numpy()
        score = average_precision_score(yva, pv)
        if cfg.sel_kappa > 0:
            score -= cfg.sel_kappa * _soft_gap(pv, yva, gva_ids)
        if score > best_score + 1e-6:
            best_score, bad, best_ep = score, 0, ep
            best = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best); model.eval()
    return model, best_ep


@torch.no_grad()
def _pred(model, X):
    out = model(torch.from_numpy(X))
    return out["probabilities"][:, 1].numpy(), out["uncertainty"].numpy(), out["embedding"].numpy()


# ----------------------------------------------------------------------------- learned review head
class DeferralHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_deferral(F_tr, err_tr, F_va, err_va, seed):
    """Small network predicting P(classifier is wrong) from [features, logit p, u]; trained on
    cross-fitted (out-of-fold) training predictions, early-stopped on validation error-AUROC."""
    torch.manual_seed(seed)
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-6
    A = torch.from_numpy(((F_tr - mu) / sd).astype(np.float32)); E = torch.from_numpy(err_tr.astype(np.float32))
    Av = torch.from_numpy(((F_va - mu) / sd).astype(np.float32))
    h = DeferralHead(A.shape[1]); opt = torch.optim.AdamW(h.parameters(), lr=3e-3, weight_decay=1e-2)
    pw = torch.tensor(max(1.0, (1 - err_tr.mean()) / max(err_tr.mean(), 1e-3)))
    gen = torch.Generator().manual_seed(seed)
    best, best_auc, bad = copy.deepcopy(h.state_dict()), -1, 0
    for ep in range(150):
        h.train()
        for idx in torch.randperm(len(E), generator=gen).split(64):
            loss = F.binary_cross_entropy_with_logits(h(A[idx]), E[idx], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        h.eval()
        with torch.no_grad():
            sv = h(Av).numpy()
        auc = roc_auc_score(err_va, sv) if 0 < err_va.sum() < len(err_va) else 0.5
        if auc > best_auc + 1e-6:
            best_auc, bad, best = auc, 0, copy.deepcopy(h.state_dict())
        else:
            bad += 1
            if bad >= 15:
                break
    h.load_state_dict(best); h.eval()
    return lambda Fx: torch.sigmoid(h(torch.from_numpy(((Fx - mu) / sd).astype(np.float32)))).detach().numpy()


def f1_threshold(y, p):
    grid = np.linspace(0.01, 0.99, 197); npos = (y == 1).sum()
    f1 = [2 * ((p >= t) & (y == 1)).sum() / ((p >= t).sum() + npos + 1e-12) for t in grid]
    return float(grid[int(np.argmax(f1))])


# ----------------------------------------------------------------------------- one CV split
def run_split(data, frame, repeat, fold, cfg: Cfg, with_deferral=True, members=None, holdout_mask=None):
    """Train a cross-fitted ensemble on one CV split; return val/test predictions (+ train OOF)."""
    y, groups = data.y, data.groups
    if holdout_mask is None:
        tr, va, te = cv_split(y, groups, repeat, fold)
    else:  # H5: held-out group never seen; split the remainder
        rest = np.where(~holdout_mask)[0]
        sub_groups = {k: v[rest] for k, v in groups.items()}
        tr_r, va_r, te_r = cv_split(y[rest], sub_groups, repeat, fold)
        tr, va, te = rest[tr_r], rest[va_r], rest[te_r]
    num, cat = feature_columns(cfg.engineered)
    pre = build_preprocessor(num, cat).fit(frame.iloc[tr][num + cat])
    T = lambda ix: pre.transform(frame.iloc[ix][num + cat]).astype(np.float32)
    Xtr, Xva, Xte = T(tr), T(va), T(te)
    ho = np.where(holdout_mask)[0] if holdout_mask is not None else np.array([], int)
    Xho = T(ho) if len(ho) else None
    cells = cell_labels(y, groups)
    gva = {a: groups[a][va] for a in ("gender", "age_band")}
    n_mem = members or cfg.n_members
    inner = StratifiedKFold(n_mem, shuffle=True, random_state=7 + 100 * repeat + fold) if n_mem > 1 else None
    oof_p = np.full(len(tr), np.nan); oof_u = np.full(len(tr), np.nan)
    P_va, U_va, P_te, U_te, epochs, P_ho, U_ho = [], [], [], [], [], [], []
    splits = list(inner.split(Xtr, cells[tr])) if inner else [(np.arange(len(tr)), np.array([], int))]
    for k, (mi, oi) in enumerate(splits):
        seed = 10_000 * repeat + 100 * fold + k
        Xa, ya, ca, _ = augment(Xtr[mi], y[tr][mi], cells[tr][mi], cfg.aug_mode, cfg.aug_ratio, seed)
        model, ep = train_member(Xa, ya, ca, Xva, y[va], gva, cfg, seed)
        epochs.append(ep)
        pv, uv, _ = _pred(model, Xva); pt, ut, _ = _pred(model, Xte)
        P_va.append(pv); U_va.append(uv); P_te.append(pt); U_te.append(ut)
        if Xho is not None:
            ph, uh, _ = _pred(model, Xho); P_ho.append(ph); U_ho.append(uh)
        if len(oi):
            po, uo, _ = _pred(model, Xtr[oi]); oof_p[oi] = po; oof_u[oi] = uo
    P_va, U_va, P_te, U_te = map(np.array, (P_va, U_va, P_te, U_te))
    res = {"idx": {"train": tr, "val": va, "test": te}, "epochs": epochs,
           "val": {"p": P_va.mean(0), "u": U_va.mean(0), "dis": P_va.var(0), "y": y[va]},
           "test": {"p": P_te.mean(0), "u": U_te.mean(0), "dis": P_te.var(0), "y": y[te]},
           "member_test_p": P_te}
    if Xho is not None:
        P_ho, U_ho = np.array(P_ho), np.array(U_ho)
        res["heldout"] = {"p": P_ho.mean(0), "u": U_ho.mean(0), "dis": P_ho.var(0), "y": y[ho], "idx": ho}
    if with_deferral and n_mem > 1:
        ty = f1_threshold(y[va], res["val"]["p"])
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        err_tr = ((oof_p >= ty).astype(int) != y[tr]).astype(int)
        err_va = ((res["val"]["p"] >= ty).astype(int) != y[va]).astype(int)
        F_tr = np.c_[Xtr, lg(oof_p), oof_u]
        F_va = np.c_[Xva, lg(res["val"]["p"]), res["val"]["u"]]
        F_te = np.c_[Xte, lg(res["test"]["p"]), res["test"]["u"]]
        f = train_deferral(F_tr, err_tr, F_va, err_va, seed=repeat * 100 + fold)
        res["val"]["defer"] = f(F_va); res["test"]["defer"] = f(F_te)
        if Xho is not None:
            res["heldout"]["defer"] = f(np.c_[Xho, lg(res["heldout"]["p"]), res["heldout"]["u"]])
        res["train_oof"] = {"p": oof_p, "u": oof_u, "y": y[tr]}
    for s in ("val", "test"):
        ix = res["idx"][s]
        res[s]["groups"] = {a: groups[a][ix] for a in groups}
    return res
