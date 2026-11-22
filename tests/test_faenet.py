"""Unit tests for the FAENet package."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.special import digamma, gammaln

from faenet.core import Cfg, _soft_tpr_penalty, add_engineered, augment, cell_labels, cv_split, train_member
from faenet.data import get_data
from faenet.evidential_loss import adjusted_alpha, edl_mse_fit, kl_to_uniform_dirichlet
from faenet.model import FAENet


def kl_ref(a):
    S = a.sum(); K = len(a)
    return gammaln(S) - gammaln(a).sum() - gammaln(K) + ((a - 1) * (digamma(a) - digamma(S))).sum()


def test_kl_matches_scipy_and_is_zero_at_uniform():
    A = np.random.default_rng(0).uniform(1, 30, size=(50, 2))
    ours = kl_to_uniform_dirichlet(torch.tensor(A, dtype=torch.float64)).numpy()
    assert np.allclose(ours, [kl_ref(a) for a in A], atol=1e-9)
    assert abs(float(kl_to_uniform_dirichlet(torch.ones(1, 2, dtype=torch.float64)))) < 1e-12


def test_adjusted_alpha_and_mse_fit():
    a = torch.tensor([[19.0, 1.0], [3.0, 7.0]]); y = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert torch.equal(adjusted_alpha(a, y), torch.tensor([[1.0, 1.0], [1.0, 7.0]]))
    p = np.array([0.75, 0.25]); ref = ((np.array([1, 0]) - p) ** 2 + p * (1 - p) / 5).sum()
    assert np.isclose(float(edl_mse_fit(torch.tensor([[3.0, 1.0]]), torch.tensor([[1.0, 0.0]]))), ref)


def test_model_outputs():
    torch.manual_seed(0); out = FAENet(12)(torch.randn(16, 12))
    assert (out["evidence"] >= 0).all() and torch.allclose(out["probabilities"].sum(1), torch.ones(16))
    assert ((out["uncertainty"] > 0) & (out["uncertainty"] <= 1)).all()
    S = out["alpha"].sum(1); assert torch.allclose(out["uncertainty"], 2 / S)


def test_soft_tpr_penalty_skips_small_groups_and_has_gradient():
    p = torch.tensor([0.9, 0.7, 0.3, 0.1, 0.5], requires_grad=True)
    gid = torch.tensor([0, 0, 1, 1, 2])           # group 2 has a single positive -> skipped (min_pos=2)
    loss = _soft_tpr_penalty(p, [gid], [3], [0], min_pos=2)
    assert np.isclose(loss.item(), (0.2 - 0.8) ** 2)
    loss.backward(); assert torch.isfinite(p.grad).all() and p.grad[4] == 0


def test_cv_split_disjoint_and_covers_everyone():
    d = get_data()
    seen = []
    for f in range(5):
        tr, va, te = cv_split(d.y, d.groups, 0, f)
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        seen.append(te)
    assert sorted(np.concatenate(seen).tolist()) == list(range(len(d.y)))


def test_engineered_features_finite_and_age_gender_not_inputs():
    d = get_data(); fr = add_engineered(d.frame)
    from faenet.core import ENGINEERED, feature_columns
    assert np.isfinite(fr[ENGINEERED].to_numpy()).all()
    num, cat = feature_columns(True)
    assert "Age" not in num and "Gender" not in cat


def test_augmentation_keeps_real_rows_and_adds_only_leavers():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)).astype(np.float32); y = (rng.random(200) < 0.15).astype(int)
    groups = {"gender": rng.choice(["Female", "Male"], 200), "age_band": rng.choice(["<30", "30-39", "40+"], 200)}
    c = cell_labels(y, groups)
    Xa, ya, ca, syn = augment(X, y, c, "smote", 0.5, 0)
    assert np.array_equal(Xa[:200], X) and (ya[syn] == 1).all() and syn.sum() > 0


def test_train_member_runs_and_returns_probabilities():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 6)).astype(np.float32); y = (X[:, 0] + rng.normal(size=300) > 1).astype(int)
    g = {"gender": rng.choice(["Female", "Male"], 300), "age_band": rng.choice(["<30", "30-39", "40+"], 300)}
    c = cell_labels(y, g)
    m, ep = train_member(X[:200], y[:200], c[:200], X[200:], y[200:], {k: v[200:] for k, v in g.items()},
                         Cfg(max_epochs=15, patience=3, lambda_f=1.0), seed=0)
    with torch.no_grad():
        p = m(torch.from_numpy(X[200:]))["probabilities"][:, 1].numpy()
    assert ((p > 0) & (p < 1)).all()
