"""Splitting and train-only preprocessing for FAENet."""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import LoadedData


def stratification_labels(data: LoadedData) -> np.ndarray:
    """Joint strata: target x fairness attributes (so every split contains positives of every group)."""
    parts = [data.y.astype(str)] + [data.groups[a] for a in data.spec.fairness_attrs]
    return np.array(["|".join(t) for t in zip(*parts)])


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    seed: int


def make_split(data: LoadedData, seed: int, frac=(0.70, 0.15, 0.15)) -> Split:
    idx = np.arange(len(data.y))
    strata = stratification_labels(data)
    tr, rest = train_test_split(idx, train_size=frac[0], stratify=strata, random_state=seed)
    rel_val = frac[1] / (frac[1] + frac[2])
    va, te = train_test_split(rest, train_size=rel_val, stratify=strata[rest], random_state=seed)
    assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0
    return Split(np.sort(tr), np.sort(va), np.sort(te), seed)


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)], remainder="drop")


@dataclass
class Prepared:
    X: dict[str, np.ndarray]          # split -> features
    y: dict[str, np.ndarray]
    groups: dict[str, dict[str, np.ndarray]]  # split -> attr -> labels
    feature_names: list[str]
    feature_families: list[str]       # original column each transformed feature came from
    class_weights: np.ndarray         # [w_0, w_1] from training split
    preprocessor: ColumnTransformer
    input_columns: list[str]


def prepare(data: LoadedData, split: Split, include_sensitive: bool = False) -> Prepared:
    spec = data.spec
    numeric = list(spec.numeric)
    categorical = list(spec.categorical)
    if include_sensitive:  # secondary experiment only
        numeric = numeric + [c for c in spec.sensitive_cols if pd.api.types.is_numeric_dtype(data.frame[c])]
        categorical = categorical + [c for c in spec.sensitive_cols
                                     if not pd.api.types.is_numeric_dtype(data.frame[c])]
    cols = numeric + categorical
    assert spec.target not in cols
    pre = build_preprocessor(numeric, categorical)
    f = data.frame
    pre.fit(f.iloc[split.train][cols])  # fit on training partition only
    X = {k: pre.transform(f.iloc[ix][cols]).astype(np.float32)
         for k, ix in (("train", split.train), ("val", split.val), ("test", split.test))}
    y = {k: data.y[ix] for k, ix in (("train", split.train), ("val", split.val), ("test", split.test))}
    groups = {k: {a: v[ix] for a, v in data.groups.items()}
              for k, ix in (("train", split.train), ("val", split.val), ("test", split.test))}

    names = list(pre.get_feature_names_out())
    families = []
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_map = {}
    for col, cats in zip(categorical, ohe.categories_):
        for c in cats:
            cat_map[f"cat__{col}_{c}"] = col
    for n in names:
        families.append(n.split("__", 1)[1] if n.startswith("num__") else cat_map[n])

    ytr = y["train"]
    counts = np.bincount(ytr, minlength=2)
    class_weights = len(ytr) / (2 * counts)   # w_c = N / (K * N_c)
    return Prepared(X, y, groups, names, families, class_weights.astype(np.float32), pre, cols)


def save_preprocessor(prep: Prepared, path: str) -> None:
    joblib.dump({"preprocessor": prep.preprocessor, "input_columns": prep.input_columns,
                 "feature_names": prep.feature_names, "feature_families": prep.feature_families,
                 "class_weights": prep.class_weights}, path)
