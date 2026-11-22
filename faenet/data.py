"""IBM HR Analytics dataset: loading, schema checks and fairness-group metadata."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger("faenet.data")


@dataclass
class DatasetSpec:
    name: str
    target: str
    positive_value: object
    numeric: list[str]
    categorical: list[str]
    drop_ids: list[str] = field(default_factory=list)        # identifiers, always dropped
    drop_if_constant: list[str] = field(default_factory=list)  # dropped only after confirming constant
    sensitive_cols: list[str] = field(default_factory=list)  # kept out of X in the primary experiment
    aliases: dict[str, str] = field(default_factory=dict)    # file column -> canonical name (logged)
    deduplicate: bool = False
    # fairness attributes used in the training penalty: name -> reference group
    fairness_attrs: dict[str, str] = field(default_factory=dict)
    # additional attributes only audited (not penalised)
    audit_attrs: list[str] = field(default_factory=list)


IBM_NUMERIC = [
    "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]
IBM_CATEGORICAL = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus", "OverTime"]

SPECS = {
    "ibm": DatasetSpec(
        name="ibm", target="Attrition", positive_value="Yes",
        numeric=IBM_NUMERIC, categorical=IBM_CATEGORICAL,
        drop_ids=["EmployeeNumber"],
        drop_if_constant=["EmployeeCount", "Over18", "StandardHours"],
        sensitive_cols=["Age", "Gender"],
        fairness_attrs={"gender": "Male", "age_band": "30-39"},
        audit_attrs=["JobRole", "Department"],
    ),
}

AGE_BANDS = ["<30", "30-39", "40+"]


def age_band(age: pd.Series) -> pd.Series:
    """Project-defined age bands (not an original column)."""
    return pd.Series(np.select([age < 30, age < 40], ["<30", "30-39"], "40+"), index=age.index)


@dataclass
class LoadedData:
    spec: DatasetSpec
    frame: pd.DataFrame           # cleaned frame with features + target (+ sensitive cols)
    y: np.ndarray                 # 0/1 target, 1 = attrition
    groups: dict[str, np.ndarray]  # attribute name -> array of group labels (strings)
    report: dict


def load_dataset(name: str, path: str, verbose: bool = True) -> LoadedData:
    spec = SPECS[name]
    df = pd.read_csv(path)
    report: dict = {"path": path, "raw_shape": list(df.shape), "raw_columns": list(df.columns)}

    # Explicit, logged aliasing (never silent)
    renamed = {}
    for src, dst in spec.aliases.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
            renamed[src] = dst
    if renamed:
        log.warning("Renamed columns %s to canonical names", renamed)
    report["renamed_columns"] = renamed

    required = [spec.target] + spec.numeric + spec.categorical + spec.sensitive_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"[{name}] expected columns missing from {path}: {missing}. "
                       f"Found: {list(df.columns)}")

    report["missing_values"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    n_dup = int(df.duplicated().sum())
    report["duplicate_rows"] = n_dup
    if spec.deduplicate and n_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        log.warning("[%s] removed %d exact duplicate rows before splitting", name, n_dup)
    report["duplicates_removed"] = n_dup if spec.deduplicate else 0

    dropped = []
    for c in spec.drop_ids:
        if c in df.columns:
            dropped.append(c)
    for c in spec.drop_if_constant:
        if c in df.columns:
            if df[c].nunique(dropna=False) == 1:
                dropped.append(c)
            else:
                raise ValueError(f"[{name}] column {c} expected constant but has "
                                 f"{df[c].nunique()} values; refusing to drop silently")
    report["dropped_columns"] = dropped
    df = df.drop(columns=dropped)

    vals = set(df[spec.target].unique())
    if spec.positive_value not in vals:
        raise ValueError(f"[{name}] positive value {spec.positive_value!r} not in target values {vals}")
    y = (df[spec.target] == spec.positive_value).astype(int).to_numpy()

    groups: dict[str, np.ndarray] = {}
    if name == "ibm":
        groups["gender"] = df["Gender"].astype(str).to_numpy()
        groups["age_band"] = age_band(df["Age"]).to_numpy()
    for a in spec.fairness_attrs:
        if a not in groups:
            groups[a] = df[a].astype(str).to_numpy()
    for a in spec.audit_attrs:
        groups[a] = df[a].astype(str).to_numpy()

    report["final_shape"] = list(df.shape)
    report["n_positive"] = int(y.sum())
    report["positive_rate"] = float(y.mean())
    report["group_counts"] = {
        a: {g: {"n": int((v == g).sum()), "positives": int(y[v == g].sum())}
            for g in sorted(set(v))}
        for a, v in groups.items()
    }
    if verbose:
        print(f"[{name}] raw shape {report['raw_shape']} -> final {report['final_shape']}; "
              f"duplicates={n_dup} (removed={report['duplicates_removed']}); dropped={dropped}")
        print(f"[{name}] positives {report['n_positive']} ({report['positive_rate']:.1%}); "
              f"missing={report['missing_values'] or 'none'}")
    return LoadedData(spec=spec, frame=df, y=y, groups=groups, report=report)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
_CACHE: dict = {}


def get_data(name: str = "ibm") -> LoadedData:
    """Load (and cache) the IBM dataset from data/raw/."""
    if name not in _CACHE:
        _CACHE[name] = load_dataset(name, DATA_PATH, verbose=False)
    return _CACHE[name]
