"""
Task graders for the OpenEnv DataCleaning environment.
Each grader takes the current DataFrame + metadata and returns a float in [0.0, 1.0].
Graders are deterministic and reproducible.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _null_score(df: pd.DataFrame, columns: List[str]) -> float:
    """Fraction of non-null cells across target columns."""
    total = len(df) * len(columns)
    if total == 0:
        return 1.0
    non_null = sum(df[c].notna().sum() for c in columns if c in df.columns)
    return non_null / total


# ---------------------------------------------------------------------------
# Task 1 — Easy Grader: Null Value Imputation
# ---------------------------------------------------------------------------

def grade_task_easy(df: pd.DataFrame, meta: Dict[str, Any]) -> Tuple[float, Dict]:
    """
    Scores [0, 1] based on:
    - 60% → fraction of nulls resolved in target columns
    - 20% → numeric columns have reasonable imputed values (within 3σ)
    - 20% → no new columns were dropped or corrupted
    """
    target_cols = meta.get("null_columns", [])
    present     = [c for c in target_cols if c in df.columns]

    # Component 1: null resolution
    null_res = _null_score(df, present)

    # Component 2: value sanity for numerics
    sanity = 1.0
    numeric_targets = ["age", "income", "loyalty_score"]
    sanity_checks = {
        "age":           (18, 75),
        "income":        (0, 200_000),
        "loyalty_score": (0, 100),
    }
    penalties = 0
    for col, (lo, hi) in sanity_checks.items():
        if col in df.columns:
            bad = df[col].dropna()
            out = ((bad < lo) | (bad > hi)).sum()
            if len(bad) > 0:
                penalties += out / len(bad)
    sanity = max(0.0, 1.0 - penalties / max(len(sanity_checks), 1))

    # Component 3: structure integrity (all expected columns still present)
    expected = ["customer_id", "age", "income", "city", "loyalty_score", "email", "signup_date"]
    struct_ok = all(c in df.columns for c in expected)

    score = 0.6 * null_res + 0.2 * sanity + 0.2 * (1.0 if struct_ok else 0.0)
    return round(float(np.clip(score, 0, 1)), 4), {
        "null_resolution": null_res,
        "value_sanity":    sanity,
        "structure_ok":    struct_ok,
    }


# ---------------------------------------------------------------------------
# Task 2 — Medium Grader: Type Coercion & Format Standardisation
# ---------------------------------------------------------------------------

_PHONE_RE  = re.compile(r"^\d{10}$")
_DATE_RE   = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EMAIL_RE  = re.compile(r"^[^@]+@[^@]+\.[^@]+$")

STANDARD_CATEGORIES = {"Category A", "Category B"}


def _price_score(series: pd.Series) -> float:
    """Fraction of values castable to float > 0."""
    good = 0
    for v in series.dropna():
        try:
            f = float(str(v).replace("$", "").replace(",", "").replace(" USD", "").strip())
            if f > 0:
                good += 1
        except Exception:
            pass
    return good / max(len(series), 1)


def _phone_score(series: pd.Series) -> float:
    return series.dropna().apply(lambda v: bool(_PHONE_RE.match(str(v)))).mean()


def _date_score(series: pd.Series) -> float:
    return series.dropna().apply(lambda v: bool(_DATE_RE.match(str(v)))).mean()


def _category_score(series: pd.Series) -> float:
    return series.dropna().apply(lambda v: v in STANDARD_CATEGORIES).mean()


def _qty_score(series: pd.Series) -> float:
    good = 0
    for v in series.dropna():
        try:
            i = int(float(str(v)))
            if i > 0:
                good += 1
        except Exception:
            pass
    return good / max(len(series), 1)


def grade_task_medium(df: pd.DataFrame, meta: Dict[str, Any]) -> Tuple[float, Dict]:
    """
    Scores based on per-column format compliance, weighted equally.
    """
    breakdown = {}

    breakdown["price"]    = _price_score(df["price"])    if "price"    in df.columns else 0.0
    breakdown["quantity"] = _qty_score(df["quantity"])   if "quantity" in df.columns else 0.0
    breakdown["phone"]    = _phone_score(df["phone"])    if "phone"    in df.columns else 0.0
    breakdown["date"]     = _date_score(df["sale_date"]) if "sale_date" in df.columns else 0.0
    breakdown["category"] = _category_score(df["category"]) if "category" in df.columns else 0.0

    score = sum(breakdown.values()) / len(breakdown)
    return round(float(np.clip(score, 0, 1)), 4), breakdown


# ---------------------------------------------------------------------------
# Task 3 — Hard Grader: Multi-Column Constraint Validation & Repair
# ---------------------------------------------------------------------------

def _check_age_range(df: pd.DataFrame) -> float:
    col = df["age"].dropna()
    if len(col) == 0:
        return 0.0
    return ((col >= 18) & (col <= 99)).sum() / len(col)


def _check_salary_positive(df: pd.DataFrame) -> float:
    col = df["salary"].dropna()
    if len(col) == 0:
        return 0.0
    return (col > 0).sum() / len(col)


def _check_date_order(df: pd.DataFrame) -> float:
    valid = 0
    total = 0
    for _, row in df.iterrows():
        try:
            s = pd.Timestamp(row["contract_start"])
            e = pd.Timestamp(row["contract_end"])
            if e >= s:
                valid += 1
            total += 1
        except Exception:
            total += 1
    return valid / max(total, 1)


def _check_dept_valid(df: pd.DataFrame, valid_depts: List[str]) -> float:
    col = df["department"].dropna()
    if len(col) == 0:
        return 0.0
    return col.isin(valid_depts).sum() / len(col)


def _check_rating_range(df: pd.DataFrame) -> float:
    col = df["performance_rating"].dropna()
    if len(col) == 0:
        return 0.0
    return ((col >= 1) & (col <= 5)).sum() / len(col)


def _check_email_format(df: pd.DataFrame) -> float:
    col = df["email"].dropna()
    if len(col) == 0:
        return 0.0
    return col.apply(lambda v: bool(_EMAIL_RE.match(str(v)))).sum() / len(col)


CONSTRAINT_WEIGHTS = {
    "age_range":       0.15,
    "salary_positive": 0.15,
    "end_after_start": 0.20,
    "dept_valid":      0.20,
    "rating_range":    0.15,
    "email_format":    0.15,
}


def grade_task_hard(df: pd.DataFrame, meta: Dict[str, Any]) -> Tuple[float, Dict]:
    """
    Weighted average of per-constraint compliance scores.
    """
    valid_depts = meta.get("valid_departments", ["HR", "Engineering", "Sales", "Finance", "Marketing"])

    breakdown = {
        "age_range":       _check_age_range(df),
        "salary_positive": _check_salary_positive(df),
        "end_after_start": _check_date_order(df),
        "dept_valid":      _check_dept_valid(df, valid_depts),
        "rating_range":    _check_rating_range(df),
        "email_format":    _check_email_format(df),
    }

    score = sum(CONSTRAINT_WEIGHTS[k] * v for k, v in breakdown.items())
    return round(float(np.clip(score, 0, 1)), 4), breakdown
