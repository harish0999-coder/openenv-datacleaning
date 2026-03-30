"""
Dataset generators for the three OpenEnv DataCleaning tasks.
Each generator produces a reproducible dirty DataFrame + ground-truth metadata.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Task 1 — Easy: Null Value Imputation (Customer Dataset)
# ---------------------------------------------------------------------------

def generate_customer_dataset(seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates a 120-row customer table with realistic null patterns.
    Returns (dirty_df, ground_truth_meta).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    n = 120

    ages    = rng.integers(18, 75, size=n).astype(float)
    incomes = rng.normal(55_000, 15_000, size=n).round(2)
    cities  = rng.choice(["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"], size=n)
    scores  = rng.uniform(0, 100, size=n).round(1)
    emails  = [f"user{i}@example.com" for i in range(n)]
    signup  = pd.date_range("2020-01-01", periods=n, freq="3D").strftime("%Y-%m-%d").tolist()

    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age":         ages,
        "income":      incomes,
        "city":        cities,
        "loyalty_score": scores,
        "email":       emails,
        "signup_date": signup,
    })

    # Inject nulls
    null_mask_age     = rng.random(n) < 0.15
    null_mask_income  = rng.random(n) < 0.12
    null_mask_city    = rng.random(n) < 0.08
    null_mask_score   = rng.random(n) < 0.10

    df.loc[null_mask_age,    "age"]           = np.nan
    df.loc[null_mask_income, "income"]        = np.nan
    df.loc[null_mask_city,   "city"]          = np.nan
    df.loc[null_mask_score,  "loyalty_score"] = np.nan

    meta = {
        "task_id": "task_easy",
        "null_columns": ["age", "income", "city", "loyalty_score"],
        "strategies": {
            "age":           "median",
            "income":        "mean",
            "city":          "mode",
            "loyalty_score": "median",
        },
        "total_nulls": int(null_mask_age.sum() + null_mask_income.sum()
                           + null_mask_city.sum() + null_mask_score.sum()),
        "max_steps": 20,
    }
    return df, meta


# ---------------------------------------------------------------------------
# Task 2 — Medium: Type Coercion & Format Standardisation (Sales Dataset)
# ---------------------------------------------------------------------------

PHONE_FORMATS = [
    "{a}{b}{c}{d}",
    "({a}) {b}-{c}{d}",
    "+91-{a}-{b}-{c}{d}",
    "{a}.{b}.{c}{d}",
    "{a} {b} {c}{d}",
]

DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d %b %Y", "%B %d, %Y"]

def _random_phone(rng) -> str:
    a = rng.integers(6000, 9999)
    b = rng.integers(100, 999)
    c = rng.integers(10, 99)
    d = rng.integers(1000, 9999)
    fmt = random.choice(PHONE_FORMATS)
    return fmt.format(a=a, b=b, c=c, d=d)

def _random_date(rng, base: pd.Timestamp) -> str:
    offset = int(rng.integers(0, 730))
    dt = base + pd.Timedelta(days=offset)
    fmt = random.choice(DATE_FORMATS)
    return dt.strftime(fmt)


def generate_sales_dataset(seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates a 100-row sales table with messy formats and wrong dtypes.
    Returns (dirty_df, ground_truth_meta).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    n = 100
    base = pd.Timestamp("2022-01-01")

    products   = rng.choice(["Widget A", "Widget B", "Gadget X", "Gadget Y"], size=n)
    quantities = rng.integers(1, 50, size=n)
    prices_raw = [str(round(float(rng.uniform(10, 500)), 2)) for _ in range(n)]
    # Inject price format issues
    for i in random.sample(range(n), 20):
        prices_raw[i] = f"${prices_raw[i]}"
    for i in random.sample(range(n), 10):
        prices_raw[i] = prices_raw[i] + " USD"

    phones     = [_random_phone(rng) for _ in range(n)]
    dates      = [_random_date(rng, base) for _ in range(n)]
    categories = rng.choice(["A", "B", "a", "b", "Category A", "Cat B", "type_a"], size=n)

    # Quantities stored as strings with occasional typos
    qty_str = [str(q) for q in quantities]
    for i in random.sample(range(n), 8):
        qty_str[i] = qty_str[i] + ".0"

    df = pd.DataFrame({
        "sale_id":   range(1, n + 1),
        "product":   products,
        "quantity":  qty_str,
        "price":     prices_raw,
        "phone":     phones,
        "sale_date": dates,
        "category":  categories,
    })

    category_map = {
        "A": "Category A", "B": "Category B",
        "a": "Category A", "b": "Category B",
        "Category A": "Category A", "Cat B": "Category B",
        "type_a": "Category A",
    }

    meta = {
        "task_id": "task_medium",
        "format_issues": {
            "price":     "strip currency symbols/text, cast to float",
            "quantity":  "cast to int",
            "phone":     "normalize to 10-digit numeric string",
            "sale_date": "normalize to YYYY-MM-DD",
            "category":  "map to standard values",
        },
        "category_map": category_map,
        "max_steps": 30,
    }
    return df, meta


# ---------------------------------------------------------------------------
# Task 3 — Hard: Multi-Column Constraint Validation & Repair
# ---------------------------------------------------------------------------

CONSTRAINTS = [
    {
        "id":          "age_range",
        "description": "age must be between 18 and 99",
        "column":      "age",
    },
    {
        "id":          "salary_positive",
        "description": "salary must be > 0",
        "column":      "salary",
    },
    {
        "id":          "end_after_start",
        "description": "contract_end must be after contract_start",
        "columns":     ["contract_start", "contract_end"],
    },
    {
        "id":          "dept_valid",
        "description": "department must be one of: HR, Engineering, Sales, Finance, Marketing",
        "column":      "department",
    },
    {
        "id":          "rating_range",
        "description": "performance_rating must be between 1 and 5",
        "column":      "performance_rating",
    },
    {
        "id":          "email_format",
        "description": "email must contain '@' and '.'",
        "column":      "email",
    },
]

VALID_DEPTS = ["HR", "Engineering", "Sales", "Finance", "Marketing"]

def generate_hr_dataset(seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates a 150-row HR table with multiple constraint violations + outliers.
    Returns (dirty_df, ground_truth_meta).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    n = 150

    ages       = rng.integers(18, 65, size=n).astype(float)
    salaries   = rng.normal(70_000, 20_000, size=n).round(2)
    depts      = rng.choice(VALID_DEPTS, size=n)
    ratings    = rng.integers(1, 6, size=n).astype(float)
    starts     = pd.date_range("2015-01-01", periods=n, freq="30D")
    ends       = starts + pd.to_timedelta(rng.integers(365, 1825, size=n), unit="D")
    emails     = [f"emp{i}@company.org" for i in range(n)]

    df = pd.DataFrame({
        "emp_id":          range(1, n + 1),
        "age":             ages,
        "salary":          salaries,
        "department":      depts,
        "performance_rating": ratings,
        "contract_start":  starts.strftime("%Y-%m-%d"),
        "contract_end":    ends.strftime("%Y-%m-%d"),
        "email":           emails,
    })

    # Inject violations
    # 1. Age out of range
    bad_age_idx = rng.choice(n, size=8, replace=False)
    df.loc[bad_age_idx, "age"] = rng.choice([-5, 120, 150, 0, 200], size=8)

    # 2. Negative salaries
    bad_sal_idx = rng.choice(n, size=6, replace=False)
    df.loc[bad_sal_idx, "salary"] = rng.uniform(-50_000, -1, size=6).round(2)

    # 3. End before start
    bad_date_idx = rng.choice(n, size=7, replace=False)
    for i in bad_date_idx:
        df.loc[i, "contract_end"] = (
            pd.Timestamp(df.loc[i, "contract_start"]) - pd.Timedelta(days=int(rng.integers(1, 365)))
        ).strftime("%Y-%m-%d")

    # 4. Invalid departments
    bad_dept_idx = rng.choice(n, size=10, replace=False)
    df.loc[bad_dept_idx, "department"] = rng.choice(
        ["Accounting", "IT", "Legal", "Ops", "R&D"], size=10
    )

    # 5. Rating out of range
    bad_rat_idx = rng.choice(n, size=5, replace=False)
    df.loc[bad_rat_idx, "performance_rating"] = rng.choice([0, 6, 7, -1], size=5).astype(float)

    # 6. Malformed emails
    bad_email_idx = rng.choice(n, size=8, replace=False)
    df.loc[bad_email_idx, "email"] = [
        "notemail", "missing_at.com", "no_dot@nodot",
        "emp@", "@company.org", "bademail", "x@", "nope.nope"
    ]

    meta = {
        "task_id": "task_hard",
        "constraints": CONSTRAINTS,
        "valid_departments": VALID_DEPTS,
        "max_steps": 40,
        "violation_counts": {
            "age_range":       int(len(bad_age_idx)),
            "salary_positive": int(len(bad_sal_idx)),
            "end_after_start": int(len(bad_date_idx)),
            "dept_valid":      int(len(bad_dept_idx)),
            "rating_range":    int(len(bad_rat_idx)),
            "email_format":    int(len(bad_email_idx)),
        },
    }
    return df, meta
