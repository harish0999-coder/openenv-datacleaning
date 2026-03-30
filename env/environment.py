"""
DataCleaningEnv — Core environment class implementing the OpenEnv spec.
Manages state, applies actions, computes rewards.
"""

from __future__ import annotations

import re
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from env.models import (
    ActionType, CastTargetType, DataAction, DataObservation,
    DataReward, EnvironmentState, FillStrategy, FormatType,
    RewardBreakdown, StepResult, ColumnProfile, ConstraintViolation,
)
from env.datasets import (
    generate_customer_dataset,
    generate_sales_dataset,
    generate_hr_dataset,
)
from graders.graders import grade_task_easy, grade_task_medium, grade_task_hard

TASK_META = {
    "task_easy":   {"name": "Null Value Imputation",                      "difficulty": "easy",   "max_steps": 20},
    "task_medium": {"name": "Type Coercion & Format Standardization",     "difficulty": "medium", "max_steps": 30},
    "task_hard":   {"name": "Multi-Column Constraint Validation & Repair","difficulty": "hard",   "max_steps": 40},
}

GRADERS = {
    "task_easy":   grade_task_easy,
    "task_medium": grade_task_medium,
    "task_hard":   grade_task_hard,
}

GENERATORS = {
    "task_easy":   generate_customer_dataset,
    "task_medium": generate_sales_dataset,
    "task_hard":   generate_hr_dataset,
}

PHONE_CLEAN = re.compile(r"\D")
EMAIL_RE    = re.compile(r"^[^@]+@[^@]+\.[^@]+$")
DATE_RE     = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VALID_DEPTS = ["HR", "Engineering", "Sales", "Finance", "Marketing"]
_CATEGORY_MAP = {
    "A": "Category A", "B": "Category B",
    "a": "Category A", "b": "Category B",
    "Category A": "Category A", "Cat B": "Category B",
    "type_a": "Category A",
}


class DataCleaningEnv:
    """Full OpenEnv-compliant data cleaning environment."""

    def __init__(self, task_id: str = "task_easy", seed: int = 42):
        assert task_id in TASK_META, f"Unknown task: {task_id}"
        self.task_id     = task_id
        self.seed        = seed
        self._step       = 0
        self._ep_reward  = 0.0
        self._history: List[str] = []
        self._last_error: Optional[str] = None
        self._df: Optional[pd.DataFrame] = None
        self._meta: Dict[str, Any] = {}
        self._prev_score = 0.0

    # ------------------------------------------------------------------
    # OpenEnv core interface
    # ------------------------------------------------------------------

    def reset(self) -> DataObservation:
        """Reset to initial (dirty) state; returns first observation."""
        gen = GENERATORS[self.task_id]
        self._df, self._meta = gen(seed=self.seed)
        self._step      = 0
        self._ep_reward = 0.0
        self._history   = []
        self._last_error = None
        self._prev_score, _ = GRADERS[self.task_id](self._df, self._meta)
        return self._observe()

    def step(self, action: DataAction) -> StepResult:
        """Apply action; return (observation, reward, done, info)."""
        if self._df is None:
            raise RuntimeError("Call reset() before step().")

        self._step += 1
        self._last_error = None
        prev_score = self._prev_score

        try:
            action_desc = self._apply_action(action)
        except Exception as exc:
            self._last_error = str(exc)
            action_desc = f"[ERROR] {exc}"

        curr_score, breakdown = GRADERS[self.task_id](self._df, self._meta)
        self._prev_score = curr_score

        reward = self._compute_reward(
            prev_score, curr_score, action, self._last_error
        )
        self._ep_reward += reward.total

        done = (
            self._step >= TASK_META[self.task_id]["max_steps"]
            or curr_score >= 0.98
            or action.action_type == ActionType.SUBMIT
        )

        self._history.append(action_desc)
        if len(self._history) > 10:
            self._history = self._history[-10:]

        obs = self._observe(done=done)
        return StepResult(observation=obs, reward=reward, done=done, info={"breakdown": breakdown})

    def state(self) -> EnvironmentState:
        """Return current internal state snapshot."""
        if self._df is None:
            raise RuntimeError("Call reset() before state().")
        score, _ = GRADERS[self.task_id](self._df, self._meta)
        issues = self._count_issues()
        return EnvironmentState(
            task_id=self.task_id,
            step_number=self._step,
            quality_score=score,
            issues_remaining=issues,
            episode_reward=self._ep_reward,
            done=self._step >= TASK_META[self.task_id]["max_steps"],
            dataset_shape=[len(self._df), len(self._df.columns)],
            active_task=TASK_META[self.task_id],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observe(self, done: bool = False) -> DataObservation:
        score, _ = GRADERS[self.task_id](self._df, self._meta)
        profiles = self._column_profiles()
        violations = self._constraint_violations()
        issues = self._count_issues()
        return DataObservation(
            task_id=self.task_id,
            step_number=self._step,
            max_steps=TASK_META[self.task_id]["max_steps"],
            total_rows=len(self._df),
            total_columns=len(self._df.columns),
            column_profiles=profiles,
            constraint_violations=violations,
            recent_actions=list(self._history[-5:]),
            quality_score=round(float(score), 4),
            issues_remaining=issues,
            last_action_error=self._last_error,
            done=done,
        )

    def _column_profiles(self) -> List[ColumnProfile]:
        profiles = []
        for col in self._df.columns:
            s     = self._df[col]
            nc    = int(s.isna().sum())
            uc    = int(s.nunique())
            samp  = [v for v in s.dropna().head(3).tolist() if v == v]
            issues: List[str] = []
            if nc > 0:
                issues.append(f"{nc} null values")
            fmt_issues = self._detect_format_issues(col, s)
            issues += fmt_issues
            profiles.append(ColumnProfile(
                name=col, dtype=str(s.dtype),
                null_count=nc, null_pct=round(nc / max(len(s), 1), 3),
                unique_count=uc, sample_values=samp,
                has_format_issues=bool(fmt_issues),
                issues=issues,
            ))
        return profiles

    def _detect_format_issues(self, col: str, series: pd.Series) -> List[str]:
        issues = []
        name = col.lower()
        if "phone" in name:
            bad = series.dropna().apply(lambda v: not bool(re.match(r"^\d{10}$", str(v)))).sum()
            if bad:
                issues.append(f"{bad} non-standard phone formats")
        if "date" in name or "start" in name or "end" in name:
            bad = series.dropna().apply(lambda v: not bool(DATE_RE.match(str(v)))).sum()
            if bad:
                issues.append(f"{bad} non-standard date formats")
        if "price" in name or "salary" in name or "income" in name:
            bad = 0
            for v in series.dropna():
                try:
                    float(str(v).replace("$", "").replace(",", "").replace(" USD", "").strip())
                except Exception:
                    bad += 1
            if bad:
                issues.append(f"{bad} unparsable numeric values")
        return issues

    def _constraint_violations(self) -> List[ConstraintViolation]:
        violations = []
        if self.task_id == "task_hard":
            df = self._df
            # age
            bad = df[~((df["age"] >= 18) & (df["age"] <= 99))].index.tolist()
            if bad:
                violations.append(ConstraintViolation(
                    constraint_id="age_range",
                    description="age must be between 18 and 99",
                    affected_rows=bad[:10], severity="error"
                ))
            # salary
            bad = df[~(df["salary"] > 0)].index.tolist()
            if bad:
                violations.append(ConstraintViolation(
                    constraint_id="salary_positive",
                    description="salary must be > 0",
                    affected_rows=bad[:10], severity="error"
                ))
            # date order
            bad = []
            for i, row in df.iterrows():
                try:
                    if pd.Timestamp(row["contract_end"]) < pd.Timestamp(row["contract_start"]):
                        bad.append(i)
                except Exception:
                    pass
            if bad:
                violations.append(ConstraintViolation(
                    constraint_id="end_after_start",
                    description="contract_end must be after contract_start",
                    affected_rows=bad[:10], severity="error"
                ))
            # dept
            bad = df[~df["department"].isin(_VALID_DEPTS)].index.tolist()
            if bad:
                violations.append(ConstraintViolation(
                    constraint_id="dept_valid",
                    description="department must be valid",
                    affected_rows=bad[:10], severity="error"
                ))
            # rating
            bad = df[~((df["performance_rating"] >= 1) & (df["performance_rating"] <= 5))].index.tolist()
            if bad:
                violations.append(ConstraintViolation(
                    constraint_id="rating_range",
                    description="performance_rating must be 1-5",
                    affected_rows=bad[:10], severity="error"
                ))
            # email
            bad = df[~df["email"].apply(lambda v: bool(EMAIL_RE.match(str(v))))].index.tolist()
            if bad:
                violations.append(ConstraintViolation(
                    constraint_id="email_format",
                    description="email must contain '@' and '.'",
                    affected_rows=bad[:10], severity="warning"
                ))
        return violations

    def _count_issues(self) -> int:
        total = int(self._df.isna().sum().sum())
        for p in self._column_profiles():
            total += len(p.issues)
        return total

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _apply_action(self, action: DataAction) -> str:
        t = action.action_type

        if t == ActionType.FILL_NULLS:
            return self._fill_nulls(action)
        elif t == ActionType.DROP_ROWS:
            return self._drop_rows(action)
        elif t == ActionType.CAST_COLUMN:
            return self._cast_column(action)
        elif t == ActionType.RENAME_VALUE:
            return self._rename_value(action)
        elif t == ActionType.NORMALIZE_FORMAT:
            return self._normalize_format(action)
        elif t == ActionType.CLIP_VALUES:
            return self._clip_values(action)
        elif t == ActionType.DROP_DUPLICATES:
            return self._drop_duplicates(action)
        elif t == ActionType.FLAG_OUTLIERS:
            return self._flag_outliers(action)
        elif t == ActionType.REPAIR_CONSTRAINT:
            return self._repair_constraint(action)
        elif t == ActionType.SUBMIT:
            return "SUBMIT — episode ending"
        else:
            raise ValueError(f"Unknown action type: {t}")

    def _col(self, action: DataAction) -> str:
        if not action.column or action.column not in self._df.columns:
            raise ValueError(f"Column '{action.column}' not found.")
        return action.column

    def _fill_nulls(self, action: DataAction) -> str:
        col = self._col(action)
        s   = action.strategy or FillStrategy.MEAN
        null_before = self._df[col].isna().sum()

        if s == FillStrategy.MEAN:
            self._df[col] = self._df[col].fillna(self._df[col].mean())
        elif s == FillStrategy.MEDIAN:
            self._df[col] = self._df[col].fillna(self._df[col].median())
        elif s == FillStrategy.MODE:
            self._df[col] = self._df[col].fillna(self._df[col].mode()[0])
        elif s == FillStrategy.FFILL:
            self._df[col] = self._df[col].ffill()
        elif s == FillStrategy.BFILL:
            self._df[col] = self._df[col].bfill()
        elif s == FillStrategy.CONSTANT:
            self._df[col] = self._df[col].fillna(action.value)
        elif s == FillStrategy.DROP:
            self._df.dropna(subset=[col], inplace=True)
            self._df.reset_index(drop=True, inplace=True)

        null_after = self._df[col].isna().sum()
        return f"fill_nulls({col}, {s}) fixed {null_before - null_after} nulls"

    def _drop_rows(self, action: DataAction) -> str:
        before = len(self._df)
        if action.row_indices:
            self._df.drop(index=action.row_indices, errors="ignore", inplace=True)
            self._df.reset_index(drop=True, inplace=True)
        return f"drop_rows removed {before - len(self._df)} rows"

    def _cast_column(self, action: DataAction) -> str:
        col = self._col(action)
        tt  = action.target_type or CastTargetType.FLOAT

        if tt == CastTargetType.INT:
            self._df[col] = pd.to_numeric(self._df[col], errors="coerce").astype("Int64")
        elif tt == CastTargetType.FLOAT:
            self._df[col] = pd.to_numeric(self._df[col], errors="coerce")
        elif tt == CastTargetType.STRING:
            self._df[col] = self._df[col].astype(str)
        elif tt == CastTargetType.DATE:
            self._df[col] = pd.to_datetime(self._df[col], errors="coerce").dt.strftime("%Y-%m-%d")
        elif tt == CastTargetType.BOOL:
            self._df[col] = self._df[col].astype(bool)

        return f"cast_column({col} → {tt})"

    def _rename_value(self, action: DataAction) -> str:
        col = self._col(action)
        self._df[col] = self._df[col].replace(action.old_value, action.new_value)
        return f"rename_value({col}: '{action.old_value}' → '{action.new_value}')"

    def _normalize_format(self, action: DataAction) -> str:
        col = self._col(action)
        ft  = action.format_type

        if ft == FormatType.PHONE:
            def clean_phone(v):
                if pd.isna(v):
                    return v
                digits = PHONE_CLEAN.sub("", str(v))
                return digits[-10:] if len(digits) >= 10 else digits
            self._df[col] = self._df[col].apply(clean_phone)

        elif ft == FormatType.DATE:
            def clean_date(v):
                if pd.isna(v):
                    return v
                try:
                    return pd.Timestamp(v).strftime("%Y-%m-%d")
                except Exception:
                    return v
            self._df[col] = self._df[col].apply(clean_date)

        elif ft == FormatType.CURRENCY:
            def clean_currency(v):
                if pd.isna(v):
                    return v
                try:
                    return float(str(v).replace("$", "").replace(",", "").replace(" USD", "").strip())
                except Exception:
                    return v
            self._df[col] = self._df[col].apply(clean_currency)

        elif ft == FormatType.EMAIL:
            self._df[col] = self._df[col].str.lower().str.strip()

        return f"normalize_format({col}, {ft})"

    def _clip_values(self, action: DataAction) -> str:
        col = self._col(action)
        lo  = action.min_val
        hi  = action.max_val
        self._df[col] = pd.to_numeric(self._df[col], errors="coerce").clip(lower=lo, upper=hi)
        return f"clip_values({col}, [{lo}, {hi}])"

    def _drop_duplicates(self, action: DataAction) -> str:
        before = len(self._df)
        subset = [action.column] if action.column else None
        self._df.drop_duplicates(subset=subset, inplace=True)
        self._df.reset_index(drop=True, inplace=True)
        return f"drop_duplicates removed {before - len(self._df)} rows"

    def _flag_outliers(self, action: DataAction) -> str:
        col = self._col(action)
        s = pd.to_numeric(self._df[col], errors="coerce")
        mu, sd = s.mean(), s.std()
        outliers = ((s - mu).abs() > 3 * sd).sum()
        return f"flag_outliers({col}): {outliers} outliers (σ > 3)"

    def _repair_constraint(self, action: DataAction) -> str:
        cid = action.constraint
        df  = self._df

        if cid == "age_range":
            med = df.loc[(df["age"] >= 18) & (df["age"] <= 99), "age"].median()
            df.loc[~((df["age"] >= 18) & (df["age"] <= 99)), "age"] = med

        elif cid == "salary_positive":
            med = df.loc[df["salary"] > 0, "salary"].median()
            df.loc[~(df["salary"] > 0), "salary"] = med

        elif cid == "end_after_start":
            for i, row in df.iterrows():
                try:
                    s = pd.Timestamp(row["contract_start"])
                    e = pd.Timestamp(row["contract_end"])
                    if e < s:
                        df.loc[i, "contract_end"] = (s + pd.Timedelta(days=365)).strftime("%Y-%m-%d")
                except Exception:
                    pass

        elif cid == "dept_valid":
            df.loc[~df["department"].isin(_VALID_DEPTS), "department"] = "HR"

        elif cid == "rating_range":
            med = df.loc[(df["performance_rating"] >= 1) & (df["performance_rating"] <= 5), "performance_rating"].median()
            df.loc[~((df["performance_rating"] >= 1) & (df["performance_rating"] <= 5)), "performance_rating"] = med

        elif cid == "email_format":
            df.loc[~df["email"].apply(lambda v: bool(EMAIL_RE.match(str(v)))), "email"] = "repaired@company.org"

        else:
            raise ValueError(f"Unknown constraint id: {cid}")

        return f"repair_constraint({cid})"

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        prev_score: float,
        curr_score: float,
        action: DataAction,
        error: Optional[str],
    ) -> DataReward:
        bd = RewardBreakdown()

        delta = curr_score - prev_score
        bd.quality_improvement = float(np.clip(delta * 10.0, -2.0, 2.0))

        if error:
            bd.invalid_action_penalty = -0.3

        bd.step_penalty = -0.02

        if action.action_type == ActionType.SUBMIT:
            bd.efficiency_bonus = float(curr_score * 2.0)

        total = (
            bd.quality_improvement
            + bd.step_penalty
            + bd.invalid_action_penalty
            + bd.efficiency_bonus
        )
        return DataReward(total=round(float(total), 4), breakdown=bd)
