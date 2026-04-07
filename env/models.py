"""
OpenEnv DataCleaning Environment — Core Typed Models
Implements the full OpenEnv spec: Observation, Action, Reward
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    FILL_NULLS        = "fill_nulls"
    DROP_ROWS         = "drop_rows"
    CAST_COLUMN       = "cast_column"
    RENAME_VALUE      = "rename_value"
    NORMALIZE_FORMAT  = "normalize_format"
    CLIP_VALUES       = "clip_values"
    DROP_DUPLICATES   = "drop_duplicates"
    FLAG_OUTLIERS     = "flag_outliers"
    REPAIR_CONSTRAINT = "repair_constraint"
    SUBMIT            = "submit"


class FillStrategy(str, Enum):
    MEAN        = "mean"
    MEDIAN      = "median"
    MODE        = "mode"
    FFILL       = "ffill"
    BFILL       = "bfill"
    CONSTANT    = "constant"
    DROP        = "drop"


class CastTargetType(str, Enum):
    INT     = "int"
    FLOAT   = "float"
    STRING  = "string"
    DATE    = "date"
    BOOL    = "bool"


class FormatType(str, Enum):
    PHONE     = "phone"
    EMAIL     = "email"
    DATE      = "date"
    POSTCODE  = "postcode"
    CURRENCY  = "currency"


# ---------------------------------------------------------------------------
# Action Model
# ---------------------------------------------------------------------------

class DataAction(BaseModel):
    """
    An action the agent can take to clean/transform the dataset.

    action_type: Which operation to perform.
    column:      Target column name (required for most actions).
    value:       Constant fill value (for FILL_NULLS with CONSTANT strategy).
    strategy:    Fill strategy for FILL_NULLS.
    target_type: Target dtype for CAST_COLUMN.
    old_value:   Value to replace in RENAME_VALUE.
    new_value:   Replacement value in RENAME_VALUE.
    format_type: Format to normalise to in NORMALIZE_FORMAT.
    min_val:     Lower bound for CLIP_VALUES.
    max_val:     Upper bound for CLIP_VALUES.
    constraint:  Constraint id to repair in REPAIR_CONSTRAINT.
    row_indices: List of row indices to operate on (optional; empty = all rows).
    """

    action_type: ActionType
    column:      Optional[str]  = None
    value:       Optional[Any]  = None
    strategy:    Optional[FillStrategy]    = None
    target_type: Optional[CastTargetType] = None
    old_value:   Optional[Any]  = None
    new_value:   Optional[Any]  = None
    format_type: Optional[FormatType]     = None
    min_val:     Optional[float] = None
    max_val:     Optional[float] = None
    constraint:  Optional[str]  = None
    row_indices: Optional[List[int]] = Field(default_factory=list)

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Observation Model
# ---------------------------------------------------------------------------

class ColumnProfile(BaseModel):
    name:         str
    dtype:        str
    null_count:   int
    null_pct:     float
    unique_count: int
    sample_values: List[Any]
    has_format_issues: bool = False
    issues: List[str] = Field(default_factory=list)


class ConstraintViolation(BaseModel):
    constraint_id:  str
    description:    str
    affected_rows:  List[int]
    severity:       str  # "error" | "warning"


class DataObservation(BaseModel):
    """
    Observation returned by step() and reset().

    task_id:           Current task identifier.
    step_number:       Current step within the episode.
    max_steps:         Episode step limit.
    total_rows:        Number of rows in the dataset.
    total_columns:     Number of columns.
    column_profiles:   Per-column quality summary.
    constraint_violations: List of active business-rule violations.
    recent_actions:    Summary of the last N actions taken.
    quality_score:     Current data quality score (0.0–1.0).
    issues_remaining:  Count of unresolved issues.
    last_action_error: Error message if the last action failed, else None.
    done:              Whether the episode has ended.
    info:              Arbitrary extra info dict.
    """

    task_id:            str
    step_number:        int
    max_steps:          int
    total_rows:         int
    total_columns:      int
    column_profiles:    List[ColumnProfile]
    constraint_violations: List[ConstraintViolation] = Field(default_factory=list)
    recent_actions:     List[str] = Field(default_factory=list)
    quality_score:      float = 0.0
    issues_remaining:   int = 0
    last_action_error:  Optional[str] = None
    done:               bool = False
    info:               Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward Model
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    quality_improvement:   float = 0.0
    null_fix_reward:       float = 0.0
    format_fix_reward:     float = 0.0
    constraint_fix_reward: float = 0.0
    efficiency_bonus:      float = 0.0
    step_penalty:          float = 0.0
    invalid_action_penalty: float = 0.0


class DataReward(BaseModel):
    """
    Reward signal returned alongside each observation.

    total:     Scalar reward for this step (can be negative).
    breakdown: Itemised reward components for interpretability.
    """

    total:     float
    breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)


# ---------------------------------------------------------------------------
# Step Result (returned by /step)
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: DataObservation
    reward:      DataReward
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment State (returned by /state)
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    task_id:        str
    step_number:    int
    quality_score:  float
    issues_remaining: int
    episode_reward: float
    done:           bool
    dataset_shape:  List[int]   # [rows, cols]
    active_task:    Dict[str, Any]
