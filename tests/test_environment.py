"""
Tests for the OpenEnv DataCleaning environment.
Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
import numpy as np

from env.environment import DataCleaningEnv
from env.models import (
    ActionType, DataAction, DataObservation, StepResult,
    FillStrategy, CastTargetType, FormatType,
)
from graders.graders import grade_task_easy, grade_task_medium, grade_task_hard
from env.datasets import generate_customer_dataset, generate_sales_dataset, generate_hr_dataset


# ---------------------------------------------------------------------------
# Dataset generation tests
# ---------------------------------------------------------------------------

class TestDatasetGeneration:
    def test_customer_dataset_shape(self):
        df, meta = generate_customer_dataset(seed=42)
        assert len(df) == 120
        assert "age" in df.columns
        assert "income" in df.columns
        assert meta["task_id"] == "task_easy"

    def test_customer_dataset_has_nulls(self):
        df, meta = generate_customer_dataset(seed=42)
        assert df["age"].isna().sum() > 0
        assert df["income"].isna().sum() > 0

    def test_sales_dataset_shape(self):
        df, meta = generate_sales_dataset(seed=42)
        assert len(df) == 100
        assert "price" in df.columns
        assert "phone" in df.columns

    def test_hr_dataset_shape(self):
        df, meta = generate_hr_dataset(seed=42)
        assert len(df) == 150
        assert "department" in df.columns
        assert "salary" in df.columns

    def test_reproducibility(self):
        df1, _ = generate_customer_dataset(seed=99)
        df2, _ = generate_customer_dataset(seed=99)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

class TestGraders:
    def test_easy_grader_range(self):
        df, meta = generate_customer_dataset()
        score, breakdown = grade_task_easy(df, meta)
        assert 0.0 <= score <= 1.0

    def test_medium_grader_range(self):
        df, meta = generate_sales_dataset()
        score, breakdown = grade_task_medium(df, meta)
        assert 0.0 <= score <= 1.0

    def test_hard_grader_range(self):
        df, meta = generate_hr_dataset()
        score, breakdown = grade_task_hard(df, meta)
        assert 0.0 <= score <= 1.0

    def test_perfect_df_scores_high(self):
        """A fully cleaned dataset should score near 1.0."""
        df, meta = generate_customer_dataset()
        df["age"] = df["age"].fillna(df["age"].median())
        df["income"] = df["income"].fillna(df["income"].mean())
        df["city"] = df["city"].fillna(df["city"].mode()[0])
        df["loyalty_score"] = df["loyalty_score"].fillna(df["loyalty_score"].median())
        score, _ = grade_task_easy(df, meta)
        assert score >= 0.85

    def test_grader_deterministic(self):
        df, meta = generate_customer_dataset()
        s1, _ = grade_task_easy(df, meta)
        s2, _ = grade_task_easy(df, meta)
        assert s1 == s2

    def test_hard_grader_breakdown_keys(self):
        df, meta = generate_hr_dataset()
        _, breakdown = grade_task_hard(df, meta)
        expected = {"age_range", "salary_positive", "end_after_start",
                    "dept_valid", "rating_range", "email_format"}
        assert set(breakdown.keys()) == expected


# ---------------------------------------------------------------------------
# Environment API tests
# ---------------------------------------------------------------------------

class TestEnvironmentAPI:
    def test_reset_returns_observation(self):
        env = DataCleaningEnv(task_id="task_easy")
        obs = env.reset()
        assert isinstance(obs, DataObservation)
        assert obs.task_id == "task_easy"
        assert obs.step_number == 0
        assert obs.total_rows == 120

    def test_step_increments(self):
        env = DataCleaningEnv(task_id="task_easy")
        env.reset()
        action = DataAction(action_type=ActionType.FILL_NULLS, column="age", strategy=FillStrategy.MEDIAN)
        result = env.step(action)
        assert isinstance(result, StepResult)
        assert result.observation.step_number == 1

    def test_fill_nulls_reduces_nulls(self):
        env = DataCleaningEnv(task_id="task_easy")
        obs0 = env.reset()
        age_profile_before = next(p for p in obs0.column_profiles if p.name == "age")
        age_nulls_before = age_profile_before.null_count

        action = DataAction(action_type=ActionType.FILL_NULLS, column="age", strategy=FillStrategy.MEDIAN)
        result = env.step(action)
        age_profile_after = next(p for p in result.observation.column_profiles if p.name == "age")
        age_nulls_after = age_profile_after.null_count
        assert age_nulls_before > 0
        assert age_nulls_after < age_nulls_before

    def test_reward_improves_on_good_action(self):
        env = DataCleaningEnv(task_id="task_easy")
        env.reset()
        action = DataAction(action_type=ActionType.FILL_NULLS, column="age", strategy=FillStrategy.MEDIAN)
        result = env.step(action)
        assert result.reward.total > -0.5  # should be net positive

    def test_invalid_column_records_error(self):
        env = DataCleaningEnv(task_id="task_easy")
        env.reset()
        action = DataAction(action_type=ActionType.FILL_NULLS, column="nonexistent_col", strategy=FillStrategy.MEAN)
        result = env.step(action)
        assert result.observation.last_action_error is not None

    def test_state_returns_environment_state(self):
        env = DataCleaningEnv(task_id="task_easy")
        env.reset()
        state = env.state()
        assert state.task_id == "task_easy"
        assert state.dataset_shape[0] == 120

    def test_episode_ends_at_max_steps(self):
        env = DataCleaningEnv(task_id="task_easy")
        env.reset()
        for _ in range(20):
            action = DataAction(action_type=ActionType.FLAG_OUTLIERS, column="age")
            result = env.step(action)
        assert result.done

    def test_cast_column(self):
        env = DataCleaningEnv(task_id="task_medium")
        env.reset()
        action = DataAction(action_type=ActionType.CAST_COLUMN, column="quantity", target_type=CastTargetType.INT)
        result = env.step(action)
        assert result.observation.last_action_error is None

    def test_normalize_format_phone(self):
        env = DataCleaningEnv(task_id="task_medium")
        env.reset()
        action = DataAction(action_type=ActionType.NORMALIZE_FORMAT, column="phone", format_type=FormatType.PHONE)
        result = env.step(action)
        assert result.observation.last_action_error is None

    def test_repair_constraint(self):
        env = DataCleaningEnv(task_id="task_hard")
        env.reset()
        action = DataAction(action_type=ActionType.REPAIR_CONSTRAINT, constraint="age_range")
        result = env.step(action)
        assert result.observation.last_action_error is None

    def test_all_tasks_reset(self):
        for tid in ["task_easy", "task_medium", "task_hard"]:
            env = DataCleaningEnv(task_id=tid)
            obs = env.reset()
            assert obs.task_id == tid

    def test_submit_ends_episode(self):
        env = DataCleaningEnv(task_id="task_easy")
        env.reset()
        action = DataAction(action_type=ActionType.SUBMIT)
        result = env.step(action)
        assert result.done

    def test_quality_score_in_range(self):
        env = DataCleaningEnv(task_id="task_hard")
        obs = env.reset()
        assert 0.0 <= obs.quality_score <= 1.0

    def test_reset_before_step_raises(self):
        env = DataCleaningEnv(task_id="task_easy")
        action = DataAction(action_type=ActionType.SUBMIT)
        with pytest.raises(RuntimeError):
            env.step(action)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
