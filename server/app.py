"""
server/app.py — OpenEnv-core compatible server entry point.
Uses openenv-core create_fastapi_app to register the DataCleaning environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv_core import (
    Action,
    Environment,
    Observation,
    create_fastapi_app,
)
from pydantic import Field

from env.environment import DataCleaningEnv as _InternalEnv
from env.models import ActionType


# ---------------------------------------------------------------------------
# OpenEnv-core compatible Action
# ---------------------------------------------------------------------------

class DataCleaningAction(Action):
    """Action for the DataCleaning environment."""

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    action_type: str = Field(default="submit", description="Type of cleaning action")
    column: Optional[str] = Field(default=None, description="Target column")
    strategy: Optional[str] = Field(default=None, description="Fill strategy")
    target_type: Optional[str] = Field(default=None, description="Cast target type")
    format_type: Optional[str] = Field(default=None, description="Format type")
    old_value: Optional[Any] = Field(default=None, description="Value to replace")
    new_value: Optional[Any] = Field(default=None, description="Replacement value")
    min_val: Optional[float] = Field(default=None, description="Minimum clip value")
    max_val: Optional[float] = Field(default=None, description="Maximum clip value")
    constraint: Optional[str] = Field(default=None, description="Constraint ID")
    value: Optional[Any] = Field(default=None, description="Constant fill value")
    row_indices: Optional[List[int]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# OpenEnv-core compatible Observation
# ---------------------------------------------------------------------------

class DataCleaningObservation(Observation):
    """Observation for the DataCleaning environment."""

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    task_id: str = Field(default="task_easy")
    step_number: int = Field(default=0)
    max_steps: int = Field(default=20)
    total_rows: int = Field(default=0)
    total_columns: int = Field(default=0)
    quality_score: float = Field(default=0.0)
    issues_remaining: int = Field(default=0)
    last_action_error: Optional[str] = Field(default=None)
    column_profiles: List[Dict[str, Any]] = Field(default_factory=list)
    constraint_violations: List[Dict[str, Any]] = Field(default_factory=list)
    recent_actions: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# OpenEnv-core Environment wrapper
# ---------------------------------------------------------------------------

class DataCleaningEnvironment(Environment):
    """OpenEnv-core compatible DataCleaning environment."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env = _InternalEnv(task_id="task_easy", seed=42)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "task_easy",
        **kwargs,
    ) -> DataCleaningObservation:
        self._env = _InternalEnv(task_id=task_id, seed=seed or 42)
        obs = self._env.reset()
        return self._convert_obs(obs)

    def step(
        self,
        action: DataCleaningAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> DataCleaningObservation:
        from env.models import DataAction
        internal_action = DataAction(
            action_type=ActionType(action.action_type),
            column=action.column,
            strategy=action.strategy,
            target_type=action.target_type,
            format_type=action.format_type,
            old_value=action.old_value,
            new_value=action.new_value,
            min_val=action.min_val,
            max_val=action.max_val,
            constraint=action.constraint,
            value=action.value,
            row_indices=action.row_indices or [],
        )
        result = self._env.step(internal_action)
        obs = self._convert_obs(result.observation)
        obs.reward = result.reward.total
        obs.done = result.done
        return obs

    def _convert_obs(self, obs) -> DataCleaningObservation:
        profiles = [p.model_dump() for p in obs.column_profiles]
        violations = [v.model_dump() for v in obs.constraint_violations]
        return DataCleaningObservation(
            task_id=obs.task_id,
            step_number=obs.step_number,
            max_steps=obs.max_steps,
            total_rows=obs.total_rows,
            total_columns=obs.total_columns,
            quality_score=obs.quality_score,
            issues_remaining=obs.issues_remaining,
            last_action_error=obs.last_action_error,
            column_profiles=profiles,
            constraint_violations=violations,
            recent_actions=obs.recent_actions,
            done=obs.done,
            reward=None,
        )


# ---------------------------------------------------------------------------
# Create FastAPI app via openenv-core
# ---------------------------------------------------------------------------

app = create_fastapi_app(
    env=DataCleaningEnvironment,
    action_cls=DataCleaningAction,
    observation_cls=DataCleaningObservation,
)


# ---------------------------------------------------------------------------
# main() entry point — required by OpenEnv spec
# ---------------------------------------------------------------------------

def main():
    """Server entry point callable by [project.scripts]."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
