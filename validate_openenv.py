"""
validate_openenv.py — Pre-submission validation script for OpenEnv DataCleaning.

Checks:
  1. openenv.yaml is valid and contains required fields
  2. All typed models importable (Observation, Action, Reward)
  3. Environment reset() / step() / state() work correctly
  4. 3+ tasks each produce scores in [0.0, 1.0]
  5. Graders are deterministic (same input → same output twice)
  6. Dockerfile exists
  7. inference.py exists at project root
  8. requirements.txt exists

Run:  python validate_openenv.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import yaml

PASS = "  \033[32m[PASS]\033[0m"
FAIL = "  \033[31m[FAIL]\033[0m"
INFO = "  \033[34m[INFO]\033[0m"

errors = 0


def check(label: str, condition: bool, detail: str = ""):
    global errors
    if condition:
        print(f"{PASS} {label}" + (f" — {detail}" if detail else ""))
    else:
        print(f"{FAIL} {label}" + (f" — {detail}" if detail else ""))
        errors += 1


def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ── 1. openenv.yaml ────────────────────────────────────────────────────────

section("1 · openenv.yaml")

try:
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)

    required_keys = ["name", "version", "description", "tasks", "api"]
    for k in required_keys:
        check(f"openenv.yaml has key '{k}'", k in spec)

    tasks = spec.get("tasks", [])
    check("3+ tasks defined in yaml", len(tasks) >= 3, f"found {len(tasks)}")

    diffs = [t.get("difficulty") for t in tasks]
    for d in ["easy", "medium", "hard"]:
        check(f"Task with difficulty='{d}'", d in diffs)

except Exception as e:
    check("openenv.yaml readable", False, str(e))

# ── 2. Typed models ────────────────────────────────────────────────────────

section("2 · Typed Pydantic models")

try:
    from env.models import DataObservation, DataAction, DataReward, StepResult, EnvironmentState
    check("DataObservation importable", True)
    check("DataAction importable", True)
    check("DataReward importable", True)
    check("StepResult importable", True)
    check("EnvironmentState importable", True)

    # Instantiate each
    from env.models import ActionType, FillStrategy
    a = DataAction(action_type=ActionType.FILL_NULLS, column="age", strategy=FillStrategy.MEDIAN)
    check("DataAction instantiates", True)

except Exception as e:
    check("Models importable", False, str(e))
    traceback.print_exc()

# ── 3. Environment API ─────────────────────────────────────────────────────

section("3 · Environment API (reset / step / state)")

try:
    from env.environment import DataCleaningEnv
    from env.models import DataAction, ActionType, FillStrategy

    env = DataCleaningEnv(task_id="task_easy")

    # reset
    obs = env.reset()
    check("reset() returns DataObservation", obs.__class__.__name__ == "DataObservation")
    check("reset() step_number == 0", obs.step_number == 0)
    check("reset() total_rows > 0", obs.total_rows > 0)

    # step
    action = DataAction(action_type=ActionType.FILL_NULLS, column="age", strategy=FillStrategy.MEDIAN)
    result = env.step(action)
    check("step() returns StepResult", result.__class__.__name__ == "StepResult")
    check("step() observation present", result.observation is not None)
    check("step() reward present", result.reward is not None)
    check("step() done is bool", isinstance(result.done, bool))
    check("step() increments step_number", result.observation.step_number == 1)

    # state
    st = env.state()
    check("state() returns EnvironmentState", st.__class__.__name__ == "EnvironmentState")
    check("state() has quality_score", 0.0 <= st.quality_score <= 1.0)

except Exception as e:
    check("Environment API works", False, str(e))
    traceback.print_exc()

# ── 4. All 3 tasks + graders ───────────────────────────────────────────────

section("4 · Tasks & graders (scores in [0.0, 1.0])")

try:
    from env.environment import DataCleaningEnv
    from env.models import DataAction, ActionType

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        env = DataCleaningEnv(task_id=task_id)
        obs = env.reset()
        score = obs.quality_score
        check(f"{task_id} reset score in [0,1]", 0.0 <= score <= 1.0, f"{score:.4f}")

        # Take a few steps and verify reward range
        rewards = []
        for _ in range(3):
            a = DataAction(action_type=ActionType.FLAG_OUTLIERS, column="age" if task_id != "task_medium" else "price")
            r = env.step(a)
            rewards.append(r.reward.total)

        check(f"{task_id} reward is numeric", all(isinstance(rw, float) for rw in rewards))
        check(f"{task_id} quality_score in [0,1] after steps", 0.0 <= r.observation.quality_score <= 1.0)

except Exception as e:
    check("Tasks+graders pass", False, str(e))
    traceback.print_exc()

# ── 5. Grader determinism ──────────────────────────────────────────────────

section("5 · Grader determinism")

try:
    from graders.graders import grade_task_easy, grade_task_medium, grade_task_hard
    from env.datasets import generate_customer_dataset, generate_sales_dataset, generate_hr_dataset

    df1, m1 = generate_customer_dataset(seed=42)
    s1a, _ = grade_task_easy(df1, m1)
    s1b, _ = grade_task_easy(df1, m1)
    check("task_easy grader deterministic", s1a == s1b, f"{s1a} == {s1b}")

    df2, m2 = generate_sales_dataset(seed=42)
    s2a, _ = grade_task_medium(df2, m2)
    s2b, _ = grade_task_medium(df2, m2)
    check("task_medium grader deterministic", s2a == s2b, f"{s2a} == {s2b}")

    df3, m3 = generate_hr_dataset(seed=42)
    s3a, _ = grade_task_hard(df3, m3)
    s3b, _ = grade_task_hard(df3, m3)
    check("task_hard grader deterministic", s3a == s3b, f"{s3a} == {s3b}")

except Exception as e:
    check("Grader determinism", False, str(e))
    traceback.print_exc()

# ── 6. Required files ──────────────────────────────────────────────────────

section("6 · Required project files")

for fname in ["Dockerfile", "inference.py", "requirements.txt", "openenv.yaml", "main.py"]:
    check(f"{fname} exists", os.path.isfile(fname))

# ── 7. Reward shape (partial progress signal) ──────────────────────────────

section("7 · Reward shaping (partial progress signal)")

try:
    from env.environment import DataCleaningEnv
    from env.models import DataAction, ActionType, FillStrategy

    env = DataCleaningEnv(task_id="task_easy")
    env.reset()

    scores = []
    for col in ["age", "income", "city", "loyalty_score"]:
        a = DataAction(action_type=ActionType.FILL_NULLS, column=col, strategy=FillStrategy.MEDIAN)
        r = env.step(a)
        scores.append(r.observation.quality_score)

    check("Quality monotonically improves on good actions", scores[-1] > scores[0],
          f"start={scores[0]:.3f} end={scores[-1]:.3f}")
    check("Reward breakdown has multiple components",
          len([f for f in r.reward.breakdown.model_fields]) >= 4)

except Exception as e:
    check("Reward shaping", False, str(e))
    traceback.print_exc()

# ── Summary ────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
if errors == 0:
    print("  \033[32m✓ ALL CHECKS PASSED — ready for submission!\033[0m")
else:
    print(f"  \033[31m✗ {errors} check(s) FAILED — fix before submitting.\033[0m")
print(f"{'='*55}\n")

sys.exit(0 if errors == 0 else 1)
