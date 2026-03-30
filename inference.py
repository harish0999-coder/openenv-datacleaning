"""
inference.py — Baseline inference script for OpenEnv DataCleaning Environment
Uses OpenAI client to run an LLM agent against all three tasks.

Usage:
    python inference.py

Required environment variables:
    API_BASE_URL   e.g. https://api.openai.com/v1
    MODEL_NAME     e.g. gpt-4o-mini
    HF_TOKEN       Your Hugging Face / OpenAI API key
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     os.environ.get("OPENAI_API_KEY", ""))

ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

MAX_STEPS    = 15          # per task (stay under 20-min wall)
TASKS        = ["task_easy", "task_medium", "task_hard"]
FALLBACK_ACTION = json.dumps({"action_type": "submit"})

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Env client helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str, seed: int = 42) -> Dict:
    r = httpx.post(f"{ENV_URL}/reset", params={"task_id": task_id, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: Dict) -> Dict:
    r = httpx.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def env_state() -> Dict:
    r = httpx.get(f"{ENV_URL}/state", timeout=30)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert data quality agent. You will be given a structured observation
from a data cleaning environment and must decide which action to take to improve
data quality.

Available action types:
- fill_nulls       : fill missing values  (needs column, strategy: mean/median/mode/ffill/bfill/constant/drop)
- cast_column      : fix dtype            (needs column, target_type: int/float/string/date)
- normalize_format : fix formats          (needs column, format_type: phone/date/currency/email)
- rename_value     : replace a value      (needs column, old_value, new_value)
- clip_values      : clamp numeric range  (needs column, min_val, max_val)
- drop_duplicates  : remove duplicates    (needs column or omit for all)
- flag_outliers    : flag statistical outliers (needs column)
- repair_constraint: fix business rule violation (needs constraint: age_range/salary_positive/end_after_start/dept_valid/rating_range/email_format)
- submit           : end episode

Rules:
1. Fix the most impactful issues first (most null values, most violations).
2. Respond ONLY with a single JSON object — no explanation, no markdown, no backticks.
3. Example: {"action_type": "fill_nulls", "column": "age", "strategy": "median"}
""".strip()


def build_user_prompt(obs: Dict) -> str:
    profiles = obs.get("column_profiles", [])
    col_summary = "\n".join(
        f"  - {p['name']}: {p['null_count']} nulls, issues={p['issues']}"
        for p in profiles
    )
    violations = obs.get("constraint_violations", [])
    viol_summary = "\n".join(
        f"  - {v['constraint_id']}: {v['description']} ({len(v['affected_rows'])} rows)"
        for v in violations
    ) or "  None"

    return f"""
Task: {obs['task_id']}
Step: {obs['step_number']} / {obs['max_steps']}
Quality score: {obs['quality_score']}
Issues remaining: {obs['issues_remaining']}

Column profiles:
{col_summary}

Constraint violations:
{viol_summary}

Recent actions:
  {obs.get('recent_actions', [])}

Choose the best next action as a JSON object.
""".strip()

# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Dict:
    text = text.strip()
    # strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to extract first { ... }
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        raise

# ---------------------------------------------------------------------------
# Single task run
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id}")
    print(f"{'='*60}")

    obs = env_reset(task_id)
    print(f"  Initial quality: {obs['quality_score']}  Issues: {obs['issues_remaining']}")

    episode_reward = 0.0
    history = []

    for step in range(1, MAX_STEPS + 1):
        user_prompt = build_user_prompt(obs)

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=256,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or FALLBACK_ACTION
        except Exception as exc:
            print(f"  [WARN] LLM call failed at step {step}: {exc}")
            raw = FALLBACK_ACTION

        try:
            action = parse_action(raw)
        except Exception:
            print(f"  [WARN] JSON parse failed; using fallback. Raw: {raw[:80]}")
            action = json.loads(FALLBACK_ACTION)

        print(f"  Step {step:02d}: {action}")

        try:
            result = env_step(action)
        except Exception as exc:
            print(f"  [ERR] env_step failed: {exc}")
            break

        obs = result["observation"]
        rwd = result["reward"]
        episode_reward += rwd["total"]

        history.append({
            "step":   step,
            "action": action,
            "reward": rwd["total"],
            "quality": obs["quality_score"],
        })

        print(f"         quality={obs['quality_score']}  reward={rwd['total']:+.3f}  issues={obs['issues_remaining']}")

        if result.get("done") or obs.get("done"):
            print(f"  Episode done at step {step}.")
            break

        if obs["quality_score"] >= 0.98:
            print("  Perfect quality achieved!")
            break

    final_state = env_state()
    final_score = final_state["quality_score"]

    return {
        "task_id":       task_id,
        "final_score":   final_score,
        "episode_reward": round(episode_reward, 4),
        "steps_taken":   step,
        "history":       history,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  OpenEnv DataCleaning — Baseline Inference")
    print(f"  Model     : {MODEL_NAME}")
    print(f"  API base  : {API_BASE_URL}")
    print(f"  Env URL   : {ENV_URL}")
    print("="*60)

    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN / OPENAI_API_KEY not set.")
        sys.exit(1)

    # Verify env is up
    try:
        r = httpx.get(f"{ENV_URL}/health", timeout=10)
        r.raise_for_status()
        print(f"\n  Environment health: {r.json()['status']}")
    except Exception as e:
        print(f"[ERROR] Cannot reach environment at {ENV_URL}: {e}")
        sys.exit(1)

    results = []
    t0 = time.time()

    for task_id in TASKS:
        result = run_task(task_id)
        results.append(result)

    elapsed = time.time() - t0

    print("\n" + "="*60)
    print("  BASELINE SCORES")
    print("="*60)
    for r in results:
        grade = "✓" if r["final_score"] >= 0.7 else "✗"
        print(
            f"  {grade} {r['task_id']:15s}  "
            f"score={r['final_score']:.4f}  "
            f"reward={r['episode_reward']:+.3f}  "
            f"steps={r['steps_taken']}"
        )

    avg = sum(r["final_score"] for r in results) / len(results)
    print(f"\n  Average score : {avg:.4f}")
    print(f"  Total runtime : {elapsed:.1f}s")
    print("="*60)

    # Write results JSON
    with open("baseline_results.json", "w") as f:
        json.dump({"results": results, "average_score": avg, "elapsed_s": elapsed}, f, indent=2)
    print("\n  Results saved to baseline_results.json")

    return results


if __name__ == "__main__":
    main()
