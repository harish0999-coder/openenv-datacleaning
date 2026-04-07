"""
inference.py — Baseline inference script for OpenEnv DataCleaning Environment.

Structured stdout output:
    [START] task=TASK_NAME
    [STEP] step=N reward=R
    [END] task=TASK_NAME score=S steps=N

Environment variables:
    API_BASE_URL  - LLM API endpoint (default: https://api.openai.com/v1)
    MODEL_NAME    - Model identifier  (default: gpt-4o-mini)
    HF_TOKEN      - API key
    ENV_URL       - Environment base URL (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     os.environ.get("OPENAI_API_KEY", ""))
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

MAX_STEPS    = 15
TASKS        = ["task_easy", "task_medium", "task_hard"]
FALLBACK_ACTION = json.dumps({"action_type": "submit"})

# ---------------------------------------------------------------------------
# OpenAI client (required by competition rules)
# ---------------------------------------------------------------------------

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str, seed: int = 42) -> Dict:
    r = httpx.post(
        f"{ENV_URL}/reset",
        params={"task_id": task_id, "seed": seed},
        timeout=30,
    )
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
# LLM prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert data quality agent.
Given a data cleaning environment observation, choose the best action.

Available action_type values:
  fill_nulls        needs: column, strategy (mean/median/mode/ffill/bfill/constant/drop)
  cast_column       needs: column, target_type (int/float/string/date)
  normalize_format  needs: column, format_type (phone/date/currency/email)
  rename_value      needs: column, old_value, new_value
  clip_values       needs: column, min_val, max_val
  drop_duplicates   needs: column (optional)
  repair_constraint needs: constraint (age_range/salary_positive/end_after_start/dept_valid/rating_range/email_format)
  submit            ends episode

Respond ONLY with a single JSON object. No markdown, no explanation.
Example: {"action_type": "fill_nulls", "column": "age", "strategy": "median"}"""


def build_prompt(obs: Dict) -> str:
    profiles = "\n".join(
        f"  {p['name']}: {p['null_count']} nulls  issues={p['issues']}"
        for p in obs.get("column_profiles", [])
    )
    violations = "\n".join(
        f"  {v['constraint_id']}: {len(v['affected_rows'])} rows"
        for v in obs.get("constraint_violations", [])
    ) or "  none"
    return (
        f"task={obs['task_id']} step={obs['step_number']}/{obs['max_steps']} "
        f"quality={obs['quality_score']} issues={obs['issues_remaining']}\n"
        f"columns:\n{profiles}\nviolations:\n{violations}"
    )


def parse_model_action(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
        raise

# ---------------------------------------------------------------------------
# Single task episode
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> Dict[str, Any]:
    # ── required structured output ─────────────────────────────────────────
    print(f"[START] task={task_id}", flush=True)

    obs = env_reset(task_id)
    episode_reward = 0.0
    step = 0

    for step in range(1, MAX_STEPS + 1):
        # Get LLM action
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(obs)},
                ],
                max_tokens=128,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or FALLBACK_ACTION
        except Exception as exc:
            failure_msg = f"Model request failed ({exc}). Using fallback action."
            print(failure_msg, flush=True)
            raw = FALLBACK_ACTION

        try:
            action = parse_model_action(raw)
        except Exception:
            action = json.loads(FALLBACK_ACTION)

        # Step environment
        try:
            result = env_step(action)
        except Exception as exc:
            print(f"env_step failed: {exc}", flush=True)
            break

        obs    = result["observation"]
        reward = result["reward"]["total"]
        episode_reward += reward

        # ── required [STEP] output ─────────────────────────────────────────
        print(f"[STEP] step={step} reward={reward:.4f}", flush=True)

        if result.get("done") or obs.get("done"):
            print("Episode complete.", flush=True)
            break

        if obs["quality_score"] >= 0.98:
            break

    else:
        print(f"Reached max steps ({MAX_STEPS}).", flush=True)

    final_score = env_state()["quality_score"]

    # ── required [END] output ──────────────────────────────────────────────
    print(f"[END] task={task_id} score={final_score:.4f} steps={step}", flush=True)

    return {
        "task_id":        task_id,
        "final_score":    final_score,
        "episode_reward": round(episode_reward, 4),
        "steps_taken":    step,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN / OPENAI_API_KEY not set.", flush=True)
        sys.exit(1)

    # Verify environment is reachable
    try:
        r = httpx.get(f"{ENV_URL}/health", timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach environment at {ENV_URL}: {e}", flush=True)
        sys.exit(1)

    results = []
    for task_id in TASKS:
        result = run_task(task_id)
        results.append(result)

    # Save results JSON
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
